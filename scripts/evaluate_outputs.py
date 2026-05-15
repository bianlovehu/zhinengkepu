"""
Heuristic quality checks for generated contest answers.

Usage:
    python scripts/evaluate_outputs.py --answers submission.csv
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


BAD_PHRASES = [
    "请稍后", "正在核实", "无法回答", "暂时无法", "耐心等待", "后台核实",
    "咨询人数较多", "LLM调用失败", "LLM服务暂不可用", "不支持的LLM Provider",
]
PICTURE_KEYWORDS = ["指示灯", "闪烁", "表带", "尺寸", "按键", "接口", "安装", "拆卸", "更换", "label", "screen", "button"]
STEP_KEYWORDS = ["如何", "怎么", "步骤", "安装", "清洁", "更换", "设置", "启动", "关闭", "调节", "how to", "what should i do"]
PRODUCT_KEYWORDS = [
    "电钻", "充电器", "健身追踪器", "表带", "空调", "洗碗机", "发电机", "吹风机",
    "摩托艇", "水泵", "温控器", "VR", "相机", "airfryer", "boat", "jetski", "camera",
    "earphones", "lawn mower", "microwave", "motherboard",
]


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def question_map(path: Path) -> Dict[str, str]:
    return {row["id"]: row["question"] for row in load_csv(path)}


def is_multi_question(question: str) -> bool:
    marks = question.count("？") + question.count("?")
    connectors = ["同时", "另外", "以及", "并且", "而且", "需要", "多久", "运费", "费用"]
    return marks > 1 or sum(1 for kw in connectors if kw in question) >= 2


def has_numbered_structure(answer: str) -> bool:
    return bool(re.search(r"(^|\n)\s*(\*\*)?\s*(\d+[\.\、]|[（(]\d+[）)])", answer))


def evaluate_row(question: str, answer: str) -> List[str]:
    issues = []
    compact_answer = answer.strip()
    lower_q = question.lower()
    lower_a = compact_answer.lower()

    if not compact_answer:
        issues.append("empty_answer")
    if len(compact_answer) < 20:
        issues.append("too_short")
    if any(phrase in compact_answer for phrase in BAD_PHRASES):
        issues.append("bad_placeholder")
    if "internal error" in lower_a or "traceback" in lower_a:
        issues.append("api_error_text")
    if is_multi_question(question) and not has_numbered_structure(compact_answer):
        issues.append("multi_question_not_numbered")
    if any(kw.lower() in lower_q for kw in STEP_KEYWORDS) and len(compact_answer) > 40 and not has_numbered_structure(compact_answer):
        issues.append("steps_not_numbered")
    generic_policy_pic_false_positive = (
        any(kw in question for kw in ["上门安装", "尺寸差价", "更大的尺寸", "质保", "维修", "配件费"])
        and not any(kw in question for kw in ["表带", "健身追踪器", "指示灯", "遥控器", "滤网"])
    )
    if (
        any(kw.lower() in lower_q for kw in PICTURE_KEYWORDS)
        and "<pic>" not in lower_a
        and not generic_policy_pic_false_positive
    ):
        issues.append("missing_pic_marker")

    q_products = [kw for kw in PRODUCT_KEYWORDS if kw.lower() in lower_q]
    if q_products and not any(kw.lower() in lower_a for kw in q_products[:3]):
        issues.append("missing_product_keyword")
    if len(compact_answer) > 1200:
        issues.append("too_long")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated answers with heuristic bad-answer rules")
    parser.add_argument("--questions", default="question_public.csv")
    parser.add_argument("--answers", default="submission.csv")
    parser.add_argument("--output", default="submission_quality_report.csv")
    parser.add_argument("--json-output", default="submission_quality_report.json")
    args = parser.parse_args()

    questions = question_map(Path(args.questions))
    answers = load_csv(Path(args.answers))
    report = []
    issue_counts: Dict[str, int] = {}

    for row in answers:
        question_id = row["id"]
        question = questions.get(question_id, "")
        answer = row.get("ret", "")
        issues = evaluate_row(question, answer)
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        if issues:
            report.append({
                "id": question_id,
                "issues": ";".join(issues),
                "question": question,
                "answer": answer,
            })

    with Path(args.output).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "issues", "question", "answer"])
        writer.writeheader()
        writer.writerows(report)

    summary = {"total": len(answers), "bad": len(report), "issue_counts": issue_counts}
    Path(args.json_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote report to {args.output}")


if __name__ == "__main__":
    main()
