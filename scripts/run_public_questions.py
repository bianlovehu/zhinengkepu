"""
批量调用 /chat API 生成公开题提交文件。

用法：
    python scripts/run_public_questions.py --limit 20
    python scripts/run_public_questions.py --input question_public.csv --output submission.csv --concurrency 4 --resume
"""
import argparse
import asyncio
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from config import get_settings


BAD_FALLBACK = (
    "非常抱歉，当前问题需要结合订单状态、商品情况和售后规则进一步核实。"
    "建议您提供订单号、商品照片或视频、物流信息和具体诉求，客服会按退换货、维修或补发规则为您处理。"
)


def api_base_url(settings) -> str:
    host = "127.0.0.1" if settings.API_HOST in {"0.0.0.0", "::"} else settings.API_HOST
    return f"http://{host}:{settings.API_PORT}"


def load_questions(path: Path, limit: int | None) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def load_existing(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row.get("ret", "") for row in csv.DictReader(f)}


async def ask_one(
    client: httpx.AsyncClient,
    row: Dict[str, str],
    base_url: str,
    token: str,
    semaphore: asyncio.Semaphore,
    retries: int,
) -> Dict[str, str]:
    async with semaphore:
        question_id = row["id"]
        question = row["question"]
        started = time.time()
        last_error = ""
        for attempt in range(retries + 1):
            try:
                response = await client.post(
                    f"{base_url.rstrip('/')}/chat/",
                    json={"question": question, "session_id": f"public_{question_id}"},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                payload = response.json()
                answer = payload.get("data", {}).get("answer", "").strip()
                return {
                    "id": question_id,
                    "ret": answer or BAD_FALLBACK,
                    "question": question,
                    "elapsed": f"{time.time() - started:.3f}",
                    "error": "",
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
        return {
            "id": question_id,
            "ret": BAD_FALLBACK,
            "question": question,
            "elapsed": f"{time.time() - started:.3f}",
            "error": last_error,
        }


def write_submission(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "ret": clean_submission_answer(row["ret"])})


def clean_submission_answer(answer: str) -> str:
    """清理影响提交观感的 Markdown 装饰符。"""
    answer = answer.replace("**", "")
    answer = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", answer)
    answer = re.sub(r"(?m)^\s*[-*]\s+", "- ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def write_debug(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_checkpoint(
    output_path: Path,
    debug_path: Path | None,
    questions: List[Dict[str, str]],
    by_id: Dict[str, Dict[str, str]],
) -> None:
    ordered_partial = [by_id[row["id"]] for row in questions if row["id"] in by_id]
    write_submission(output_path, ordered_partial)
    if debug_path:
        write_debug(debug_path, ordered_partial)


async def main_async(args) -> None:
    settings = get_settings()
    base_url = args.base_url or api_base_url(settings)
    token = args.token or settings.API_TOKEN
    questions = load_questions(Path(args.input), args.limit)
    existing = load_existing(Path(args.output)) if args.resume else {}

    completed = [
        {"id": row["id"], "ret": existing[row["id"]], "question": row["question"], "elapsed": "0", "error": "resumed"}
        for row in questions
        if row["id"] in existing and existing[row["id"]].strip()
    ]
    pending = [row for row in questions if row["id"] not in existing or not existing[row["id"]].strip()]
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    by_id = {row["id"]: row for row in completed}
    output_path = Path(args.output)
    debug_path = Path(args.debug_output) if args.debug_output else None

    if completed:
        write_checkpoint(output_path, debug_path, questions, by_id)
        print(f"resumed {len(completed)} rows from {args.output}")

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            ask_one(client, row, base_url, token, semaphore, args.retries)
            for row in pending
        ]
        for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            by_id[result["id"]] = result
            if idx % args.progress_every == 0 or idx == len(tasks):
                print(f"finished {idx}/{len(tasks)} pending, id={result['id']}, error={result['error'][:80]}")
            if idx % args.checkpoint_every == 0 or idx == len(tasks):
                write_checkpoint(output_path, debug_path, questions, by_id)
                print(f"checkpoint wrote {len(by_id)}/{len(questions)} rows to {args.output}")

    ordered = [by_id[row["id"]] for row in questions]
    write_submission(output_path, ordered)
    if debug_path:
        write_debug(debug_path, ordered)
    print(f"wrote {len(ordered)} rows to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate contest submission CSV from /chat API")
    parser.add_argument("--input", default="question_public.csv")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--debug-output", default="submission_debug.jsonl")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
