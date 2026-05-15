"""
Question routing and query expansion for contest customer-service tasks.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List


POLICY_SOURCE = "客服政策手册.txt"


@dataclass
class RouteResult:
    route: str
    sub_questions: List[str] = field(default_factory=list)
    keywords: Dict[str, List[str]] = field(default_factory=dict)
    needs_images: bool = False


class QuestionRouter:
    """Route public benchmark questions to policy/manual/mixed retrieval."""

    POLICY_KEYWORDS = [
        "7天", "七天", "无理由", "退货", "换货", "退款", "运费", "发票", "抬头",
        "重开", "物流", "快递", "发货", "待揽收", "丢失", "丢件", "乡镇", "配送",
        "包装", "破损", "少发", "漏发", "错发", "瑕疵", "划痕", "二手", "拆封",
        "污渍", "假货", "正品", "虚假宣传", "投诉", "辱骂", "赔偿", "质保", "保修",
        "维修", "人为损坏", "售后", "补发", "补寄", "临期", "过期", "上门安装",
        "配件费", "以旧换新", "试用装", "终身维修", "售后保障卡", "纸质版说明书",
        "保质期", "生产日期", "受潮", "健康", "坏了", "坏", "使用一次",
        "尺寸差价", "更大的尺寸", "差价", "详情页", "无线充电", "续航", "翻新机",
        "免费安装", "上门检修", "拉回仓库", "维修时间",
    ]
    MANUAL_KEYWORDS = [
        "怎么", "如何", "步骤", "使用", "安装", "清洁", "更换", "调节", "设置",
        "指示灯", "闪烁", "按钮", "接口", "滤网", "遥控器", "自清洁", "专用盐",
        "亮碟剂", "电池", "充电", "故障", "部件", "组件", "manual", "how", "what",
        "where", "battery", "boat", "airfryer", "jetski", "camera", "earphones",
    ]
    IMAGE_KEYWORDS = [
        "图", "图片", "位置", "指示灯", "闪烁", "表带", "尺寸", "部件", "按钮",
        "接口", "安装", "拆卸", "更换", "label", "screen", "button", "indicator",
    ]

    PRODUCT_ALIASES = {
        "DCB107": ["DCB107", "电钻", "充电器", "指示灯"],
        "DCB112": ["DCB112", "电钻", "充电器", "指示灯"],
        "DCB101": ["DCB101", "电钻", "充电器", "指示灯"],
        "表带": ["健身追踪器", "表带", "尺寸"],
        "专用盐": ["洗碗机", "专用盐"],
        "亮碟剂": ["洗碗机", "亮碟剂"],
        "遥控器": ["空调", "遥控器"],
        "滤网": ["空调", "滤网"],
        "自清洁": ["空调", "自清洁"],
        "airfryer": ["airfryer", "air fryer", "User manual", "Before first use"],
        "air fryer": ["airfryer", "air fryer", "User manual", "Before first use"],
        "boat": ["boat", "210FSH", "OWNER", "battery conversion"],
        "battery conversion": ["boat", "battery conversion"],
        "jetski": ["jetski", "WaveRunner"],
        "camera": ["camera", "User Guide"],
        "earphones": ["earphones", "earbuds", "pairing"],
    }

    def route(self, question: str, intent: Dict | None = None) -> RouteResult:
        sub_questions = self.decompose(question)
        policy_hits = self._count_hits(question, self.POLICY_KEYWORDS)
        manual_hits = self._count_hits(question, self.MANUAL_KEYWORDS)
        route = "manual"
        if policy_hits and not self._strong_manual_signal(question):
            route = "policy"
        elif policy_hits and self._strong_policy_signal(question):
            route = "policy"
        elif policy_hits and manual_hits and len(sub_questions) > 1:
            route = "mixed"

        if route not in {"mixed", "policy"} and len(sub_questions) > 1:
            sub_routes = [
                "policy" if self._count_hits(part, self.POLICY_KEYWORDS) else "manual"
                for part in sub_questions
            ]
            if len(set(sub_routes)) > 1:
                route = "mixed"

        keywords = self.expand_keywords(question, intent)
        needs_images = any(kw.lower() in question.lower() for kw in self.IMAGE_KEYWORDS)
        if intent:
            needs_images = needs_images or bool(intent.get("needs_images"))
        return RouteResult(route=route, sub_questions=sub_questions, keywords=keywords, needs_images=needs_images)

    def expand_keywords(self, question: str, intent: Dict | None = None) -> Dict[str, List[str]]:
        keywords = {"high": [], "medium": [], "low": []}
        if intent and isinstance(intent.get("keywords"), dict):
            for level in keywords:
                keywords[level].extend(intent["keywords"].get(level, []))

        lower = question.lower()
        for alias, values in self.PRODUCT_ALIASES.items():
            if alias.lower() in lower:
                keywords["high"].extend(values[:2])
                keywords["medium"].extend(values[2:])

        policy_terms = [kw for kw in self.POLICY_KEYWORDS if kw.lower() in lower]
        keywords["high"].extend(policy_terms[:4])
        keywords["medium"].extend(policy_terms[4:])

        return {level: self._dedupe(values) for level, values in keywords.items()}

    def decompose(self, question: str) -> List[str]:
        cleaned = self._normalize_question(question)
        parts = [p.strip(" ，,。；;") for p in re.split(r"[？?]\s*", cleaned) if p.strip(" ，,。；;")]
        if len(parts) > 1:
            return parts

        splitters = ["同时", "另外", "并且", "而且", "以及", "还需要", "还想", "另外还"]
        pattern = "|".join(map(re.escape, splitters))
        parts = [p.strip(" ，,。；;") for p in re.split(pattern, cleaned) if len(p.strip()) > 6]
        if len(parts) > 1:
            return parts
        return [cleaned]

    def filters_for_route(self, route: str) -> Dict | None:
        if route == "policy":
            return {"source": POLICY_SOURCE}
        return None

    def _normalize_question(self, question: str) -> str:
        text = question.replace('"""', '"').replace("“", '"').replace("”", '"')
        text = text.replace("，\n", "，").replace(",\n", ",")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r'"+\s*,?\s*"+', "，", text)
        return text.strip().strip('"').strip("，,")

    def _count_hits(self, question: str, keywords: List[str]) -> int:
        lower = question.lower()
        return sum(1 for kw in keywords if kw.lower() in lower)

    def _strong_manual_signal(self, question: str) -> bool:
        lower = question.lower()
        return bool(
            re.search(r"\b(how|what|where|when|which)\b", lower)
            or any(model in question.upper() for model in ["DCB101", "DCB107", "DCB112"])
            or any(kw in question for kw in ["指示灯", "遥控器", "滤网", "专用盐", "亮碟剂", "表带"])
        )

    def _strong_policy_signal(self, question: str) -> bool:
        return any(kw in question for kw in ["投诉", "退款", "退货", "换货", "发票", "物流", "快递", "运费", "赔偿"])

    def _dedupe(self, values: List[str]) -> List[str]:
        seen = set()
        output = []
        for value in values:
            if not value:
                continue
            key = value.lower()
            if key not in seen:
                seen.add(key)
                output.append(value)
        return output
