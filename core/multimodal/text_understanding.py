"""
文本理解模块
"""
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TextUnderstanding:
    """
    文本理解器

    负责：
    - 文本预处理
    - 关键信息提取
    - 问题结构分析
    """

    def __init__(self):
        self.stopwords = set(["的", "了", "在", "是", "我", "有", "和", "就", "不", "人"])

    def preprocess(self, text: str) -> str:
        """
        文本预处理

        - 去除多余空白
        - 规范化标点
        - 去除特殊字符
        """
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 规范化引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()

    def extract_product_info(self, text: str) -> Dict[str, any]:
        """
        提取产品相关信息

        Returns:
            {
                "model_numbers": List[str],  # 型号
                "product_category": str,  # 产品类别
                "parts": List[str]  # 零部件
            }
        """
        # 提取型号（通常是字母+数字组合）
        model_pattern = r'[A-Z]{2,}[0-9]{2,}[A-Z0-9]*'
        models = re.findall(model_pattern, text.upper())

        # 产品类别关键词
        categories = {
            "电钻": ["电钻", "钻"],
            "电动工具": ["电动工具", "工具"],
            "健身追踪器": ["健身追踪器", "追踪器", "手表"],
            "冰箱": ["冰箱", "冷藏"],
            "吹风机": ["吹风机", "吹风"],
        }

        detected_category = ""
        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                detected_category = category
                break

        return {
            "model_numbers": list(set(models)),
            "product_category": detected_category,
            "parts": []
        }

    def split_sub_questions(self, text: str) -> List[str]:
        """
        拆分复合问句

        识别多个问号或特定连接词，将复杂问题拆分为多个子问题
        """
        # 按问号拆分
        if text.count('？') > 1 or text.count('?') > 1:
            sentences = re.split(r'[？?]', text)
            return [s.strip() for s in sentences if s.strip()]

        # 按连接词拆分
        connectors = ["并且", "而且", "还有", "另外", "以及", "还有"]
        for conn in connectors:
            if conn in text:
                parts = text.split(conn)
                return [p.strip() for p in parts if p.strip()]

        return [text]

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        提取关键词

        基于TF-IDF思想的简单关键词提取
        """
        # 分词（简单按字符）
        words = []
        current_word = ""

        for char in text:
            if char.isalnum():
                current_word += char
            else:
                if current_word and len(current_word) >= 2:
                    words.append(current_word)
                current_word = ""

        if current_word and len(current_word) >= 2:
            words.append(current_word)

        # 过滤停用词
        words = [w for w in words if w not in self.stopwords]

        # 统计词频
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [w for w, _ in sorted_words[:top_n]]
