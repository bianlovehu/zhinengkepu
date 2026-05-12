"""
RAG检索测试脚本
用法: python scripts/test_retrieval.py "关键词1" "关键词2" ...

示例:
  python scripts/test_retrieval.py "电钻"
  python scripts/test_retrieval.py "电钻" "电池" "充电器"
  python scripts/test_retrieval.py
"""
import asyncio
import sys
import os
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings


async def test_retrieval(keywords: List[str] = None):
    """测试RAG检索"""
    settings = get_settings()

    print("=" * 70)
    print("RAG检索测试")
    print("=" * 70)

    print(f"\n[配置信息]")
    print(f"  - 向量库: {settings.VECTOR_DB_TYPE}")
    print(f"  - 持久化路径: {settings.CHROMA_PERSIST_DIR}")
    print(f"  - Embedding模型: {settings.EMBEDDING_MODEL}")
    print(f"  - 相似度阈值: {settings.SIMILARITY_THRESHOLD}")
    print(f"  - 默认TopK: {settings.TOP_K}")

    # 初始化检索器
    print(f"\n[初始化] 正在加载向量库...")
    from core.rag.retriever import RAGRetriever
    retriever = RAGRetriever()

    # 检查向量库状态
    try:
        collection = retriever.vector_store.get_collection("texts")
        count = collection.count()
        print(f"  - texts collection: {count} 条数据")
    except Exception as e:
        print(f"  - 警告: 无法获取集合信息 - {e}")

    # 如果没有提供关键词，提示用户输入
    if not keywords:
        print("\n" + "=" * 70)
        print("请输入要测试的关键词（多个用逗号分隔，直接回车退出）:")
        user_input = input("> ").strip()
        if not user_input:
            print("退出测试")
            return
        keywords = [kw.strip() for kw in user_input.split(",") if kw.strip()]

    # 测试每个关键词
    for keyword in keywords:
        print("\n" + "=" * 70)
        print(f"查询: 「{keyword}」")
        print("=" * 70)

        # 模拟意图识别提取分级关键词（简化版）
        keywords_by_level = _extract_keywords_from_query(keyword)

        print(f"  高优先级关键词 (high): {keywords_by_level['high']}")
        print(f"  中优先级关键词 (medium): {keywords_by_level['medium']}")
        print(f"  低优先级关键词 (low): {keywords_by_level['low']}")

        results = await retriever.retrieve(
            query=keyword,
            top_k=5,
            need_images=False,
            keywords_by_level=keywords_by_level
        )

        texts = results.get("texts", [])

        if not texts:
            print("  [无结果] 没有找到相关内容")
            print(f"  可能原因: 相关内容相似度低于阈值 {settings.SIMILARITY_THRESHOLD}")
        else:
            print(f"  找到 {len(texts)} 条结果:\n")
            for i, item in enumerate(texts, 1):
                vector_score = item.get('vector_score', item['score'])
                keyword_score = item.get('keyword_score', 0)
                keyword_details = item.get('keyword_details', {})
                source_matched = item.get('source_matched', False)
                source_indicator = " [来源匹配]" if source_matched else ""

                print(f"  【结果 {i}】 综合分: {item['score']:.4f} (向量:{vector_score:.4f} + 高优先级:{keyword_details.get('high',0):.3f}/中优先级:{keyword_details.get('medium',0):.3f}){source_indicator}")
                print(f"    来源: {item.get('source', '未知') or '未知'}")

                # 截取内容预览（保留前200字符）
                content = item['content']
                preview = content[:200] + "..." if len(content) > 200 else content
                preview = preview.replace('\n', ' ')
                print(f"    内容: {preview}")
                print()

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


def _extract_keywords_from_query(query: str) -> dict:
    """
    从查询中提取关键词（简化版，按优先级分类）

    实际应调用 LLM 意图识别来获取分级关键词

    Args:
        query: 用户查询

    Returns:
        dict: {"high": [], "medium": [], "low": []}
    """
    import re

    # 产品词（核心）
    product_keywords = [
        "电钻", "充电器", "电池", "电池包", "电动工具",
        "发电机", "洗碗机", "VR", "头显", "显示器",
        "电锯", "角磨机", "冲击钻", "螺丝刀", "锯子"
    ]

    # 问题词
    issue_keywords = ["指示灯", "闪烁", "故障", "报警", "错误", "异常", "发热", "噪音", "充电"]

    # 通用词
    common_keywords = ["灯", "问题", "怎么", "如何", "什么", "哪个", "哪里", "为什么"]

    # 型号匹配
    model_pattern = r'(DCB\d+|DC[A-Z0-9]+|BP[A-Z0-9]+|BL[A-Z0-9]+|B[A-Z0-9]+|BF[A-Z0-9]+|BD[A-Z0-9]+|DW[A-Z0-9]+)'
    models = [m.upper() for m in re.findall(model_pattern, query, re.IGNORECASE)]

    # 提取产品词
    products = [kw for kw in product_keywords if kw in query]

    # 提取问题词
    issues = [kw for kw in issue_keywords if kw in query]

    # 提取通用词
    common = [kw for kw in common_keywords if kw in query]

    return {
        "high": models + products,
        "medium": issues,
        "low": common
    }


def main():
    # 从命令行参数获取关键词
    keywords = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(test_retrieval(keywords))


if __name__ == "__main__":
    main()
