"""
诊断工具 - 检查向量库中是否包含电钻相关内容
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings


async def check_drill_content():
    settings = get_settings()
    print("=" * 70)
    print("检查向量库中的电钻相关内容")
    print("=" * 70)

    # 检查向量库
    print("\n[1] 向量库内容检查:")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

        try:
            collection = client.get_collection("texts")
        except:
            print("  没有找到 documents collection")
            return

        count = collection.count()
        print(f"  documents collection 共有 {count} 条数据")

        # 统计来源
        print("\n[2] 文档来源分析:")
        all_data = collection.get(include=["metadatas"])
        source_counts = {}
        for metadata in all_data.get("metadatas", []):
            if metadata:
                source = metadata.get("source_file", "未知")
                source_counts[source] = source_counts.get(source, 0) + 1

        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  共 {len(sorted_sources)} 个来源文件:")

        for source, count in sorted_sources[:20]:
            marker = "***" if any(kw in source.lower() for kw in ['电钻', '钻', 'dcb', 'drill']) else "   "
            print(f"  [{marker}] {source}: {count} 条")

        # 搜索测试
        print("\n[3] 关键词搜索测试:")
        from core.rag.embedding import EmbeddingModel

        model = EmbeddingModel(
            model_name=settings.EMBEDDING_MODEL,
            dimension=settings.EMBEDDING_DIMENSION,
            provider="openai",
            rate_limit=0.2
        )

        search_queries = ["电钻", "DCB107", "电钻指示灯"]

        for query in search_queries:
            try:
                embedding = await model.embed(query)
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=2,
                    include=["documents", "metadatas", "distances"]
                )

                if results["documents"] and results["documents"][0]:
                    top_score = 1 / (1 + results["distances"][0][0])
                    top_source = results["metadatas"][0][0].get("source_file", "未知") if results["metadatas"][0] else "未知"
                    print(f"\n  查询「{query}」:")
                    print(f"    来源: {top_source}")
                    print(f"    相似度: {top_score:.4f}")
                    print(f"    内容: {results['documents'][0][0][:100]}...")
                else:
                    print(f"\n  查询「{query}」: 无结果")
            except Exception as e:
                print(f"\n  查询「{query}」失败: {e}")

    except Exception as e:
        print(f"错误: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(check_drill_content())
