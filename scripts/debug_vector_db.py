"""
调试工具 - 检查向量库状态
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from core.rag.embedding import EmbeddingModel


async def check_vector_db():
    """检查向量库状态"""
    settings = get_settings()

    print("=" * 60)
    print("向量库状态检查")
    print("=" * 60)

    # 1. 配置信息
    print("\n📋 配置信息:")
    print(f"  - 向量库路径: {settings.CHROMA_PERSIST_DIR}")
    print(f"  - 向量维度配置: {settings.EMBEDDING_DIMENSION}")
    print(f"  - Embedding模型: {settings.EMBEDDING_MODEL}")
    print(f"  - 相似度阈值: {settings.SIMILARITY_THRESHOLD}")
    print(f"  - TOP_K: {settings.TOP_K}")

    # 2. 检查向量库文件
    print("\n📁 向量库目录:")
    vector_db_path = Path(settings.CHROMA_PERSIST_DIR)
    if vector_db_path.exists():
        print(f"  ✅ 目录存在: {vector_db_path}")
        # 列出子目录
        for item in vector_db_path.iterdir():
            print(f"     - {item.name}/")
    else:
        print(f"  ❌ 目录不存在: {vector_db_path}")
        return

    # 3. 检查 Collection
    print("\n📦 Collection 信息:")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(vector_db_path))
        collections = client.list_collections()

        if not collections:
            print("  ⚠️  没有找到任何 Collection（向量库为空）")
        else:
            for col in collections:
                print(f"  ✅ Collection: {col.name}")
                print(f"     - 数据条数: {col.count()}")
                print(f"     - 元数据: {col.metadata}")
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")

    # 4. 检查 Embedding 维度
    print("\n🔢 Embedding 维度检测:")
    try:
        # 调用 API 获取实际维度
        test_text = "这是一个测试文本"
        model = EmbeddingModel(
            model_name=settings.EMBEDDING_MODEL,
            dimension=settings.EMBEDDING_DIMENSION,
            provider="openai"
        )
        embedding = await model.embed(test_text)
        actual_dim = len(embedding)
        print(f"  - 配置维度: {settings.EMBEDDING_DIMENSION}")
        print(f"  - 实际维度: {actual_dim}")

        if actual_dim != settings.EMBEDDING_DIMENSION:
            print(f"  ⚠️  维度不匹配！请更新 .env 中的 EMBEDDING_DIMENSION={actual_dim}")
        else:
            print("  ✅ 维度匹配")
    except Exception as e:
        print(f"  ❌ 检测失败: {e}")

    # 5. 测试检索
    print("\n🔍 测试检索:")
    try:
        from core.rag.retriever import RAGRetriever
        retriever = RAGRetriever()

        test_query = "产品功能介绍"
        print(f"  - 测试查询: {test_query}")

        results = await retriever.retrieve(test_query, top_k=3)

        if results["texts"]:
            print(f"  ✅ 检索成功，返回 {len(results['texts'])} 条结果")
            for i, r in enumerate(results["texts"], 1):
                print(f"     [{i}] 相似度: {r['score']:.4f}")
                print(f"         来源: {r['source']}")
                print(f"         内容: {r['content'][:80]}...")
        else:
            print("  ⚠️  没有检索到任何结果")
            print("     可能原因:")
            print("     1. 向量库为空（需要重新构建）")
            print("     2. 相似度阈值过高（当前: {}）".format(settings.SIMILARITY_THRESHOLD))
            print("     3. Embedding维度不匹配")

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(check_vector_db())
