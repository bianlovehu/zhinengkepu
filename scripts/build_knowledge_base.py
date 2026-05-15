"""
构建知识库脚本

Usage:
    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --docs_dir ./manual --clear
    python scripts/build_knowledge_base.py --images_dir ./images
"""
import sys
import asyncio
import argparse
import logging
import time
import re
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from core.rag.document_loader import DocumentLoader
from core.rag.chunker import TextChunker
from core.rag.image_index import ImageIndexer
from core.rag.retriever import RAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def log_section(title: str):
    """打印分隔标题"""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def log_subsection(title: str):
    """打印子标题"""
    logger.info("")
    logger.info(f"▶ {title}")


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


def clear_existing_indexes(settings):
    """清空文本、图片 collection 和图片元数据索引。"""
    log_section("清除现有向量库")
    retriever = RAGRetriever()
    retriever.clear_collections(["texts", "images"])

    image_index_file = Path(settings.DATA_DIR) / "processed" / "image_index.json"
    if image_index_file.exists():
        image_index_file.unlink()
        logger.info(f"已删除图片索引元数据: {image_index_file}")


def log_document_quality(documents):
    logger.info("📈 文档解析质量:")
    for doc in documents:
        newline_count = doc.content.count("\n")
        heading_count = len(re.findall(r"(?m)^\s*#{1,6}\s+", doc.content))
        logger.info(
            f"   - {doc.source}: {len(doc.content):,} 字符, "
            f"{newline_count} 换行, {heading_count} 标题, "
            f"language={doc.metadata.get('language', 'unknown')}"
        )


def log_chunk_quality(chunks):
    lengths = [len(chunk.content) for chunk in chunks]
    if not lengths:
        return
    logger.info("📈 分块质量:")
    logger.info(f"   - 最小/最大/平均长度: {min(lengths)} / {max(lengths)} / {sum(lengths)/len(lengths):.1f}")
    logger.info(f"   - 超过 900 字符: {sum(1 for length in lengths if length > 900)}")


def find_image_directory(docs_dir: str, images_dir: str = None) -> Path:
    if images_dir:
        return Path(images_dir)
    return Path(docs_dir) / "插图"


def build_image_contexts(documents, image_dir: Path):
    """基于图片 ID、占位符附近文本和文件名生成图片索引描述。"""
    if not image_dir.exists() or not image_dir.is_dir():
        return []

    image_files = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    lower_lookup = {stem.lower(): stem for stem in image_files}
    pending_by_prefix = {}
    for stem in sorted(image_files):
        prefix = re.sub(r"[_-]?\d+$", "", stem).lower()
        pending_by_prefix.setdefault(prefix, []).append(stem)

    images = {}
    placeholder_seq = 0

    for doc in documents:
        text = doc.content
        text_lower = text.lower()
        manual_id = doc.metadata.get("manual_id", doc.source)
        for match in re.finditer(r"<PIC>\s*([A-Za-z0-9_\-]+)?|\[PIC:([A-Za-z0-9_\-]+)\]|(Manual\d+_\d+)", text):
            explicit_id = next((g for g in match.groups() if g), None)
            context = _context_window(text, match.start(), match.end())
            image_id = None

            if explicit_id:
                image_id = lower_lookup.get(explicit_id.lower())
            if not image_id:
                image_id = _next_image_for_document(doc, pending_by_prefix)
            if not image_id or image_id in images:
                placeholder_seq += 1
                continue

            path = image_files[image_id]
            images[image_id] = {
                "image_path": str(path),
                "image_id": image_id,
                "page_num": None,
                "manual_id": manual_id,
                "source": doc.source,
                "description": (
                    f"产品手册插图: {path.name}\n"
                    f"来源: {doc.source}\n"
                    f"附近正文: {context}"
                ),
            }

        for stem, path in image_files.items():
            stem_lower = stem.lower()
            if stem in images or stem_lower not in text_lower:
                continue
            pos = text_lower.find(stem_lower)
            images[stem] = {
                "image_path": str(path),
                "image_id": stem,
                "page_num": None,
                "manual_id": manual_id,
                "source": doc.source,
                "description": (
                    f"产品手册插图: {path.name}\n"
                    f"来源: {doc.source}\n"
                    f"附近正文: {_context_window(text, pos, pos + len(stem))}"
                ),
            }

    return list(images.values())


def _context_window(text: str, start: int, end: int, radius: int = 180) -> str:
    before = text[max(0, start - radius):start]
    after = text[end:min(len(text), end + radius)]
    context = f"{before} {after}"
    context = re.sub(r"\s+", " ", context)
    return context.strip()[:360]


def _next_image_for_document(doc, pending_by_prefix):
    candidates = [
        doc.metadata.get("title", ""),
        doc.metadata.get("full_title", ""),
        doc.metadata.get("original_source", ""),
        doc.source,
    ]
    normalized = " ".join(candidates).lower()
    alias_map = {
        "空调": "air_conditioner",
        "烤箱": "oven",
        "电钻": "drill0",
        "水泵": "pump",
        "摩托艇": "jetski",
        "洗碗机": "dish_washer",
        "发电机": "generator",
        "功能键盘": "function_keyboard",
        "冰箱": "fridge",
        "温控器": "thermostat",
        "vr": "vr",
        "相机": "camera",
        "健身追踪器": "fitness_trackers",
        "健身单车": "exercise_bikes",
        "儿童电动摩托车": "rideon_motorcycle",
        "蓝牙激光鼠标": "mouse",
        "吹风机": "blower",
    }
    for keyword, prefix in alias_map.items():
        if keyword.lower() in normalized:
            queue = pending_by_prefix.get(prefix.lower())
            if queue:
                return queue.pop(0)
    return None


class ImageIndexProgressTracker:
    """图像索引进度跟踪器"""

    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.last_log_time = 0

    def __call__(self, current: int, total: int, status: str):
        """进度回调"""
        self.current = current
        elapsed = time.time() - self.start_time

        # 控制日志频率：每5秒或最后一条才打印
        should_log = (
            current == total or  # 最后一条
            current == 1 or  # 第一条
            time.time() - self.last_log_time > 5  # 超过5秒
        )

        if should_log:
            self.last_log_time = time.time()
            progress = current / total * 100
            avg_time = elapsed / current if current > 0 else 0
            remaining = avg_time * (total - current)
            logger.info(
                f"  图片处理: {current}/{total} ({progress:.0f}%) "
                f"| 已用时: {format_time(elapsed)} "
                f"| 预计剩余: {format_time(remaining)} "
                f"| 当前: {status[:30]}..."
            )


async def build_knowledge_base(
    docs_dir: str = None,
    images_dir: str = None,
    clear_existing: bool = False,
    skip_images: bool = False,
    image_limit: int = None
):
    """
    构建知识库

    Args:
        docs_dir: 文档目录路径
        clear_existing: 是否清除现有数据
    """
    settings = get_settings()
    docs_dir = docs_dir or str(settings.RAW_DOCS_DIR)
    total_start_time = time.time()

    log_section("知识库构建任务启动")
    logger.info(f"📁 文档目录: {docs_dir}")
    logger.info(f"📊 分块大小: {settings.CHUNK_SIZE} 字符")
    logger.info(f"📊 重叠大小: {settings.CHUNK_OVERLAP} 字符")
    logger.info(f"💾 向量库路径: {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"📦 Embedding批量大小: {settings.EMBEDDING_BATCH_SIZE}")
    logger.info(f"⏱️ Embedding请求间隔: {settings.EMBEDDING_REQUEST_INTERVAL} 秒")

    if clear_existing:
        clear_existing_indexes(settings)

    # ========== 步骤 1: 加载文档 ==========
    log_section("步骤 1/5: 加载文档")
    load_start = time.time()
    loader = DocumentLoader()

    try:
        documents = loader.load_directory(docs_dir)
        load_time = time.time() - load_start

        logger.info(f"✅ 成功加载 {len(documents)} 个文档 ({format_time(load_time)})")

        # 统计各类型文档数量
        ext_counts = {}
        total_chars = 0
        for doc in documents:
            ext = Path(doc.source).suffix.upper() or "TXT"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_chars += len(doc.content)

        logger.info(f"📈 文档统计:")
        for ext, count in sorted(ext_counts.items()):
            logger.info(f"   - {ext}: {count} 个")
        logger.info(f"   - 总字符数: {total_chars:,} 字符")
        log_document_quality(documents)

    except Exception as e:
        logger.error(f"❌ 加载文档失败: {e}")
        return

    if not documents:
        logger.warning("⚠️  未找到任何文档，请检查目录路径")
        return

    # ========== 步骤 2: 文本分块 ==========
    log_section("步骤 2/5: 文本分块")
    log_subsection("分块策略: 语义分块（优先） → 固定大小分块（降级）")

    chunker = TextChunker(
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP
    )

    chunk_start = time.time()
    chunks = chunker.chunk_documents([
        {"content": doc.content, "source": doc.source, "metadata": doc.metadata}
        for doc in documents
    ])
    chunk_time = time.time() - chunk_start

    logger.info(f"✅ 分块完成 ({format_time(chunk_time)})")
    logger.info(f"📊 总计: {len(chunks)} 个文本块")
    logger.info(f"📈 平均每文档: {len(chunks)/len(documents):.1f} 块")
    log_chunk_quality(chunks)

    oversized = [chunk for chunk in chunks if len(chunk.content) > 900]
    if oversized:
        raise RuntimeError(f"存在 {len(oversized)} 个超长 chunk，请先修复切块策略")

    # ========== 步骤 3: 添加到向量库 ==========
    log_section("步骤 3/5: 添加文本到向量库")
    logger.info(f"🔢 准备添加 {len(chunks)} 个文本块...")

    retriever = RAGRetriever()

    docs_for_index = [
        {
            "content": chunk.content,
            "source": chunk.metadata.get("source", ""),
            "metadata": {"chunk_id": chunk.chunk_id, **chunk.metadata}
        }
        for chunk in chunks
    ]

    try:
        vec_start = time.time()
        count = await retriever.add_documents(docs_for_index)
        vec_time = time.time() - vec_start

        logger.info(f"✅ 成功添加 {count} 个文本块 ({format_time(vec_time)})")
        logger.info(f"📊 平均速度: {count/vec_time:.1f} 块/秒")
    except Exception as e:
        logger.error(f"❌ 添加到向量库失败: {e}")
        return

    total_images = 0
    # ========== 步骤 4: 索引插图 ==========
    if skip_images:
        log_section("步骤 4/5: 跳过插图索引")
        logger.info("已按参数 --skip_images 跳过图片索引，只构建文本知识库。")
    else:
        log_section("步骤 4/5: 索引插图")
        image_indexer = ImageIndexer()
        image_dir = find_image_directory(docs_dir, images_dir)

    if not skip_images and image_dir.exists() and image_dir.is_dir():
        log_subsection("索引插图文件（文件名 + 文档附近上下文）")
        logger.info(f"📁 插图目录: {image_dir}")

        images_to_index = build_image_contexts(documents, image_dir)
        if image_limit is not None:
            images_to_index = images_to_index[:max(0, image_limit)]
            logger.info(f"📊 按 --image_limit 限制本次图片索引数量: {len(images_to_index)}")
        total_image_files = len([f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
        logger.info(f"📊 插图目录共有 {total_image_files} 个图片文件")
        logger.info(f"📊 匹配到文档上下文的: {len(images_to_index)} 个")

        if images_to_index:
            img_start = time.time()
            tracker = ImageIndexProgressTracker(len(images_to_index))

            # 使用批量处理
            added = await image_indexer.add_images_batch(
                images_to_index,
                progress_callback=tracker
            )
            img_time = time.time() - img_start

            total_images = added
            logger.info(f"✅ 插图索引完成 ({format_time(img_time)})")
            logger.info(f"📊 处理速度: {total_images/img_time:.1f} 张/秒")

            # 显示索引统计
            vector_count = image_indexer.get_collection_count()
            logger.info(f"📊 向量库中图片数量: {vector_count}")
    elif not skip_images:
        logger.warning(f"⚠️  插图目录不存在: {image_dir}")
        logger.info("💡 提示: 如有插图，请确保放在文档目录下的'插图'子文件夹中")

    # ========== 步骤 5: 完成 ==========
    total_time = time.time() - total_start_time

    log_section("知识库构建完成")
    logger.info(f"📋 构建摘要:")
    logger.info(f"   - 文档数量: {len(documents)} 个")
    logger.info(f"   - 文本块数: {len(chunks)} 个")
    logger.info(f"   - 插图数量: {total_images} 张")
    logger.info(f"   - Embedding模型: {settings.EMBEDDING_MODEL}")
    logger.info(f"   - 向量维度: {settings.EMBEDDING_DIMENSION}")
    logger.info(f"   - 向量库: {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"⏱️  总耗时: {format_time(total_time)}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ✅ 知识库已就绪，可以启动服务进行测试")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="构建知识库")
    parser.add_argument("--docs_dir", "-d", help="文档目录路径")
    parser.add_argument("--clear", "-c", action="store_true", help="清除现有数据")
    parser.add_argument("--images_dir", "-i", help="图片目录路径")
    parser.add_argument("--skip_images", action="store_true", help="只构建文本知识库，跳过图片索引")
    parser.add_argument("--image_limit", type=int, help="限制本次索引的图片数量，用于小批量测试")

    args = parser.parse_args()

    # 使用asyncio运行异步构建
    asyncio.run(build_knowledge_base(
        docs_dir=args.docs_dir,
        images_dir=args.images_dir,
        clear_existing=args.clear,
        skip_images=args.skip_images,
        image_limit=args.image_limit
    ))


if __name__ == "__main__":
    main()
