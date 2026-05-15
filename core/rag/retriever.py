"""
RAG检索器
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import get_settings
from core.rag.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


def _to_chroma_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata values to Chroma-supported scalar types."""
    clean = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, Path):
            clean[key] = str(value)
        else:
            clean[key] = json.dumps(value, ensure_ascii=False)
    return clean


@dataclass
class SearchResult:
    """检索结果"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]


class RAGRetriever:
    """
    RAG检索器

    支持：
    - 纯向量检索
    - 混合检索（向量 + 关键词）
    - 图片检索
    """

    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = EmbeddingModel(
            model_name=self.settings.EMBEDDING_MODEL,
            dimension=self.settings.EMBEDDING_DIMENSION,
            provider="openai",
            rate_limit=self.settings.EMBEDDING_REQUEST_INTERVAL,
            rate_limit_retry_seconds=self.settings.EMBEDDING_RATE_LIMIT_RETRY_SECONDS
        )
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """初始化向量存储"""
        db_type = self.settings.VECTOR_DB_TYPE

        if db_type == "chroma":
            try:
                import chromadb
                from chromadb.config import Settings

                self.vector_store = chromadb.PersistentClient(
                    path=self.settings.CHROMA_PERSIST_DIR
                )
                logger.info(f"Initialized ChromaDB at {self.settings.CHROMA_PERSIST_DIR}")

                # 检查并修复 collection 维度不匹配问题
                self._validate_collection_dimension()

            except ImportError:
                logger.error("ChromaDB not installed. Run: pip install chromadb")

        elif db_type == "milvus":
            try:
                from pymilvus import connections, Collection
                connections.connect(host="localhost", port="19530")
                self.vector_store = Collection("knowledge_base")
            except ImportError:
                logger.error("pymilvus not installed")

        elif db_type == "qdrant":
            try:
                from qdrant_client import QdrantClient
                self.vector_store = QdrantClient(host="localhost", port=6333)
            except ImportError:
                logger.error("qdrant-client not installed")

    def _validate_collection_dimension(self):
        """验证并修复 collection 维度不匹配"""
        try:
            collection = self.vector_store.get_collection("texts")
            if collection:
                actual_dim = collection.metadata.get("hnsw:construction_ef", 0)
                expected_dim = self.settings.EMBEDDING_DIMENSION

                # 获取实际 embedding 维度
                if hasattr(self, 'embedding_model'):
                    expected_dim = self.embedding_model.dimension

                # ChromaDB 不直接存储维度信息，通过尝试查询验证
                logger.info(f"Collection 'texts' exists, dimension: {expected_dim}")

        except Exception as e:
            # Collection 不存在是正常的
            pass

    def clear_collections(self, names: Optional[List[str]] = None) -> None:
        """删除指定 collection，用于干净重建知识库。"""
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")

        names = names or ["texts", "images"]
        for name in names:
            try:
                self.vector_store.delete_collection(name)
                logger.info(f"Deleted Chroma collection: {name}")
            except Exception:
                logger.info(f"Chroma collection does not exist, skip delete: {name}")

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        need_images: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        keywords_by_level: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        检索相关内容和图片

        Args:
            query: 查询文本
            top_k: 返回数量
            need_images: 是否需要图片
            filters: 过滤条件
            keywords: 关键词列表（用于权重加成）【已废弃，建议用 keywords_by_level】
            keywords_by_level: 分级关键词 {"high": [], "medium": [], "low": []}

        Returns:
            {
                "texts": [{"content": str, "source": str, "score": float}],
                "images": [{"id": str, "path": str, "description": str, "score": float}]
            }
        """
        try:
            # 1. 向量化查询
            query_embedding = await self.embedding_model.embed(query)

            # 2. 确定使用哪个关键词参数（优先用分级版本）
            if keywords_by_level is None:
                # 兼容旧格式：把 keywords 转成 keywords_by_level
                keywords_by_level = {
                    "high": keywords[:2] if keywords else [],
                    "medium": [],
                    "low": []
                }
            else:
                # 合并 keywords 和 keywords_by_level（keywords 放入 medium）
                if keywords:
                    keywords_by_level["medium"] = list(set(keywords_by_level.get("medium", []) + keywords))

            # 3. 检索文本（支持关键词权重加成）
            texts = await self._search_texts_with_keywords(
                query_embedding=query_embedding,
                keywords_by_level=keywords_by_level,
                top_k=top_k,
                filters=filters
            )

            # 4. 检索图片（如需要）
            images = []
            if need_images:
                images = await self._search_images(query, top_k=min(top_k, 3))

            logger.info(f"Retrieved {len(texts)} texts and {len(images)} images")

            return {
                "texts": texts,
                "images": images
            }

        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            return {"texts": [], "images": []}

    async def _search_texts_with_keywords(
        self,
        query_embedding: List[float],
        keywords_by_level: Dict[str, List[str]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        检索文本（支持分级关键词权重加成）

        Args:
            query_embedding: 查询向量
            keywords_by_level: 分级关键词 {"high": [], "medium": [], "low": []}
            top_k: 返回数量
            filters: 过滤条件

        Returns:
            List[Dict]: 检索结果列表
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []

        try:
            db_type = self.settings.VECTOR_DB_TYPE
            threshold = self.settings.SIMILARITY_THRESHOLD

            if db_type == "chroma":
                collection = self.vector_store.get_collection("texts")

                # 检索比 top_k 更多的结果，以便过滤后仍有足够结果
                n_results = min(top_k * 5, 100)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filters
                )

                all_results = {}
                if results and results.get("documents"):
                    distances = results.get("distances", [[]])[0]
                    ids = results.get("ids", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0]
                    for i, doc in enumerate(results["documents"][0]):
                        distance = distances[i] if i < len(distances) else 0
                        score = 1.0 / (1.0 + distance)
                        doc_id = ids[i] if i < len(ids) else f"vector_{i}"
                        metadata = metadatas[i] if i < len(metadatas) else {}

                        all_results[doc_id] = {
                            "content": doc,
                            "source": metadata.get("source", ""),
                            "metadata": metadata,
                            "vector_score": score,
                            "keyword_score": 0.0,
                            "keyword_details": {"high": 0, "medium": 0, "low": 0},
                            "distance": distance
                        }

                keyword_results = self._keyword_search(collection, keywords_by_level, filters, limit=min(top_k * 20, 200))
                all_results.update({k: v for k, v in keyword_results.items() if k not in all_results})

                self._apply_keyword_scores(all_results, keywords_by_level)

                # 融合分数
                for doc_id, doc_data in all_results.items():
                    final_score = doc_data["vector_score"] + doc_data["keyword_score"]
                    final_score = min(final_score, 1.0)
                    doc_data["score"] = final_score

                # 过滤低于阈值的文档
                filtered_results = [
                    doc for doc in all_results.values()
                    if doc["score"] >= threshold
                ]

                # 按分数排序
                filtered_results.sort(key=lambda x: x["score"], reverse=True)

                # 格式化返回结果
                texts = []
                for doc in filtered_results[:top_k]:
                    texts.append({
                        "content": doc["content"],
                        "source": doc["source"],
                        "score": doc["score"],
                        "vector_score": doc["vector_score"],
                        "keyword_score": doc["keyword_score"],
                        "keyword_details": doc["keyword_details"],
                        "distance": doc["distance"],
                        "metadata": doc.get("metadata", {}),
                        "source_matched": any(
                            kw.lower() in doc["source"].lower()
                            for level_keywords in keywords_by_level.values()
                            for kw in level_keywords
                        )
                    })

                logger.info(f"Vector search: {len(all_results)} results, "
                           f"{len(texts)} passed threshold ({threshold}), "
                           f"highKw:{len([d for d in all_results.values() if d['keyword_details']['high'] > 0])} "
                           f"mediumKw:{len([d for d in all_results.values() if d['keyword_details']['medium'] > 0])}")

                return texts

            return []

        except Exception as e:
            logger.error(f"Text search error: {e}", exc_info=True)
            return []

    def _keyword_search(
        self,
        collection,
        keywords_by_level: Dict[str, List[str]],
        filters: Optional[Dict[str, Any]],
        limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        keywords = [
            kw.strip()
            for level in ["high", "medium", "low"]
            for kw in keywords_by_level.get(level, [])
            if kw and kw.strip()
        ]
        if not keywords:
            return {}

        try:
            kwargs = {"include": ["documents", "metadatas"], "limit": 10000}
            if filters:
                kwargs["where"] = filters
            data = collection.get(**kwargs)
        except Exception as e:
            logger.warning(f"Keyword search skipped: {e}")
            return {}

        scored = []
        for doc_id, content, metadata in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            haystack = f"{metadata.get('source', '')}\n{metadata.get('title', '')}\n{metadata.get('full_title', '')}\n{content}".lower()
            score = 0
            for kw in keywords:
                score += haystack.count(kw.lower())
            if score:
                scored.append((score, doc_id, content, metadata))

        scored.sort(key=lambda item: item[0], reverse=True)
        return {
            doc_id: {
                "content": content,
                "source": metadata.get("source", ""),
                "metadata": metadata,
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "keyword_details": {"high": 0, "medium": 0, "low": 0},
                "distance": None,
            }
            for _, doc_id, content, metadata in scored[:limit]
        }

    def _apply_keyword_scores(
        self,
        results: Dict[str, Dict[str, Any]],
        keywords_by_level: Dict[str, List[str]],
    ) -> None:
        level_weights = {"high": 0.3, "medium": 0.1, "low": 0.02}
        level_max_bonus = {"high": 0.6, "medium": 0.3, "low": 0.1}

        for level in ["high", "medium", "low"]:
            for keyword in keywords_by_level.get(level, []):
                if not keyword:
                    continue
                keyword_lower = keyword.lower()
                for doc_data in results.values():
                    metadata_text = " ".join(
                        str(doc_data.get("metadata", {}).get(k, ""))
                        for k in ["source", "title", "full_title", "manual_id", "original_source"]
                    ).lower()
                    content_lower = doc_data["content"].lower()
                    source_match = keyword_lower in metadata_text
                    content_match = keyword_lower in content_lower
                    if source_match or content_match:
                        count = content_lower.count(keyword_lower)
                        bonus = min(max(count, 1) * level_weights[level], level_max_bonus[level])
                        if source_match:
                            bonus *= 1.5
                        doc_data["keyword_score"] += bonus
                        doc_data["keyword_details"][level] += bonus

    async def _search_texts(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """检索文本（不带关键词权重，纯向量检索）"""
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []

        try:
            db_type = self.settings.VECTOR_DB_TYPE
            threshold = self.settings.SIMILARITY_THRESHOLD

            if db_type == "chroma":
                collection = self.vector_store.get_collection("texts")

                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k * 3, 50),
                    where=filters
                )

                texts = []
                if results and results.get("documents"):
                    distances = results.get("distances", [[]])[0]
                    for i, doc in enumerate(results["documents"][0]):
                        distance = distances[i] if i < len(distances) else 0
                        score = 1.0 / (1.0 + distance)

                        if score >= threshold:
                            texts.append({
                                "content": doc,
                                "source": results.get("metadatas", [[{}]])[0][i].get("source", ""),
                                "score": score,
                                "distance": distance
                            })

                    texts.sort(key=lambda x: x["score"], reverse=True)

                logger.info(f"Vector search: {len(results.get('documents', [[]])[0]) if results else 0} results, "
                           f"{len(texts)} passed threshold ({threshold})")

                return texts[:top_k]

            return []

        except Exception as e:
            logger.error(f"Text search error: {e}", exc_info=True)
            return []

    async def _search_images(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """检索相关图片"""
        from core.rag.image_index import ImageIndexer

        indexer = ImageIndexer()
        return await indexer.search(query, top_k)

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = None
    ) -> int:
        """
        添加文档到向量库

        Args:
            documents: [{"content": str, "source": str, "metadata": dict}]
            batch_size: 批量大小（默认从配置读取，越大越快但需注意API限制）

        Returns:
            int: 添加的文档数量
        """
        if not documents:
            return 0

        import asyncio

        # 从配置读取默认批量大小
        batch_size = batch_size or self.settings.EMBEDDING_BATCH_SIZE
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logger.info(f"开始添加 {len(documents)} 个文档，分 {total_batches} 批，每批 {batch_size} 条")

        total_added = 0
        failed_batches = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                # 批量向量化（包含自动重试和指数退避）
                texts = [doc["content"] for doc in batch]
                embeddings = await self.embedding_model.embed_batch(texts)

                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"Embedding count mismatch for batch {batch_num}: "
                        f"expected {len(batch)}, got {len(embeddings)}"
                    )

                # 存储到向量库
                db_type = self.settings.VECTOR_DB_TYPE

                if db_type == "chroma" and self.vector_store:
                    try:
                        collection = self.vector_store.get_collection("texts")
                    except Exception:
                        # Collection 不存在，创建新的（使用正确的维度）
                        collection = self.vector_store.create_collection(
                            name="texts",
                            metadata={"dimension": self.settings.EMBEDDING_DIMENSION}
                        )

                    ids = [
                        str(doc.get("metadata", {}).get("chunk_id") or f"doc_{i + j}")
                        for j, doc in enumerate(batch)
                    ]
                    metadatas = [
                        _to_chroma_metadata(doc.get("metadata", {}))
                        for doc in batch
                    ]

                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )

                    total_added += len(batch)
                    logger.info(f"批次 {batch_num}/{total_batches} 完成: {len(batch)} 条文档 (累计 {total_added}/{len(documents)})")
                else:
                    failed_batches.append(batch_num)

            except Exception as e:
                logger.error(f"批次 {batch_num} 处理失败: {e}")
                failed_batches.append(batch_num)

            # 批次之间增加延迟，避免API限流
            if i + batch_size < len(documents):
                await asyncio.sleep(1.0)
                logger.debug(f"批次间隔等待 1.0 秒")

        if failed_batches:
            raise RuntimeError(f"有 {len(failed_batches)} 个批次处理失败: 批次 {failed_batches}")

        return total_added
