"""
图片索引器

支持：
- 图片描述生成（基于LLM视觉理解）
- 图像描述向量存储（与文本统一向量库）
- 图像语义检索（基于描述embedding）
"""
import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import get_settings
from core.llm.base import LLMClient
from core.rag.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class ImageIndexer:
    """
    图片索引器

    功能：
    - 使用LLM生成图片描述
    - 对描述进行向量 embedding
    - 存储到 ChromaDB 向量库
    - 支持语义检索相关图片
    """

    COLLECTION_NAME = "images"

    def __init__(self):
        self.settings = get_settings()
        self.llm_client = LLMClient()
        self.embedding_model = EmbeddingModel(
            model_name=self.settings.EMBEDDING_MODEL,
            dimension=self.settings.EMBEDDING_DIMENSION,
            provider="openai",
            rate_limit=self.settings.EMBEDDING_REQUEST_INTERVAL,
            rate_limit_retry_seconds=self.settings.EMBEDDING_RATE_LIMIT_RETRY_SECONDS
        )
        self.vector_store = None
        self._init_vector_store()
        self._load_metadata_index()

    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            import chromadb
            from chromadb.config import Settings

            self.vector_store = chromadb.PersistentClient(
                path=self.settings.CHROMA_PERSIST_DIR
            )

            # 确保 images collection 存在
            try:
                self.vector_store.get_collection(self.COLLECTION_NAME)
            except Exception:
                self.vector_store.create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"dimension": self.settings.EMBEDDING_DIMENSION}
                )
                logger.info(f"Created new collection: {self.COLLECTION_NAME}")

            logger.info(f"Image indexer initialized, vector store: {self.settings.CHROMA_PERSIST_DIR}")

        except ImportError:
            logger.error("ChromaDB not installed")
        except Exception as e:
            logger.error(f"Failed to init vector store: {e}")

    def _load_metadata_index(self):
        """加载元数据索引（用于存储不在向量库中的信息）"""
        self.index_file = Path(self.settings.DATA_DIR) / "processed" / "image_index.json"
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.metadata_index = json.load(f)
            except Exception:
                self.metadata_index = {"images": {}}
        else:
            self.metadata_index = {"images": {}}

    def _save_metadata_index(self):
        """保存元数据索引"""
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata_index, f, ensure_ascii=False, indent=2)

    async def add_image(
        self,
        image_path: str,
        image_id: str,
        page_num: Optional[int] = None,
        manual_description: Optional[str] = None,
        skip_embedding: bool = False
    ) -> Dict[str, Any]:
        """
        添加图片到索引

        Args:
            image_path: 图片路径
            image_id: 图片ID（用于引用）
            page_num: 页码
            manual_description: 手动描述
            skip_embedding: 是否跳过embedding（用于批量添加）

        Returns:
            Dict: 图片信息
        """
        # 生成或使用描述
        if manual_description:
            description = manual_description
        else:
            description = await self._generate_description(image_path)

        # 提取关键词
        keywords = self._extract_keywords(description)

        # 构建图片信息
        image_info = {
            "id": image_id,
            "path": str(Path(image_path).absolute()),
            "page": page_num,
            "description": description,
            "keywords": keywords,
            "source_file": str(Path(image_path).name)
        }

        # 存储元数据
        self.metadata_index["images"][image_id] = {
            "path": image_info["path"],
            "page": page_num,
            "source_file": image_info["source_file"]
        }
        self._save_metadata_index()

        # 存入向量库
        if not skip_embedding and self.vector_store:
            await self._add_to_vector_store(image_id, description, keywords, image_info)

        logger.debug(f"Added image to index: {image_id}")
        return image_info

    async def add_images_batch(
        self,
        images: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> int:
        """
        批量添加图片（优化版）

        Args:
            images: 图片列表 [{"image_path": str, "image_id": str, "page_num": int, "description": str}]
            progress_callback: 进度回调函数

        Returns:
            int: 成功添加的数量
        """
        if not images:
            return 0

        total = len(images)
        logger.info(f"批量处理 {total} 张图片...")
        added_count = 0

        # 第一步：收集所有描述（先生成所有图片描述）
        descriptions = []
        for idx, img in enumerate(images):
            desc = img.get("description") or img.get("manual_description")
            if not desc:
                # 需要生成描述
                desc = await self._generate_description(img["image_path"])
            descriptions.append(desc)

            # 更新进度（描述生成进度）
            if progress_callback:
                progress_callback(idx + 1, total, f"生成描述: {img['image_id']}")

        # 第二步：批量生成 embedding（一次 API 调用处理多条）
        logger.info("批量生成 embedding...")
        if descriptions and self.embedding_model:
            # 分批生成 embedding
            batch_size = max(
                1,
                min(
                    self.settings.IMAGE_EMBEDDING_BATCH_SIZE,
                    self.settings.EMBEDDING_BATCH_SIZE,
                )
            )
            all_embeddings = []
            for i in range(0, len(descriptions), batch_size):
                batch = descriptions[i:i + batch_size]
                batch_no = i // batch_size + 1
                total_batches = (len(descriptions) + batch_size - 1) // batch_size
                logger.info(
                    f"生成图片 embedding 批次 {batch_no}/{total_batches}: "
                    f"{len(batch)} 条"
                )
                try:
                    batch_emb = await self.embedding_model.embed_batch(batch)
                    all_embeddings.extend(batch_emb)
                except Exception as e:
                    raise RuntimeError(f"Image embedding batch failed: {e}") from e
                if i + batch_size < len(descriptions) and self.settings.IMAGE_EMBEDDING_BATCH_DELAY > 0:
                    await asyncio.sleep(self.settings.IMAGE_EMBEDDING_BATCH_DELAY)

        # 第三步：批量存入向量库
        if self.vector_store and all_embeddings:
            collection = self.vector_store.get_collection(self.COLLECTION_NAME)

            ids = [img["image_id"] for img in images]
            embeddings = all_embeddings
            documents = descriptions
            metadatas = [
                {
                    "keywords": ",".join(self._extract_keywords(desc)),
                    "page": img.get("page_num"),
                    "source_file": Path(img["image_path"]).name,
                    "manual_id": img.get("manual_id", ""),
                    "source": img.get("source", ""),
                }
                for img, desc in zip(images, descriptions)
            ]

            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"批量存入向量库成功: {len(images)} 条")
            except Exception as e:
                raise RuntimeError(f"批量存入向量库失败: {e}") from e

        # 第四步：更新元数据索引
        for img, desc in zip(images, descriptions):
            image_id = img["image_id"]
            self.metadata_index["images"][image_id] = {
                "path": str(Path(img["image_path"]).absolute()),
                "page": img.get("page_num"),
                "source_file": Path(img["image_path"]).name,
                "description": desc,
                "keywords": self._extract_keywords(desc),
                "manual_id": img.get("manual_id", ""),
                "source": img.get("source", ""),
            }
            added_count += 1

            if progress_callback:
                progress_callback(added_count, total, f"处理完成: {image_id}")

        self._save_metadata_index()
        logger.info(f"批量添加完成: {added_count}/{total} 张图片")
        return added_count

    async def _add_to_vector_store(
        self,
        image_id: str,
        description: str,
        keywords: List[str],
        image_info: Dict[str, Any]
    ):
        """将图片描述添加到向量库"""
        try:
            # 生成 embedding
            embedding = await self.embedding_model.embed(description)

            collection = self.vector_store.get_collection(self.COLLECTION_NAME)

            collection.add(
                ids=[image_id],
                embeddings=[embedding],
                documents=[description],
                metadatas=[{
                    "keywords": ",".join(keywords),
                    "page": image_info.get("page"),
                    "source_file": image_info.get("source_file")
                }]
            )

            logger.debug(f"Added image embedding to vector store: {image_id}")

        except Exception as e:
            logger.error(f"Failed to add image to vector store: {e}")

    async def _generate_description(self, image_path: str) -> str:
        """使用LLM生成图片描述"""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请简要描述这张图片的内容，包括：1) 图片类型（示意图、流程图、实物图、界面截图等）；2) 主要展示的内容；3) 可能的用途或关联的产品功能。请用50-100字描述。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            }

            description = await self.llm_client.chat_with_vision(
                messages=[message],
                temperature=0.3,
                model=self.vision_model
            )

            return description

        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return "图片描述生成失败"

    @property
    def vision_model(self) -> str:
        """获取视觉模型名称"""
        return self.settings.VISION_MODEL

    def _extract_keywords(self, description: str) -> List[str]:
        """从描述中提取关键词"""
        keywords = []

        # 产品/部件相关
        product_keywords = [
            "电池", "指示灯", "按钮", "开关", "屏幕", "表带", "充电器",
            "传感器", "接口", "按键", "麦克风", "扬声器", "摄像头", "天线",
            "电源", "适配器", "底座", "支架", "外壳", "面板", "显示屏",
            "LED", "USB", "充电线", "防水", "蓝牙", "WiFi", "GPS"
        ]

        # 状态/动作相关
        state_keywords = [
            "闪烁", "亮起", "充电", "放电", "故障", "待机", "运行",
            "连接", "断开", "同步", "校准", "设置", "配置", "安装",
            "拆卸", "更换", "清洁", "维护", "重启", "关机", "开机"
        ]

        # 类型相关
        type_keywords = [
            "示意图", "流程图", "实物图", "界面图", "接线图",
            "电路图", "结构图", "爆炸图", "安装图", "操作图",
            "界面截图", "状态图", "位置图", "尺寸图"
        ]

        for kw in product_keywords + state_keywords + type_keywords:
            if kw in description:
                keywords.append(kw)

        return list(set(keywords))[:15]

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索相关图片（向量检索）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            List[Dict]: 图片信息列表
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []

        try:
            # 生成查询向量
            query_embedding = await self.embedding_model.embed(query)

            # 向量检索
            collection = self.vector_store.get_collection(self.COLLECTION_NAME)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # 整理结果
            images = []
            if results and results.get("documents"):
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 0

                    # 获取完整元数据
                    img_metadata = self.metadata_index["images"].get(doc_id, {})

                    images.append({
                        "id": doc_id,
                        "path": img_metadata.get("path", ""),
                        "description": doc,
                        "keywords": metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
                        "page": metadata.get("page"),
                        "score": 1.0 - distance  # 转换为相似度
                    })

            return images

        except Exception as e:
            logger.error(f"Image search error: {e}")
            return []

    def get_image_by_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取图片信息"""
        return self.metadata_index["images"].get(image_id)

    def get_image_path(self, image_id: str) -> Optional[str]:
        """获取图片路径"""
        img = self.get_image_by_id(image_id)
        return img.get("path") if img else None

    def get_collection_count(self) -> int:
        """获取向量库中图片数量"""
        if not self.vector_store:
            return 0
        try:
            collection = self.vector_store.get_collection(self.COLLECTION_NAME)
            return collection.count()
        except Exception:
            return 0

    async def batch_add_from_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        manual_descriptions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """从PDF批量提取并索引图片"""
        from core.rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        images = loader.extract_images_from_pdf(pdf_path, output_dir)

        results = []
        for img_info in images:
            result = await self.add_image(
                image_path=img_info["path"],
                image_id=img_info["id"],
                page_num=img_info.get("page"),
                manual_description=manual_descriptions.get(img_info["id"])
            )
            results.append(result)

        return results

    def rebuild_index(self):
        """重建索引（清空向量库中的图片数据）"""
        if not self.vector_store:
            return False

        try:
            self.vector_store.delete_collection(self.COLLECTION_NAME)
            self.vector_store.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"dimension": self.settings.EMBEDDING_DIMENSION}
            )
            self.metadata_index = {"images": {}}
            self._save_metadata_index()
            logger.info("Image index rebuilt successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            return False
