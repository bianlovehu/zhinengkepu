"""
文本分块器
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """文本块"""
    content: str
    chunk_id: str
    metadata: Dict[str, Any]


class TextChunker:
    """
    文本分块器

    支持多种分块策略：
    - 固定大小分块
    - 语义分块（按段落/标题）
    - 递归分块
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 700
    ):
        """
        Args:
            chunk_size: 目标块大小（字符数）
            overlap: 块之间的重叠大小
            min_chunk_size: 最小块大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max(max_chunk_size, chunk_size)

    def chunk_text(
        self,
        text: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """
        对文本进行分块

        Args:
            text: 待分块文本
            source: 来源标识
            metadata: 额外元数据

        Returns:
            List[TextChunk]: 文本块列表
        """
        metadata = metadata or {}

        # 获取文档标题（用于添加到chunk上下文）
        doc_title = metadata.get("title", "")
        full_title = metadata.get("full_title", source)

        raw_chunks = self._semantic_chunk(text)
        bounded_chunks = []
        for raw_chunk in raw_chunks:
            bounded_chunks.extend(self._split_oversized_chunk(raw_chunk))

        chunks = []
        for i, chunk_text in enumerate(bounded_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            prefixed_chunk = self._prefix_chunk_with_title(chunk_text, doc_title, full_title)
            chunks.append(TextChunk(
                content=prefixed_chunk,
                chunk_id=f"{source}_{i}" if source else f"chunk_{i}",
                metadata={
                    **metadata,
                    "source": source,
                    "chunk_index": i,
                    "title": doc_title,
                    "char_count": len(prefixed_chunk),
                }
            ))

        logger.info(f"Chunked into {len(chunks)} pieces")
        return chunks

    def _prefix_chunk_with_title(
        self,
        chunk_text: str,
        doc_title: str,
        full_title: str
    ) -> str:
        """
        在chunk前添加文档标题作为上下文

        这有助于向量检索时能通过标题匹配到相关内容
        """
        if not doc_title:
            return chunk_text

        # 构建前缀：文档类型/标题
        prefix = f"[{full_title}] "

        # 检查chunk是否已包含标题前缀（避免重复）
        if chunk_text.startswith(prefix):
            return chunk_text

        # 在chunk开头添加标题前缀
        return prefix + chunk_text

    def _semantic_chunk(self, text: str) -> List[str]:
        """
        语义分块

        按段落、标题等语义边界切分
        """
        blocks = self._paragraph_blocks(text)
        chunks = []
        current = []
        current_len = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            is_heading = self._is_heading(block)
            would_exceed = current_len + len(block) + 2 > self.chunk_size

            if current and (is_heading or would_exceed):
                chunks.append("\n\n".join(current).strip())
                current = []
                current_len = 0

            current.append(block)
            current_len += len(block) + 2

        if current:
            chunks.append("\n\n".join(current).strip())

        return chunks if chunks else [text.strip()]

    def _paragraph_blocks(self, text: str) -> List[str]:
        """按标题和段落提取语义块。"""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"(?m)^\s*(#{1,6}\s+)", r"\1", normalized)
        raw_blocks = re.split(r"\n\s*\n", normalized)

        blocks = []
        for raw in raw_blocks:
            raw = raw.strip()
            if not raw:
                continue
            # 同一段里混入多个 Markdown 标题时继续拆开。
            parts = re.split(r"(?=\n?#{1,6}\s+)", raw)
            blocks.extend(part.strip() for part in parts if part.strip())
        return blocks

    def _is_heading(self, text: str) -> bool:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        return bool(
            first_line.startswith("#") or
            re.match(r"^\d+[\.\、]\s+", first_line) or
            re.match(r"^[A-Z][A-Z0-9\s\-:/]{4,}$", first_line) or
            re.match(r"^<h[1-6]>", first_line, re.IGNORECASE)
        )

    def _split_oversized_chunk(self, text: str) -> List[str]:
        """保证最终 chunk 不超过 max_chunk_size。"""
        text = text.strip()
        if len(text) <= self.max_chunk_size:
            return [text] if text else []

        sentences = self._split_sentences(text)
        chunks = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > self.max_chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._fixed_text_windows(sentence))
                continue
            if current and len(current) + len(sentence) + 1 > self.chunk_size:
                chunks.append(current.strip())
                current = self._overlap_tail(current) + sentence
            else:
                current = f"{current} {sentence}".strip()

        if current:
            chunks.append(current.strip())

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？!?；;])\s+|(?<=\.)\s+(?=[A-Z#])|\n+", text)
        return [part for part in parts if part.strip()]

    def _fixed_text_windows(self, text: str) -> List[str]:
        chunks = []
        start = 0
        step_back = min(self.overlap, max(0, self.chunk_size // 4))
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            window = text[start:end]
            if end < len(text):
                boundary = max(
                    window.rfind("。"),
                    window.rfind("！"),
                    window.rfind("？"),
                    window.rfind(";"),
                    window.rfind(". "),
                    window.rfind(" ")
                )
                if boundary >= self.min_chunk_size:
                    window = window[:boundary + 1]
                    end = start + len(window)
            chunks.append(window.strip())
            if end >= len(text):
                break
            start = max(end - step_back, start + 1)
        return chunks

    def _overlap_tail(self, text: str) -> str:
        if self.overlap <= 0 or len(text) <= self.overlap:
            return ""
        return text[-self.overlap:].lstrip() + " "

    def _fixed_chunk(
        self,
        text: str,
        source: str,
        metadata: Dict[str, Any],
        doc_title: str = "",
        full_title: str = ""
    ) -> List[TextChunk]:
        """固定大小分块。保留兼容入口，实际使用有上限的窗口切分。"""
        chunks = []
        for chunk_index, chunk_text in enumerate(self._fixed_text_windows(text)):
            chunks.append(TextChunk(
                content=self._prefix_chunk_with_title(chunk_text, doc_title, full_title),
                chunk_id=f"{source}_{chunk_index}" if source else f"chunk_{chunk_index}",
                metadata={**metadata, "source": source, "chunk_index": chunk_index, "title": doc_title}
            ))

        return chunks

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[TextChunk]:
        """
        批量分块文档

        Args:
            documents: [{"content": str, "source": str, "metadata": dict}]

        Returns:
            List[TextChunk]: 所有文本块
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(
                text=doc.get("content", ""),
                source=doc.get("source", ""),
                metadata=doc.get("metadata", {})
            )
            all_chunks.extend(chunks)

        return all_chunks
