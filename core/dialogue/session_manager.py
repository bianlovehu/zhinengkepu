"""
会话管理器
"""
import time
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

from config import get_settings

logger = logging.getLogger(__name__)


class SessionManager:
    """
    会话管理器

    负责多轮对话的状态维护、历史记录管理
    """

    _instance = None

    def __init__(self):
        self.settings = get_settings()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._init_from_storage()

    @classmethod
    def get_instance(cls) -> "SessionManager":
        """单例获取"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_from_storage(self):
        """从存储初始化（可选：可扩展为Redis等）"""
        pass

    def create_session(self, session_id: str) -> Dict[str, Any]:
        """
        创建新会话

        Args:
            session_id: 会话ID

        Returns:
            Dict: 会话信息
        """
        if session_id in self.sessions:
            return self.sessions[session_id]

        session = {
            "session_id": session_id,
            "created_at": time.time(),
            "last_active": time.time(),
            "messages": [],
            "context": {}
        }

        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")

        return session

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[str]) -> Dict[str, Any]:
        """获取或创建会话"""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session["last_active"] = time.time()
            return session

        new_id = session_id or f"session_{int(time.time() * 1000)}"
        return self.create_session(new_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        images: List[bytes] = None
    ):
        """
        添加消息到会话

        Args:
            session_id: 会话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            images: 可选的图片
        """
        session = self.get_or_create_session(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }

        if images:
            message["images_count"] = len(images)

        session["messages"].append(message)
        session["last_active"] = time.time()

        # 限制历史长度
        max_history = self.settings.MAX_SESSION_HISTORY
        if len(session["messages"]) > max_history * 2:  # 双向消息
            session["messages"] = session["messages"][-max_history * 2:]

    def get_history(
        self,
        session_id: str,
        last_n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        获取对话历史

        Args:
            session_id: 会话ID
            last_n: 只返回最近N条消息

        Returns:
            List[Dict]: 消息列表
        """
        session = self.get_session(session_id)
        if not session:
            return []

        messages = session.get("messages", [])

        if last_n:
            return messages[-last_n:]

        return messages

    def update_context(self, session_id: str, key: str, value: Any):
        """更新会话上下文"""
        session = self.get_or_create_session(session_id)
        session["context"][key] = value

    def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        """获取会话上下文"""
        session = self.get_session(session_id)
        if not session:
            return default
        return session.get("context", {}).get(key, default)

    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")

    def cleanup_expired(self):
        """清理过期会话"""
        current_time = time.time()
        expire_seconds = self.settings.SESSION_EXPIRE_SECONDS

        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session["last_active"] > expire_seconds
        ]

        for sid in expired:
            self.clear_session(sid)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def list_sessions(self) -> List[str]:
        """列出所有会话ID"""
        return list(self.sessions.keys())

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息摘要"""
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session_id,
            "message_count": len(session.get("messages", [])),
            "created_at": session.get("created_at"),
            "last_active": session.get("last_active"),
            "context_keys": list(session.get("context", {}).keys())
        }
