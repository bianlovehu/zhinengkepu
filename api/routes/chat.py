"""
聊天对话端点
"""
import base64
import time
import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Header, HTTPException, Request

from api.models.request import ChatRequest
from api.models.response import ChatResponse, ResponseData
from api.middleware.auth import verify_token
from core.llm.generator import LLMGenerator
from core.multimodal.intent_recognition import IntentRecognizer
from core.dialogue.session_manager import SessionManager
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
    x_client_type: Optional[str] = Header(None),
):
    """
    多模态对话交互端点

    - 验证认证令牌
    - 解析用户问题（文本+可选图片）
    - 进行意图识别、检索、回答生成
    - 返回结构化响应
    """
    request_id = x_request_id or str(uuid.uuid4())

    try:
        # 1. 认证验证
        if authorization:
            verify_token(authorization)

        # 2. 处理session_id
        session_id = request.session_id or str(uuid.uuid4())

        # 3. 获取会话管理器
        session_mgr = SessionManager.get_instance()

        # 4. 图片处理（Base64 -> bytes）
        images_bytes = []
        if request.images:
            for img_b64 in request.images:
                if img_b64.startswith("data:image"):
                    img_b64 = img_b64.split(",", 1)[1]
                images_bytes.append(base64.b64decode(img_b64))

        # 5. 调用核心处理逻辑
        logger.info(f"[{request_id}] Processing question: {request.question}")

        answer = await process_question(
            question=request.question,
            images=images_bytes,
            session_id=session_id,
            request_id=request_id
        )

        # 6. 更新会话历史
        session_mgr.add_message(session_id, "user", request.question, images_bytes)
        session_mgr.add_message(session_id, "assistant", answer)

        # 7. 返回响应
        return ChatResponse(
            code=0,
            msg="success",
            data=ResponseData(
                answer=answer,
                session_id=session_id,
                timestamp=int(time.time())
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def process_question(
    question: str,
    images: List[bytes],
    session_id: str,
    request_id: str
) -> str:
    """
    处理用户问题的核心逻辑

    流程：意图识别 -> RAG检索 -> 思维链拆解 -> 答案生成 -> 幻觉检测
    """
    # 获取会话历史
    session_mgr = SessionManager.get_instance()
    history = session_mgr.get_history(session_id)

    # 1. 意图识别
    intent_recognizer = IntentRecognizer()
    intent_result = await intent_recognizer.recognize(question, images)

    logger.info(f"[{request_id}] Intent: {intent_result}")

    # 2. RAG检索（传入分级关键词用于权重加成）
    from core.rag.retriever import RAGRetriever
    retriever = RAGRetriever()
    search_results = await retriever.retrieve(
        query=question,
        top_k=settings.TOP_K,
        need_images=intent_result.get("needs_images", False),
        keywords_by_level=intent_result.get("keywords", {"high": [], "medium": [], "low": []})
    )

    # 3. 思维链拆解（如需要）
    from core.dialogue.chain_of_thought import ChainOfThought
    cot = ChainOfThought()
    should_decompose, sub_questions = cot.should_decompose(question)

    if should_decompose:
        logger.info(f"[{request_id}] Decomposing into {len(sub_questions)} sub-questions")

    # 4. 答案生成
    generator = LLMGenerator()
    answer = await generator.generate(
        question=question,
        context=search_results,
        history=history,
        images=images,
        intent=intent_result
    )

    # 5. 幻觉检测（可选）
    from core.llm.hallucination_check import HallucinationChecker
    checker = HallucinationChecker()
    is_valid = await checker.check(answer, search_results)

    if not is_valid:
        logger.warning(f"[{request_id}] Potential hallucination detected")

    return answer
