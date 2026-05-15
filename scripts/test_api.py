"""
测试脚本

Usage:
    python scripts/test_api.py
"""
import sys
import asyncio
import base64
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from config import get_settings


def _api_base_url(settings) -> str:
    host = "127.0.0.1" if settings.API_HOST in {"0.0.0.0", "::"} else settings.API_HOST
    return f"http://{host}:{settings.API_PORT}"


async def test_text_chat():
    """测试文本对话"""
    settings = get_settings()

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{_api_base_url(settings)}/chat/",
            json={
                "question": "我的DCB107电钻指示灯闪烁是什么意思？"
            },
            headers={
                "Authorization": f"Bearer {settings.API_TOKEN}",
                "Content-Type": "application/json"
            }
        )

        print("状态码:", response.status_code)
        print("响应:", response.json())


async def test_multimodal_chat():
    """测试多模态对话"""
    settings = get_settings()

    # 读取示例图片
    image_path = Path(__file__).parent.parent / "examples" / "test_image.png"

    if image_path.exists():
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{_api_base_url(settings)}/chat/",
                json={
                    "question": "请帮我看一下这张图片的问题",
                    "images": [f"data:image/png;base64,{image_b64}"],
                    "session_id": "test_session_001"
                },
                headers={
                    "Authorization": f"Bearer {settings.API_TOKEN}",
                    "Content-Type": "application/json"
                }
            )

            print("状态码:", response.status_code)
            print("响应:", response.json())
    else:
        print(f"示例图片不存在: {image_path}")


async def test_multi_turn():
    """测试多轮对话"""
    settings = get_settings()

    async with httpx.AsyncClient(timeout=120.0) as client:
        session_id = "multi_turn_test"

        # 第一轮
        r1 = await client.post(
            f"{_api_base_url(settings)}/chat/",
            json={
                "question": "我想更换表带",
                "session_id": session_id
            },
            headers={
                "Authorization": f"Bearer {settings.API_TOKEN}"
            }
        )
        print("第1轮:", r1.json().get("data", {}).get("answer", "")[:100])

        # 第二轮（追问）
        r2 = await client.post(
            f"{_api_base_url(settings)}/chat/",
            json={
                "question": "有其他尺寸可选吗？",
                "session_id": session_id
            },
            headers={
                "Authorization": f"Bearer {settings.API_TOKEN}"
            }
        )
        print("第2轮:", r2.json().get("data", {}).get("answer", "")[:100])


async def main():
    print("=" * 50)
    print("开始测试API")
    print("=" * 50)

    try:
        print("\n--- 测试文本对话 ---")
        await test_text_chat()

        print("\n--- 测试多轮对话 ---")
        await test_multi_turn()

        print("\n--- 测试多模态对话 ---")
        await test_multimodal_chat()

    except Exception as e:
        print(f"测试失败: {type(e).__name__}: {e!r}")


if __name__ == "__main__":
    asyncio.run(main())
