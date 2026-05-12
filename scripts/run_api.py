"""
启动API服务脚本

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080 --host 0.0.0.0
"""
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="启动多模态客服智能体API")
    parser.add_argument("--host", "-H", default=None, help="监听地址")
    parser.add_argument("--port", "-p", type=int, default=None, help="监听端口")
    parser.add_argument("--reload", "-r", action="store_true", help="开发模式热重载")
    parser.add_argument("--workers", "-w", type=int, default=1, help="工作进程数")

    args = parser.parse_args()
    settings = get_settings()

    host = args.host or settings.API_HOST
    port = args.port or settings.API_PORT
    reload = args.reload

    logger.info("=" * 50)
    logger.info("启动多模态客服智能体API")
    logger.info("=" * 50)
    logger.info(f"地址: {host}:{port}")
    logger.info(f"文档: http://{host}:{port}/docs")
    logger.info(f"热重载: {'开启' if reload else '关闭'}")
    logger.info("=" * 50)

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=args.workers if not reload else 1
    )


if __name__ == "__main__":
    main()
