"""IndexTTS2 API 服务启动脚本。

用法:
    python api.py
    python api.py --host 127.0.0.1 --port 9000
    INDEXTTS_PORT=9000 python api.py
"""

import argparse
import uvicorn

from api_server.config import Settings


def main():
    settings = Settings()

    parser = argparse.ArgumentParser(description="IndexTTS2 API Server")
    parser.add_argument("--host", type=str, default=settings.host, help="监听地址")
    parser.add_argument("--port", type=int, default=settings.port, help="监听端口")
    args = parser.parse_args()

    uvicorn.run(
        "api_server.main:create_app",
        host=args.host,
        port=args.port,
        factory=True,
    )


if __name__ == "__main__":
    main()
