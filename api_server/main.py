import asyncio
import os
from contextlib import asynccontextmanager

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_server.config import Settings
from api_server.service import TTSService
from api_server.file_manager import FileManager
from api_server.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时清理资源。"""
    settings = Settings()

    # 初始化单例
    service = TTSService.get_instance(settings)
    file_manager = FileManager(settings.temp_dir, settings.temp_file_ttl_seconds)

    # 模型加载是 CPU/磁盘密集型操作，放入线程池避免阻塞事件循环
    print(">> 正在加载 IndexTTS2 模型（可能需要几分钟）...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, service.load_model)
    print(">> 模型加载完成。")

    # 挂载到 app.state
    app.state.service = service
    app.state.file_manager = file_manager

    # 后台定时清理过期临时文件
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(60)
            file_manager.cleanup_expired()

    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield  # 应用运行期间

    # 关闭清理
    cleanup_task.cancel()
    await loop.run_in_executor(None, service.unload_model)
    print(">> 模型已卸载。")


def create_app() -> FastAPI:
    app = FastAPI(
        title="IndexTTS2 API",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app
