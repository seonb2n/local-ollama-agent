from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from app.api.code_generation_routes import code_router
from app.api.session_management_routes import session_router
from app.config import settings
from app.models import AgentStatus, HealthCheckResponse
from app.services.ollama_service import ollama_service

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    # 시작 시 실행
    logger.info("🚀 Code Generator Agent 시작 중...")

    # Ollama 서비스 초기화
    try:
        await ollama_service.initialize()
        logger.info("✅ Ollama 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"❌ Ollama 서비스 초기화 실패: {e}")

    yield

    # 종료 시 실행
    logger.info("🛑 Code Generator Agent 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="자연어로 명령하여 코드를 자동 생성하는 AI 에이전트",
    lifespan=lifespan
)

# API 라우터 등록
app.include_router(code_router, prefix="/api/v1", tags=["Code Generation"])
app.include_router(session_router, prefix="/api/v1", tags=["Code Generation"])

# CORS 설정
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 개발용 - 프로덕션에서는 제한 필요
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 전역 상태 (나중에 Redis 등으로 대체 가능)
agent_status = AgentStatus(
    is_busy=False,
    current_task=None,
    queue_size=0,
    last_activity=None
)


@app.get("/", response_model=dict)
async def root():
    """루트 엔드포인트"""
    return {
        "message": f"🤖 {settings.app_name} API",
        "version": settings.app_version,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        # Ollama 연결 테스트
        ollama_status = "connected" if await ollama_service.test_connection() else "disconnected"
        available_models = await ollama_service.get_available_models()

        return HealthCheckResponse(
            status="healthy" if ollama_status == "connected" else "unhealthy",
            timestamp=datetime.now(),
            ollama_status=ollama_status,
            available_models=available_models
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/status", response_model=AgentStatus)
async def get_agent_status():
    """에이전트 상태 조회"""
    return agent_status


@app.get("/models")
async def list_models():
    """사용 가능한 모델 목록"""
    available_models = await ollama_service.get_available_models()
    return {
        "available_models": available_models,
        "current_model": settings.default_model,
        "backup_model": settings.backup_model
    }


# 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "서버에서 오류가 발생했습니다.",
            "timestamp": datetime.now().isoformat()
        }
    )


# 개발용 실행 함수
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        reload_dirs=["app"] if settings.debug else None,  # app 폴더만 감시
        reload_excludes=[
            "../generated_code/*"
            "generated_code",
            "logs",
            "*.log",
            "__pycache__",
            "*.pyc",
            ".git"
        ] if settings.debug else None
    )