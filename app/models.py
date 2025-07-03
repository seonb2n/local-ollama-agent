from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class CodeLanguage(str, Enum):
    """지원하는 프로그래밍 언어"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"


class ProjectType(str, Enum):
    """프로젝트 타입"""
    WEB_APP = "web_app"
    API = "api"
    SCRIPT = "script"
    CRAWLER = "crawler"
    DATA_ANALYSIS = "data_analysis"


class CodeGenerationRequest(BaseModel):
    """코드 생성 요청 모델"""
    description: str = Field(..., description="생성할 앱/코드에 대한 설명")
    language: CodeLanguage = Field(default=CodeLanguage.PYTHON, description="프로그래밍 언어")
    project_type: Optional[ProjectType] = Field(default=None, description="프로젝트 타입")
    requirements: Optional[List[str]] = Field(default=[], description="추가 요구사항")
    framework: Optional[str] = Field(default=None, description="사용할 프레임워크 (예: fastapi, flask)")


class CodeGenerationResponse(BaseModel):
    """코드 생성 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    code: Optional[str] = Field(default=None, description="생성된 코드")
    description: Optional[str] = Field(default=None, description="설명")
    filename: Optional[str] = Field(default=None, description="저장된 파일명")
    file_path: Optional[str] = Field(default=None, description="파일 경로")
    dependencies: Optional[List[str]] = Field(default=[], description="필요한 패키지 목록")
    execution_time: Optional[float] = Field(default=None, description="실행 시간 (초)")


class CodeExecutionRequest(BaseModel):
    """코드 실행 요청 모델"""
    filename: str = Field(..., description="실행할 파일명")
    arguments: Optional[List[str]] = Field(default=[], description="실행 인자")


class CodeExecutionResponse(BaseModel):
    """코드 실행 응답 모델"""
    success: bool = Field(..., description="실행 성공 여부")
    output: Optional[str] = Field(default=None, description="실행 결과")
    error: Optional[str] = Field(default=None, description="에러 메시지")
    execution_time: Optional[float] = Field(default=None, description="실행 시간")


class FileListResponse(BaseModel):
    """파일 목록 응답 모델"""
    files: List[Dict[str, Any]] = Field(..., description="파일 목록")
    total_count: int = Field(..., description="총 파일 개수")


class HealthCheckResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str = Field(..., description="서비스 상태")
    timestamp: datetime = Field(..., description="체크 시간")
    ollama_status: str = Field(..., description="Ollama 연결 상태")
    available_models: List[str] = Field(..., description="사용 가능한 모델 목록")


class AgentStatus(BaseModel):
    """에이전트 상태 모델"""
    is_busy: bool = Field(..., description="작업 중 여부")
    current_task: Optional[str] = Field(default=None, description="현재 작업")
    queue_size: int = Field(default=0, description="대기 중인 작업 수")
    last_activity: Optional[datetime] = Field(default=None, description="마지막 활동 시간")