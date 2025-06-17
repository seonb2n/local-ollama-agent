"""
FastAPI 라우터 - API 엔드포인트 정의
"""
import os
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    CodeExecutionRequest,
    CodeExecutionResponse,
    FileListResponse
)
from app.services.context_manager import context_manager
from app.services.ollama_service import ollama_service
from app.config import settings

router = APIRouter()


@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest, session_id: Optional[str] = None):
    """컨텍스트를 활용한 코드 생성 API"""
    try:
        start_time = time.time()

        # 세션이 없으면 새로 생성
        if not session_id:
            session_id = context_manager.create_session()

        # 요청 로깅
        print(f"🤖 코드 생성 요청 (세션: {session_id[:8]}...): {request.description[:50]}...")

        # 컨텍스트를 활용한 코드 생성
        generated_code = await ollama_service.generate_code_with_context(
            description=request.description,
            language=request.language.value,
            framework=request.framework,
            session_id=session_id
        )

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.language.value}_app_{timestamp}.{_get_file_extension(request.language.value)}"
        file_path = os.path.join(settings.generated_code_path, filename)

        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(generated_code)

        # 의존성 추출
        dependencies = _extract_dependencies(generated_code, request.language.value)

        execution_time = time.time() - start_time

        # 컨텍스트에 대화 기록 추가
        context_manager.add_conversation_turn(
            session_id=session_id,
            user_request=request.description,
            assistant_response=f"코드를 생성했습니다: {filename}",
            generated_code=generated_code,
            filename=filename,
            metadata={
                "language": request.language.value,
                "framework": request.framework,
                "dependencies": dependencies,
                "execution_time": execution_time
            }
        )

        print(f"✅ 코드 생성 완료: {filename} ({execution_time:.1f}초)")

        return CodeGenerationResponse(
            success=True,
            message="코드가 성공적으로 생성되었습니다.",
            code=generated_code,
            filename=filename,
            file_path=file_path,
            dependencies=dependencies,
            execution_time=execution_time
        )

    except Exception as e:
        print(f"❌ 코드 생성 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"코드 생성 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """생성된 코드 실행 API (Python만 지원)"""
    try:
        file_path = os.path.join(settings.generated_code_path, request.filename)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"파일을 찾을 수 없습니다: {request.filename}"
            )

        # Python 파일만 실행 지원
        if not request.filename.endswith('.py'):
            raise HTTPException(
                status_code=400,
                detail="현재는 Python 파일만 실행 가능합니다."
            )

        # 코드 실행
        start_time = time.time()

        import subprocess
        result = subprocess.run(
            ["python", file_path] + request.arguments,
            capture_output=True,
            text=True,
            timeout=30,  # 30초 타임아웃
            cwd=settings.generated_code_path
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            return CodeExecutionResponse(
                success=True,
                output=result.stdout,
                error=None,
                execution_time=execution_time
            )
        else:
            return CodeExecutionResponse(
                success=False,
                output=result.stdout,
                error=result.stderr,
                execution_time=execution_time
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="코드 실행 시간이 초과되었습니다 (30초 제한)."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"코드 실행 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/files", response_model=FileListResponse)
async def list_generated_files():
    """생성된 파일 목록 조회"""
    try:
        files = []
        code_path = settings.generated_code_path

        if os.path.exists(code_path):
            for filename in os.listdir(code_path):
                file_path = os.path.join(code_path, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files.append({
                        "filename": filename,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": os.path.splitext(filename)[1]
                    })

        # 최신 파일부터 정렬
        files.sort(key=lambda x: x["created"], reverse=True)

        return FileListResponse(
            files=files,
            total_count=len(files)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_file(filename: str):
    """생성된 파일 다운로드"""
    try:
        file_path = os.path.join(settings.generated_code_path, filename)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"파일을 찾을 수 없습니다: {filename}"
            )

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 다운로드 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/sessions")
async def create_session(user_id: Optional[str] = None):
    """새 대화 세션 생성"""
    session_id = context_manager.create_session(user_id)
    return {"session_id": session_id, "message": "새 세션이 생성되었습니다."}


@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    session_info = context_manager.get_session_history(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return session_info


@router.get("/sessions")
async def list_sessions(user_id: Optional[str] = None):
    """세션 목록 조회"""
    sessions = context_manager.get_all_sessions(user_id)
    return {"sessions": sessions, "total": len(sessions)}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    success = context_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"message": "세션이 삭제되었습니다."}


@router.get("/sessions/{session_id}/context")
async def get_session_context(session_id: str):
    """세션의 현재 컨텍스트 조회"""
    context = context_manager.get_context_for_llm(session_id, include_code=False)
    if not context:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"session_id": session_id, "context": context}


def _get_file_extension(language: str) -> str:
    """언어별 파일 확장자 반환"""
    extensions = {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "java": "java",
        "go": "go",
        "rust": "rs"
    }
    return extensions.get(language, "txt")


def _extract_dependencies(code: str, language: str) -> List[str]:
    """코드에서 의존성 추출 (간단한 파싱)"""
    dependencies = []

    if language == "python":
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # import numpy -> numpy
                # from fastapi import FastAPI -> fastapi
                if line.startswith('import '):
                    dep = line.replace('import ', '').split()[0].split('.')[0]
                else:  # from ... import
                    dep = line.split('from ')[1].split(' import')[0].split('.')[0]

                # 표준 라이브러리 제외
                if dep not in ['os', 'sys', 'json', 'time', 'datetime', 're', 'random']:
                    dependencies.append(dep)

    return list(set(dependencies))  # 중복 제거