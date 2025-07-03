"""
코드 생성 관련 라우터 - 코드 생성, 실행, 파일 관리
"""
import os
import time
import subprocess
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.facade.code_generation_facade_service import code_generation_facade
from ..services.context_management_service import context_service
from ..models import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    CodeExecutionRequest,
    CodeExecutionResponse,
    FileListResponse
)

import sys

from ..util.sanitize_string import clean_markdown_code_blocks, extract_code_only

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ..config import settings

code_router = APIRouter(prefix="/code", tags=["Code Generation"])


@code_router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest, session_id: Optional[str] = None):
    """컨텍스트를 활용한 코드 생성 API"""
    try:
        start_time = time.time()
        print(f"🤖 코드 생성 요청 (세션: {session_id[:8]}...): {request.description[:50]}...")

        # 기존 세션 파일 확인
        existing_filename = context_service.get_session_file(session_id) if session_id else None
        existing_file_path = None
        
        if existing_filename:
            existing_file_path = os.path.join(settings.generated_code_path, existing_filename)
            if not os.path.exists(existing_file_path):
                existing_file_path = None
                existing_filename = None
        
        generated_code, description = await code_generation_facade.generate_code_with_context(
            description=request.description,
            language=request.language.value,
            framework=request.framework,
            session_id=session_id,
            existing_file_path=existing_file_path
        )

        # 파일명 결정 및 파일 쓰기
        is_modification = bool(existing_filename)
        
        if is_modification:
            # 기존 파일 덮어쓰기
            filename = existing_filename
            file_path = os.path.join(settings.generated_code_path, filename)
            print(f"📝 기존 파일 수정: {filename}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(generated_code)
        else:
            # 새 파일 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{request.language.value}_app_{timestamp}.{_get_file_extension(request.language.value)}"
            file_path = os.path.join(settings.generated_code_path, filename)
            
            if session_id:
                context_service.set_session_file(session_id, filename)
            print(f"📄 새 파일 생성: {filename}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(generated_code)

        dependencies = _extract_dependencies(generated_code, request.language.value)

        execution_time = time.time() - start_time

        # 컨텍스트에 대화 기록 추가
        action_message = f"코드를 {'수정' if is_modification else '생성'}했습니다: {filename}"
        context_service.add_conversation(
            session_id=session_id,
            user_request=request.description,
            assistant_response=action_message,
            generated_code=generated_code,
            filename=filename,
            metadata={
                "language": request.language.value,
                "framework": request.framework,
                "dependencies": dependencies,
                "execution_time": execution_time,
                "is_modification": is_modification
            }
        )

        print(f"✅ 코드 생성 완료: {filename} ({execution_time:.1f}초)")

        success_message = f"코드가 성공적으로 {'수정' if is_modification else '생성'}되었습니다."
        
        return CodeGenerationResponse(
            success=True,
            message=success_message,
            code=generated_code,
            description=description,
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


@code_router.post("/execute", response_model=CodeExecutionResponse)
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

        start_time = time.time()

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


@code_router.get("/files", response_model=FileListResponse)
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


@code_router.get("/download/{filename}")
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


@code_router.delete("/files/{filename}")
async def delete_file(filename: str):
    """생성된 파일 삭제"""
    try:
        file_path = os.path.join(settings.generated_code_path, filename)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"파일을 찾을 수 없습니다: {filename}"
            )

        os.remove(file_path)
        return {"message": f"파일 '{filename}'이 삭제되었습니다."}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@code_router.get("/file-info/{filename}")
async def get_file_info(filename: str):
    """파일 상세 정보 조회"""
    try:
        file_path = os.path.join(settings.generated_code_path, filename)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"파일을 찾을 수 없습니다: {filename}"
            )

        stat = os.stat(file_path)

        # 파일 내용 미리보기 (처음 500자)
        with open(file_path, 'r', encoding='utf-8') as f:
            preview = f.read(500)

        return {
            "filename": filename,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": os.path.splitext(filename)[1],
            "preview": preview,
            "line_count": preview.count('\n') + 1,
            "dependencies": _extract_dependencies(preview, _get_language_from_extension(filename))
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )


# 유틸리티 함수들
def _get_file_extension(language: str) -> str:
    """언어별 파일 확장자 반환"""
    extensions = {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "java": "java",
        "go": "go",
        "rust": "rs",
        "cpp": "cpp",
        "c": "c",
        "csharp": "cs"
    }
    return extensions.get(language, "txt")


def _get_language_from_extension(filename: str) -> str:
    """파일 확장자에서 언어 추출"""
    extension = os.path.splitext(filename)[1].lower()
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp"
    }
    return language_map.get(extension, "unknown")


def _extract_dependencies(code: str, language: str) -> list[str]:
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
                    parts = line.split('from ')
                    if len(parts) > 1:
                        dep = parts[1].split(' import')[0].split('.')[0]
                    else:
                        continue

                # 표준 라이브러리 제외
                if dep not in ['os', 'sys', 'json', 'time', 'datetime', 're', 'random', 'math', 'collections']:
                    dependencies.append(dep)

    elif language == "javascript" or language == "typescript":
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if 'import' in line and 'from' in line:
                # import { something } from 'package'
                if "'" in line:
                    dep = line.split("'")[1]
                elif '"' in line:
                    dep = line.split('"')[1]
                else:
                    continue

                # 상대 경로 제외
                if not dep.startswith('.') and not dep.startswith('/'):
                    dependencies.append(dep)
            elif line.startswith('const ') and 'require(' in line:
                # const package = require('package')
                if "'" in line:
                    dep = line.split("'")[1]
                elif '"' in line:
                    dep = line.split('"')[1]
                else:
                    continue

                if not dep.startswith('.') and not dep.startswith('/'):
                    dependencies.append(dep)

    return list(set(dependencies))  # 중복 제거