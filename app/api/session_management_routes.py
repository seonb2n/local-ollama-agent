"""
세션 관리 관련 라우터 - 세션 생성, 조회, 삭제, 컨텍스트 관리
"""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.context_management_service import context_service

router = APIRouter(prefix="/sessions", tags=["Session Management"])


# Request/Response 모델들
class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    session_name: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str]
    created_at: str
    message: str


class ImprovementSettingsRequest(BaseModel):
    enabled: Optional[bool] = None
    quality_threshold: Optional[float] = None
    max_iterations: Optional[int] = None


# 세션 관리 API들
@router.post("", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest = SessionCreateRequest()):
    """새 대화 세션 생성"""
    try:
        session_id = context_service.create_session(request.user_id)

        # 세션 정보 조회하여 생성 시간 포함
        session_info = context_service.get_session_history(session_id)
        created_at = session_info.get('created_at', '') if session_info else ''

        return SessionResponse(
            session_id=session_id,
            user_id=request.user_id,
            created_at=created_at,
            message="새 세션이 생성되었습니다."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 생성 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/{session_id}")
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    try:
        session_info = context_service.get_session_history(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        return {
            "session_id": session_id,
            "session_info": session_info,
            "total_conversations": len(session_info.get('conversations', [])),
            "last_activity": session_info.get('updated_at', ''),
            "created_at": session_info.get('created_at', '')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("")
async def list_sessions(user_id: Optional[str] = None, limit: int = 50, offset: int = 0):
    """세션 목록 조회"""
    try:
        all_sessions = context_service.get_all_sessions(user_id)

        # 페이지네이션 적용
        total = len(all_sessions)
        sessions = all_sessions[offset:offset + limit]

        return {
            "sessions": sessions,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_next": offset + limit < total,
            "has_prev": offset > 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    try:
        success = context_service.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        return {
            "message": "세션이 삭제되었습니다.",
            "session_id": session_id,
            "deleted_at": context_service.get_current_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/{session_id}/context")
async def get_session_context(session_id: str, include_code: bool = False):
    """세션의 현재 컨텍스트 조회"""
    try:
        context = context_service.get_context_for_llm(session_id, include_code=include_code)
        if not context:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        return {
            "session_id": session_id,
            "context": context,
            "include_code": include_code,
            "context_length": len(context),
            "retrieved_at": context_service.get_current_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"컨텍스트 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/{session_id}/conversations")
async def get_session_conversations(session_id: str, limit: int = 20, offset: int = 0):
    """세션의 대화 기록 조회"""
    try:
        session_info = context_service.get_session_history(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        conversations = session_info.get('conversations', [])
        total = len(conversations)

        # 최신 대화부터 정렬하고 페이지네이션 적용
        conversations.reverse()  # 최신순으로 정렬
        paginated_conversations = conversations[offset:offset + limit]

        return {
            "session_id": session_id,
            "conversations": paginated_conversations,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_next": offset + limit < total,
            "has_prev": offset > 0
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"대화 기록 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/{session_id}/clear")
async def clear_session_history(session_id: str):
    """세션의 대화 기록 초기화 (세션은 유지, 대화만 삭제)"""
    try:
        # 세션 존재 확인
        session_info = context_service.get_session_history(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        # 대화 기록 초기화 (구체적인 구현은 context_service에 메서드 추가 필요)
        success = context_service.clear_session_conversations(session_id)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="대화 기록 초기화에 실패했습니다."
            )

        return {
            "message": "대화 기록이 초기화되었습니다.",
            "session_id": session_id,
            "cleared_at": context_service.get_current_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"대화 기록 초기화 중 오류가 발생했습니다: {str(e)}"
        )

