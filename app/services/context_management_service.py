"""
컨텍스트 관리 서비스 - 대화 히스토리 및 세션 관리
"""
import logging
from typing import Optional, Dict, Any
from .context_manager import context_manager

logger = logging.getLogger(__name__)


class ContextManagementService:
    """컨텍스트 관리 전용 서비스"""

    def __init__(self):
        self.context_manager = context_manager

    def create_session(self, user_id: Optional[str] = None) -> str:
        """새 대화 세션 생성"""
        return self.context_manager.create_session(user_id)

    def add_conversation(
        self,
        session_id: str,
        user_request: str,
        assistant_response: str,
        generated_code: Optional[str] = None,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """대화 기록 추가"""
        self.context_manager.add_conversation_turn(
            session_id=session_id,
            user_request=user_request,
            assistant_response=assistant_response,
            generated_code=generated_code,
            filename=filename,
            metadata=metadata or {}
        )

    def get_context_for_llm(self, session_id: str, include_code: bool = True) -> str:
        """LLM용 컨텍스트 정보 조회"""
        return self.context_manager.get_context_for_llm(session_id, include_code)

    def get_session_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 히스토리 조회"""
        return self.context_manager.get_session_history(session_id)

    def get_all_sessions(self, user_id: Optional[str] = None) -> list:
        """세션 목록 조회"""
        return self.context_manager.get_all_sessions(user_id)

    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        return self.context_manager.delete_session(session_id)

    def is_code_modification_request(self, description: str) -> bool:
        """코드 수정 요청인지 판단"""
        modification_keywords = [
            "수정", "변경", "바꿔", "제거", "삭제", "추가해줘", "고쳐",
            "주석 제거", "주석 추가", "리팩토링", "최적화",
            "이 코드를", "방금 만든", "너가 만든", "기존 코드",
            "이 앱에", "이 프로그램에", "위 코드에"
        ]

        description_lower = description.lower()
        return any(keyword in description_lower for keyword in modification_keywords)


# 싱글톤 인스턴스
context_service = ContextManagementService()