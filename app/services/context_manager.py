"""
컨텍스트 관리 서비스 - 대화 기록과 생성된 코드를 관리
"""
import os
import sys
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ..config import settings


@dataclass
class ConversationTurn:
    """대화 턴 데이터 클래스"""
    turn_id: str
    user_request: str
    assistant_response: str
    generated_code: Optional[str] = None
    filename: Optional[str] = None
    timestamp: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CodeContext:
    """코드 컨텍스트 정보"""
    language: str
    framework: Optional[str]
    project_type: str
    current_files: List[str]
    dependencies: List[str]
    main_functionality: str


@dataclass
class ConversationSession:
    """대화 세션"""
    session_id: str
    user_id: Optional[str]
    created_at: str
    last_activity: str
    turns: List[ConversationTurn]
    code_context: Optional[CodeContext]
    session_summary: str = ""

    def __post_init__(self):
        if not self.turns:
            self.turns = []


class ContextManager:
    """컨텍스트 관리자"""

    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.session_timeout = timedelta(hours=24)  # 24시간 후 세션 만료
        self.max_context_turns = 10  # 최대 유지할 대화 턴 수
        self.context_storage_path = os.path.join(settings.generated_code_path, "contexts")
        self._ensure_storage_directory()

    def _ensure_storage_directory(self):
        """컨텍스트 저장 디렉토리 생성"""
        os.makedirs(self.context_storage_path, exist_ok=True)

    def create_session(self, user_id: Optional[str] = None) -> str:
        """새 대화 세션 생성"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            turns=[],
            code_context=None
        )

        self.sessions[session_id] = session
        self._save_session(session)

        return session_id

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """세션 조회"""
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # 세션 만료 확인
            last_activity = datetime.fromisoformat(session.last_activity)
            if datetime.now() - last_activity > self.session_timeout:
                self.delete_session(session_id)
                return None

            return session

        # 저장된 세션에서 로드 시도
        return self._load_session(session_id)

    def add_conversation_turn(self, session_id: str, user_request: str,
                              assistant_response: str, generated_code: Optional[str] = None,
                              filename: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """대화 턴 추가"""
        session = self.get_session(session_id)
        if not session:
            return False

        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_request=user_request,
            assistant_response=assistant_response,
            generated_code=generated_code,
            filename=filename,
            metadata=metadata or {}
        )

        session.turns.append(turn)
        session.last_activity = datetime.now().isoformat()

        # 컨텍스트 길이 제한
        if len(session.turns) > self.max_context_turns:
            # 오래된 턴 제거 (첫 번째는 유지 - 초기 컨텍스트)
            session.turns = [session.turns[0]] + session.turns[-(self.max_context_turns - 1):]

        # 코드 컨텍스트 업데이트
        self._update_code_context(session, turn)

        # 세션 저장
        self._save_session(session)

        return True

    def _update_code_context(self, session: ConversationSession, turn: ConversationTurn):
        """코드 컨텍스트 업데이트"""
        if not turn.generated_code:
            return

        # 언어와 프레임워크 추출
        language = self._detect_language(turn.generated_code)
        framework = self._detect_framework(turn.generated_code)
        dependencies = self._extract_dependencies(turn.generated_code)

        if not session.code_context:
            session.code_context = CodeContext(
                language=language,
                framework=framework,
                project_type="unknown",
                current_files=[],
                dependencies=dependencies,
                main_functionality=turn.user_request[:100]
            )
        else:
            # 기존 컨텍스트 업데이트
            session.code_context.dependencies.extend(dependencies)
            session.code_context.dependencies = list(set(session.code_context.dependencies))

        # 생성된 파일 추가
        if turn.filename:
            if turn.filename not in session.code_context.current_files:
                session.code_context.current_files.append(turn.filename)

    def _detect_language(self, code: str) -> str:
        """코드에서 언어 감지"""
        code_lower = code.lower()

        if any(keyword in code_lower for keyword in ['def ', 'import ', 'from ', 'python']):
            return "python"
        elif any(keyword in code_lower for keyword in ['function ', 'const ', 'let ', 'var ']):
            return "javascript"
        elif any(keyword in code_lower for keyword in ['public class', 'public static', 'java']):
            return "java"
        else:
            return "unknown"

    def _detect_framework(self, code: str) -> Optional[str]:
        """코드에서 프레임워크 감지"""
        code_lower = code.lower()

        if 'fastapi' in code_lower or 'from fastapi' in code_lower:
            return "fastapi"
        elif 'flask' in code_lower or 'from flask' in code_lower:
            return "flask"
        elif 'django' in code_lower:
            return "django"
        elif 'react' in code_lower or 'jsx' in code_lower:
            return "react"
        else:
            return None

    def _extract_dependencies(self, code: str) -> List[str]:
        """코드에서 의존성 추출"""
        dependencies = []
        lines = code.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Python imports
                if line.startswith('import '):
                    dep = line.replace('import ', '').split()[0].split('.')[0]
                else:  # from ... import
                    dep = line.split('from ')[1].split(' import')[0].split('.')[0]

                # 표준 라이브러리 제외
                if dep not in ['os', 'sys', 'json', 'time', 'datetime', 're', 'random', 'typing']:
                    dependencies.append(dep)

        return dependencies

    def get_context_for_llm(self, session_id: str, include_code: bool = True) -> str:
        """LLM에 제공할 컨텍스트 문자열 생성"""
        session = self.get_session(session_id)
        if not session:
            return ""

        context_parts = []

        # 세션 요약
        if session.session_summary:
            context_parts.append(f"세션 요약: {session.session_summary}")

        # 코드 컨텍스트
        if session.code_context:
            ctx = session.code_context
            context_parts.append(f"현재 프로젝트 정보:")
            context_parts.append(f"- 언어: {ctx.language}")
            if ctx.framework:
                context_parts.append(f"- 프레임워크: {ctx.framework}")
            if ctx.dependencies:
                context_parts.append(f"- 사용 중인 라이브러리: {', '.join(ctx.dependencies)}")
            if ctx.current_files:
                context_parts.append(f"- 생성된 파일들: {', '.join(ctx.current_files)}")
            context_parts.append(f"- 주요 기능: {ctx.main_functionality}")

        # 최근 대화 기록
        context_parts.append("\n이전 대화 기록:")

        # 최근 3-5턴만 포함 (토큰 수 제한)
        recent_turns = session.turns[-5:] if len(session.turns) > 5 else session.turns

        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"\n{i}. 사용자: {turn.user_request}")
            context_parts.append(
                f"   응답: {turn.assistant_response[:200]}{'...' if len(turn.assistant_response) > 200 else ''}")

            if include_code and turn.generated_code and turn.filename:
                # 코드는 요약만 포함
                code_summary = self._summarize_code(turn.generated_code)
                context_parts.append(f"   생성된 파일: {turn.filename} ({code_summary})")

        return "\n".join(context_parts)

    def _summarize_code(self, code: str) -> str:
        """코드 요약"""
        lines = code.split('\n')

        # 주요 함수/클래스 추출
        functions = []
        classes = []

        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                func_name = line.split('def ')[1].split('(')[0]
                functions.append(func_name)
            elif line.startswith('class '):
                class_name = line.split('class ')[1].split('(')[0].split(':')[0]
                classes.append(class_name)

        summary_parts = []
        if classes:
            summary_parts.append(f"클래스: {', '.join(classes[:3])}")
        if functions:
            summary_parts.append(f"함수: {', '.join(functions[:3])}")

        return "; ".join(summary_parts) if summary_parts else f"{len(lines)}줄의 코드"

    def get_session_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 히스토리 조회"""
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "total_turns": len(session.turns),
            "code_context": asdict(session.code_context) if session.code_context else None,
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "user_request": turn.user_request,
                    "timestamp": turn.timestamp,
                    "has_code": bool(turn.generated_code),
                    "filename": turn.filename
                }
                for turn in session.turns
            ]
        }

    def update_session_summary(self, session_id: str, summary: str):
        """세션 요약 업데이트"""
        session = self.get_session(session_id)
        if session:
            session.session_summary = summary
            self._save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]

        # 저장된 파일도 삭제
        session_file = os.path.join(self.context_storage_path, f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
            return True

        return False

    def _save_session(self, session: ConversationSession):
        """세션을 파일로 저장"""
        session_file = os.path.join(self.context_storage_path, f"{session.session_id}.json")

        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"세션 저장 실패: {e}")

    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        """파일에서 세션 로드"""
        session_file = os.path.join(self.context_storage_path, f"{session_id}.json")

        if not os.path.exists(session_file):
            return None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # ConversationSession 객체로 변환
            session = ConversationSession(**data)

            # ConversationTurn 객체들로 변환
            session.turns = [ConversationTurn(**turn_data) for turn_data in data['turns']]

            # CodeContext 객체로 변환
            if data.get('code_context'):
                session.code_context = CodeContext(**data['code_context'])

            self.sessions[session_id] = session
            return session

        except Exception as e:
            print(f"세션 로드 실패: {e}")
            return None

    def cleanup_expired_sessions(self):
        """만료된 세션들 정리"""
        current_time = datetime.now()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            last_activity = datetime.fromisoformat(session.last_activity)
            if current_time - last_activity > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.delete_session(session_id)

        return len(expired_sessions)

    def get_all_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """모든 세션 목록 조회"""
        sessions_info = []

        for session in self.sessions.values():
            if user_id and session.user_id != user_id:
                continue

            sessions_info.append({
                "session_id": session.session_id,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "turn_count": len(session.turns),
                "has_code_context": bool(session.code_context)
            })

        return sorted(sessions_info, key=lambda x: x['last_activity'], reverse=True)


# 싱글톤 인스턴스
context_manager = ContextManager()