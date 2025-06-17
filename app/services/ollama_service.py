"""
Ollama 서비스 - LLM 모델과의 통신을 담당
"""
import os
import sys
import aiohttp
import logging
from typing import List, Optional, Dict, Any
from langchain_community.llms import Ollama

from .context_manager import context_manager

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ..config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Ollama LLM 서비스 클래스"""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.default_model = settings.default_model
        self.backup_model = settings.backup_model
        self.llm = None

    async def initialize(self):
        """서비스 초기화"""
        try:
            # LangChain Ollama 객체 생성
            self.llm = Ollama(
                model=self.default_model,
                base_url=self.base_url
            )

            # 연결 테스트
            await self.test_connection()
            logger.info(f"✅ Ollama 서비스 초기화 완료 - 모델: {self.default_model}")

        except Exception as e:
            logger.error(f"❌ Ollama 서비스 초기화 실패: {e}")
            # 백업 모델로 재시도
            try:
                self.llm = Ollama(
                    model=self.backup_model,
                    base_url=self.base_url
                )
                logger.info(f"✅ 백업 모델로 초기화 완료 - 모델: {self.backup_model}")
            except Exception as backup_error:
                logger.error(f"❌ 백업 모델도 실패: {backup_error}")
                raise

    async def test_connection(self) -> bool:
        """Ollama 서버 연결 테스트"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        return True
                    return False
        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            return False

    async def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 조회"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['name'] for model in data.get('models', [])]
                    return []
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            return []

    async def generate_code(self, prompt: str, **kwargs) -> str:
        """기본 코드 생성 요청"""
        if not self.llm:
            await self.initialize()

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"코드 생성 실패: {e}")
            raise

    async def generate_code_with_context(self, description: str, language: str = "python",
                                         framework: Optional[str] = None,
                                         session_id: Optional[str] = None) -> str:
        """컨텍스트를 활용한 스마트 코드 생성"""
        if not self.llm:
            await self.initialize()

        # 컨텍스트 가져오기
        context_info = ""
        if session_id:
            context_info = context_manager.get_context_for_llm(session_id)

        # 요청 유형 분석
        is_modification_request = self._is_code_modification_request(description)

        if context_info and is_modification_request:
            # 기존 코드 수정 요청
            enhanced_prompt = f"""
이전 대화 컨텍스트:
{context_info}

현재 요청: {description}

위의 컨텍스트에서 가장 최근에 생성된 코드를 기반으로 다음 작업을 수행해주세요:
- 요청사항: {description}
- 기존 코드의 구조와 기능은 유지하면서 요청된 수정사항만 적용
- 완전한 수정된 코드를 제공 (부분 코드가 아닌 전체 코드)

수정된 완전한 코드:
"""
        elif context_info:
            # 기존 프로젝트에 새 기능 추가
            base_template = self._get_template_by_language(description, language, framework)
            enhanced_prompt = f"""
이전 대화 컨텍스트:
{context_info}

현재 요청: {base_template}

위의 컨텍스트를 참고하여 다음 조건을 만족해주세요:
1. 기존 프로젝트와 일관성 유지 (같은 언어, 스타일, 패턴)
2. 이미 사용 중인 라이브러리 활용
3. 기존 파일들과 호환되는 구조
4. 점진적이고 발전적인 코드 생성

답변:
"""
        else:
            # 새로운 프로젝트 시작
            base_template = self._get_template_by_language(description, language, framework)
            enhanced_prompt = base_template

        try:
            response = self.llm.invoke(enhanced_prompt)
            return response
        except Exception as e:
            logger.error(f"컨텍스트 기반 코드 생성 실패: {e}")
            raise

    def _is_code_modification_request(self, description: str) -> bool:
        """코드 수정 요청인지 판단"""
        modification_keywords = [
            "수정", "변경", "바꿔", "제거", "삭제", "추가해줘", "고쳐",
            "주석 제거", "주석 추가", "리팩토링", "최적화",
            "이 코드를", "방금 만든", "너가 만든", "기존 코드",
            "이 앱에", "이 프로그램에", "위 코드에"
        ]

        description_lower = description.lower()
        return any(keyword in description_lower for keyword in modification_keywords)

    def _get_template_by_language(self, description: str, language: str, framework: Optional[str]) -> str:
        """언어별 템플릿 선택"""
        templates = {
            "python": self._get_python_template(description, framework),
            "javascript": self._get_javascript_template(description, framework),
            "java": self._get_java_template(description, framework)
        }

        return templates.get(language, templates["python"])

    async def generate_code_with_template(self, description: str, language: str = "python",
                                          framework: Optional[str] = None) -> str:
        """템플릿을 사용한 코드 생성"""
        template = self._get_template_by_language(description, language, framework)
        return await self.generate_code(template)

    def _get_python_template(self, description: str, framework: Optional[str]) -> str:
        """Python 코드 생성 템플릿"""
        framework_info = ""
        if framework:
            framework_info = f"프레임워크는 {framework}을 사용해주세요."

        return f"""
                다음 요구사항에 맞는 완전한 Python 코드를 작성해주세요:

                요구사항: {description}
                {framework_info}
    
                다음 조건을 만족해야 합니다:
                1. 모든 필요한 import 문 포함
                2. 실행 가능한 완전한 코드
                3. 에러 처리 포함 (try-except)
                4. 상세한 주석으로 코드 설명
                5. 메인 실행 부분 포함 (if __name__ == "__main__":)
                6. 사용자 친화적인 출력 메시지

                코드만 출력하고 다른 설명은 최소화해주세요.
                """

    def _get_javascript_template(self, description: str, framework: Optional[str]) -> str:
        """JavaScript 코드 생성 템플릿"""
        framework_info = ""
        if framework:
            framework_info = f"프레임워크는 {framework}을 사용해주세요."

        return f"""
            다음 요구사항에 맞는 완전한 JavaScript 코드를 작성해주세요:

            요구사항: {description}
            {framework_info}

            다음 조건을 만족해야 합니다:
            1. 모든 필요한 import/require 문 포함
            2. 실행 가능한 완전한 코드
            3. 에러 처리 포함 (try-catch)
            4. 상세한 주석으로 코드 설명
            5. 사용자 친화적인 출력

            코드만 출력하고 다른 설명은 최소화해주세요.
            """

    def _get_java_template(self, description: str, framework: Optional[str]) -> str:
        """Java 코드 생성 템플릿"""
        framework_info = ""
        if framework:
            framework_info = f"프레임워크는 {framework}을 사용해주세요."

        return f"""
            다음 요구사항에 맞는 완전한 Java 코드를 작성해주세요:

            요구사항: {description}
            {framework_info}

            다음 조건을 만족해야 합니다:
            1. 완전한 클래스 구조
            2. 모든 필요한 import 문 포함
            3. 실행 가능한 main 메소드
            4. 예외 처리 포함
            5. 상세한 주석으로 코드 설명

            코드만 출력하고 다른 설명은 최소화해주세요.
            """


# 싱글톤 인스턴스
ollama_service = OllamaService()