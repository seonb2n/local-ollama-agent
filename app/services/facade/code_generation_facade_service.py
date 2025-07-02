"""
코드 생성 퍼사드 서비스 - 전체 코드 생성 프로세스 조율
"""
import logging
from typing import Optional
from ..ollama_service import ollama_service
from ..context_management_service import context_service
from ..improvement_service import improvement_service
from ..web_search_service import web_search_service
from ...repository.RagIntegration import RAGIntegration

logger = logging.getLogger(__name__)


class CodeGenerationFacade:
    """코드 생성 전체 프로세스를 조율하는 퍼사드 서비스"""

    def __init__(self):
        self.ollama_service = ollama_service
        self.context_service = context_service
        self.improvement_service = improvement_service
        self.web_search_service = web_search_service

        self.rag_integration = None
        self.enable_rag = True
        self.enable_self_improvement = True

    async def initialize(self):
        """퍼사드 서비스 초기화"""
        await self.ollama_service.initialize()

        if self.enable_rag:
            self.rag_integration = RAGIntegration()
            await self.rag_integration.initialize(self.ollama_service.default_model)
            logger.info("✅ RAG 시스템 초기화 완료")

    async def generate_code_with_context(
            self,
            description: str,
            language: str = "python",
            framework: Optional[str] = None,
            session_id: Optional[str] = None,
            enable_improvement: Optional[bool] = None,
            existing_file_path: Optional[str] = None
    ) -> str:
        """컨텍스트를 활용한 스마트 코드 생성"""

        # 세션이 없으면 새로 생성
        if not session_id:
            session_id = context_service.create_session()

        if not self.ollama_service.llm:
            await self.initialize()

        # Self-improvement 사용 여부 결정
        use_improvement = enable_improvement if enable_improvement is not None else self.enable_self_improvement

        # 1. 컨텍스트 정보 수집
        context_info = ""
        if session_id:
            context_info = self.context_service.get_context_for_llm(session_id)

        # 2. 기존 파일 내용 읽기
        existing_code = ""
        if existing_file_path:
            try:
                with open(existing_file_path, 'r', encoding='utf-8') as f:
                    existing_code = f.read()
                logger.info(f"📖 기존 파일 읽기 완료: {existing_file_path}")
            except Exception as e:
                logger.warning(f"기존 파일 읽기 실패: {e}")

        # 3. 외부 정보 수집 (RAG + Web Search)
        external_info = await self._gather_external_information(description, language)

        # 4. 프롬프트 생성
        enhanced_prompt = self._build_context_aware_prompt(
            description, language, framework, context_info, external_info, existing_code
        )

        try:
            # 4. 초기 코드 생성
            logger.info(f"🚀 코드 생성 시작 - 언어: {language}, 개선모드: {use_improvement}")
            initial_response = await self.ollama_service.generate_response(enhanced_prompt)

            if not use_improvement:
                return initial_response

            # 5. Self-improvement 수행
            final_response = await self.improvement_service.perform_improvement_cycle(
                initial_response, description, language, framework, session_id
            )

            return final_response

        except Exception as e:
            logger.error(f"컨텍스트 기반 코드 생성 실패: {e}")
            raise

    async def _gather_external_information(self, description: str, language: str) -> str:
        """외부 정보 수집 (RAG + Web Search)"""
        external_info = ""

        # RAG 검색
        if self.enable_rag and self.rag_integration:
            optimized_keyword = self.web_search_service.get_optimized_query(description, language)
            should_use_rag = await self.rag_integration.should_use_rag(description)

            if should_use_rag:
                logger.info("🔍 RAG 시스템을 통한 지식 검색 중...")
                try:
                    search_results = await self.rag_integration.search_knowledge(optimized_keyword)
                    if search_results:
                        external_info += f"""
**관련 기술 문서 (RAG 검색 결과):**
{search_results}
"""
                        logger.info("✅ RAG 검색 완료")
                except Exception as e:
                    logger.error(f"RAG 검색 실패: {e}")

        # 웹 검색 (RAG 결과가 없거나 부족한 경우)
        if not external_info:
            web_search_needed = await self.web_search_service.should_perform_web_search(
                description, language, None
            )
            if web_search_needed:
                logger.info("🔍 웹 검색 수행 중...")
                optimized_keyword = self.web_search_service.get_optimized_query(description, language)
                web_search_info = await self.web_search_service.perform_web_search(optimized_keyword)
                external_info += web_search_info
                logger.info("✅ 웹 검색 완료")

        return external_info

    def _build_context_aware_prompt(
            self,
            description: str,
            language: str,
            framework: Optional[str],
            context_info: str,
            external_info: str = "",
            existing_code: str = ""
    ) -> str:
        """컨텍스트를 고려한 프롬프트 구성"""

        is_modification_request = self.context_service.is_code_modification_request(description) or bool(existing_code)

        # 외부 정보 섹션
        external_section = ""
        if external_info:
            external_section = f"""
**최신 정보 및 참고 자료:**
{external_info}

위의 정보를 참고하여 현재 상황에 맞는 최적의 코드를 작성해주세요.
"""

        if existing_code or (context_info and is_modification_request):
            # 기존 코드 수정 요청
            code_section = f"\n\n**현재 파일 내용:**\n```{language}\n{existing_code}\n```" if existing_code else ""
            return f"""
이전 대화 컨텍스트:
{context_info}

{external_section}{code_section}

현재 요청: {description}

위의 기존 코드를 기반으로 다음 작업을 수행해주세요:
- 요청사항: {description}
- 기존 코드의 구조와 기능은 유지하면서 요청된 수정사항만 적용
- 완전한 수정된 코드를 제공 (부분 코드가 아닌 전체 코드)
- 기존 스타일과 패턴 일관성 유지

수정된 완전한 코드:
"""
        elif context_info:
            # 기존 프로젝트에 새 기능 추가
            base_template = self._get_template_by_language(description, language, framework)
            return f"""
이전 대화 컨텍스트:
{context_info}

{external_section}

현재 요청: {base_template}

위의 컨텍스트를 참고하여 다음 조건을 만족해주세요:
1. 기존 프로젝트와 일관성 유지 (같은 언어, 스타일, 패턴)
2. 이미 사용 중인 라이브러리 활용
3. 기존 파일들과 호환되는 구조
4. 점진적이고 발전적인 코드 생성
5. 기존 아키텍처 패턴 준수

답변:
"""
        else:
            # 새로운 프로젝트 시작
            template = self._get_template_by_language(description, language, framework)
            if external_section:
                return f"{external_section}\n\n{template}"
            return template

    def _get_template_by_language(self, description: str, language: str, framework: Optional[str]) -> str:
        """언어별 템플릿 선택"""
        templates = {
            "python": self._get_python_template(description, framework),
            "javascript": self._get_javascript_template(description, framework),
            "java": self._get_java_template(description, framework)
        }
        return templates.get(language, templates["python"])

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

    # 편의 메서드들
    def set_improvement_enabled(self, enabled: bool):
        """Self-improvement 기능 활성화/비활성화"""
        self.enable_self_improvement = enabled
        self.improvement_service.set_improvement_enabled(enabled)

    def set_quality_threshold(self, threshold: float):
        """품질 임계값 설정"""
        self.improvement_service.set_quality_threshold(threshold)

    def set_max_iterations(self, max_iter: int):
        """최대 반복 횟수 설정"""
        self.improvement_service.set_max_iterations(max_iter)


# 싱글톤 인스턴스
code_generation_facade = CodeGenerationFacade()