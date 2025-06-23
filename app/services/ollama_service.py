"""
Ollama 서비스 - LLM 모델과의 통신을 담당
"""
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import aiohttp
import logging
from typing import List, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_community.llms import Ollama
import json
from .context_manager import context_manager
from .dto.self_improvements import ImprovementIteration, ReflectionResult
from ..repository.RagIntegration import RAGIntegration

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
        self.max_iterations = 3  # 최대 개선 횟수
        self.min_acceptable_score = 7.5  # 최소 허용 점수
        self.improvement_history: Dict[str, List[ImprovementIteration]] = {}
        self.enable_self_improvement = True
        self.enable_web_search = True
        self.web_search_threshold = 0.7  # 웹 검색 필요도 임계값
        self.max_search_results = 3

        load_dotenv('.env.local')

        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

        self.rag_integration = None
        self.enable_rag = True  # RAG 기능 활성화 여부

    async def initialize(self):
        """서비스 초기화"""
        try:
            # LangChain Ollama 객체 생성
            self.llm = Ollama(
                model=self.default_model,
                base_url=self.base_url
            )

            if self.enable_rag:
                self.rag_integration = RAGIntegration()
                await self.rag_integration.initialize(self.default_model)
                logger.info("✅ RAG 시스템 초기화 완료")

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

    async def generate_code_with_context(
            self,
            description: str,
            language: str = "python",
            framework: Optional[str] = None,
            session_id: Optional[str] = None,
            enable_improvement: Optional[bool] = None
    ) -> str:
        """컨텍스트를 활용한 스마트 코드 생성 (Self-improvement 통합)"""

        if not self.llm:
            await self.initialize()

        # Self-improvement 사용 여부 결정
        use_improvement = enable_improvement if enable_improvement is not None else self.enable_self_improvement

        # 세션별 개선 히스토리 초기화
        if session_id and session_id not in self.improvement_history:
            self.improvement_history[session_id] = []

        # 1. 컨텍스트 정보 수집
        context_info = ""
        if session_id:
            context_info = context_manager.get_context_for_llm(session_id)

        # 1-1. RAG 검색 수행 (웹 검색보다 우선)
        rag_info = ""
        optimized_keyword = self._get_optimized_query(description, language)
        if self.enable_rag and self.rag_integration:
            should_use_rag = await self.rag_integration.should_use_rag(description)
            if should_use_rag:
                logger.info("🔍 RAG 시스템을 통한 지식 검색 중...")
                try:
                    # RAG로 관련 문서 검색
                    search_results = await self.rag_integration.search_knowledge(optimized_keyword)
                    if search_results:
                        rag_info = f"""
        **관련 기술 문서 (RAG 검색 결과):**
        {search_results}
        """
                        logger.info("✅ RAG 검색 완료")
                except Exception as e:
                    logger.error(f"RAG 검색 실패: {e}")

            # 1-2. 웹 검색 수행 (RAG 결과가 없거나 부족한 경우)
        web_search_info = ""
        if not rag_info and await self._should_perform_web_search(description, language, framework):
            logger.info("🔍 웹 검색 수행 중...")
            web_search_info = await self._perform_web_search(optimized_keyword)
            logger.info("✅ 웹 검색 완료")

        # 2. 요청 유형 분석 및 프롬프트 생성
        enhanced_prompt = self._build_context_aware_prompt(
            description, language, framework, context_info, rag_info + web_search_info
        )

        try:
            # 3. 초기 코드 생성
            logger.info(f"🚀 코드 생성 시작 - 언어: {language}, 개선모드: {use_improvement}")
            initial_response = self.llm.invoke(enhanced_prompt)

            if not use_improvement:
                # Self-improvement 비활성화 시 바로 반환
                return initial_response

            # 4. Self-improvement 수행
            final_response, _ = await self._perform_self_improvement_cycle(
                initial_response, description, language, framework, session_id
            )

            return final_response

        except Exception as e:
            logger.error(f"컨텍스트 기반 코드 생성 실패: {e}")
            raise

    def _build_context_aware_prompt(
            self,
            description: str,
            language: str,
            framework: Optional[str],
            context_info: str,
            web_search_info: str = ""
    ) -> str:
        """컨텍스트를 고려한 프롬프트 구성"""

        is_modification_request = self._is_code_modification_request(description)

        # 웹 검색 정보 추가
        web_search_section = ""
        if web_search_info:
            web_search_section = f"""
        **최신 정보 (웹 검색 결과):**
        {web_search_info}

        위의 최신 정보를 참고하여 현재 상황에 맞는 최적의 코드를 작성해주세요.
        """

        if context_info and is_modification_request:
            # 기존 코드 수정 요청
            return f"""
이전 대화 컨텍스트:
{context_info}

웹 검색 항목을 통한 최신 정보:
{web_search_section}

현재 요청: {description}

위의 컨텍스트에서 가장 최근에 생성된 코드를 기반으로 다음 작업을 수행해주세요:
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
            return self._get_template_by_language(description, language, framework)

    async def _perform_self_improvement_cycle(
            self,
            initial_response: str,
            description: str,
            language: str,
            framework: Optional[str],
            session_id: Optional[str]
    ) -> Tuple[str, List[ImprovementIteration]]:
        """Self-improvement 사이클 수행"""

        current_response = initial_response
        iterations = []

        for iteration in range(self.max_iterations):
            logger.info(f"🔄 Self-improvement 반복 {iteration + 1}/{self.max_iterations}")

            # Self-reflection 수행
            reflection_result = await self.perform_self_reflection(
                current_response, description, language, framework, session_id
            )

            # 점수가 충분히 높으면 종료
            if reflection_result.score >= self.min_acceptable_score:
                logger.info(f"✅ 만족스러운 품질 달성 (점수: {reflection_result.score})")
                break
            else:
                logger.info(f"✅ 현재 품질 (점수: {reflection_result.score})")

            # 개선된 응답 생성
            improved_response = await self._generate_improved_response(
                current_response, reflection_result, description, language, framework, session_id
            )

            # 반복 기록 저장
            iteration_record = ImprovementIteration(
                iteration=iteration + 1,
                original_response=current_response,
                reflection_result=reflection_result,
                improved_response=improved_response,
                improvement_reason=f"점수 {reflection_result.score:.1f} - {', '.join(reflection_result.issues[:2])}",
                timestamp=time.time()
            )
            iterations.append(iteration_record)

            # 다음 반복을 위해 현재 응답 업데이트
            current_response = improved_response

            logger.info(f"📈 반복 {iteration + 1} 완료 - 점수: {reflection_result.score:.1f}")

        # 세션 히스토리에 추가
        if session_id:
            self.improvement_history[session_id].extend(iterations)

        return current_response, iterations

    async def perform_self_reflection(
            self,
            response: str,
            original_request: str,
            language: str,
            framework: Optional[str],
            session_id: Optional[str] = None
    ) -> ReflectionResult:
        """Self-reflection을 통한 응답 평가 (컨텍스트 고려)"""

        # 컨텍스트 정보 수집
        context_guidance = ""
        if session_id:
            context_info = context_manager.get_context_for_llm(session_id)
            if context_info:
                context_guidance = f"""
                            **프로젝트 컨텍스트:**
                            {context_info}
                            **컨텍스트 일관성 평가 포함:**
                            - 기존 코드 스타일과의 일치성
                            - 사용된 라이브러리의 일관성
                            - 아키텍처 패턴의 연속성
                            - 네이밍 컨벤션의 일치성
                            """

        reflection_prompt = f"""
                        다음 코드를 종합적으로 평가해주세요:
                        **원본 요청:** {original_request}
                        **언어:** {language}
                        **프레임워크:** {framework or "없음"}
                        {context_guidance}
                        **생성된 코드:**
                        ```{language}
                        {response}
                        ```
                        다음 기준으로 평가하고 JSON 형식으로 답변해주세요:
                        {{
                            "score": 점수 (0-10, 소수점 1자리),
                            "code_quality": {{
                                "score": 점수 (0-10),
                                "issues": ["문제점1", "문제점2"],
                                "good_points": ["장점1", "장점2"]
                            }},
                            "completeness": {{
                                "score": 점수 (0-10),
                                "missing_features": ["누락된 기능1", "누락된 기능2"],
                                "satisfied_requirements": ["만족된 요구사항1", "만족된 요구사항2"]
                            }},
                            "context_consistency": {{
                                "score": 점수 (0-10),
                                "inconsistencies": ["불일치 사항1", "불일치 사항2"],
                                "good_alignment": ["잘 맞는 부분1", "잘 맞는 부분2"]
                            }},
                            "best_practices": {{
                                "score": 점수 (0-10),
                                "violations": ["위반사항1", "위반사항2"],
                                "good_practices": ["좋은 관례1", "좋은 관례2"]
                            }},
                            "error_handling": {{
                                "score": 점수 (0-10),
                                "missing_error_handling": ["누락된 에러 처리1", "누락된 에러 처리2"],
                                "good_error_handling": ["좋은 에러 처리1", "좋은 에러 처리2"]
                            }},
                            "overall_issues": ["전체적인 문제점1", "전체적인 문제점2", "전체적인 문제점3"],
                            "improvement_suggestions": ["개선 제안1", "개선 제안2", "개선 제안3"],
                            "overall_assessment": "전체적인 평가 및 요약"
                        }}
                        평가 기준:
                        - 9-10: 뛰어남 (production-ready, 컨텍스트 완벽 일치)
                        - 7-8: 좋음 (minor improvements needed)
                        - 5-6: 보통 (moderate improvements needed)
                        - 3-4: 나쁨 (major improvements needed)
                        - 0-2: 매우 나쁨 (complete rewrite needed)
                        """

        try:
            reflection_response = self.llm.invoke(reflection_prompt)

            try:
                reflection_data = json.loads(reflection_response)
            except json.JSONDecodeError:
                reflection_data = self._extract_reflection_from_text(reflection_response)

            issues = reflection_data.get("overall_issues", [])
            suggestions = reflection_data.get("improvement_suggestions", [])

            return ReflectionResult(
                score=float(reflection_data.get("score", 5.0)),
                issues=issues,
                suggestions=suggestions,
                overall_assessment=reflection_data.get("overall_assessment", "평가를 완료했습니다.")
            )

        except Exception as e:
            logger.error(f"Self-reflection 실패: {e}")
            return ReflectionResult(
                score=5.0,
                issues=["평가 중 오류 발생"],
                suggestions=["재평가 필요"],
                overall_assessment="평가를 완료하지 못했습니다."
            )

    async def _generate_improved_response(
            self,
            original_response: str,
            reflection_result: ReflectionResult,
            original_request: str,
            language: str,
            framework: Optional[str],
            session_id: Optional[str] = None
    ) -> str:
        """개선된 응답 생성 (컨텍스트 활용)"""

        # 컨텍스트 정보 수집
        context_guidance = ""
        if session_id:
            context_info = context_manager.get_context_for_llm(session_id)
            if context_info:
                context_guidance = f"""
**이전 대화 컨텍스트:**
{context_info}

**컨텍스트 활용 지침:**
- 기존 프로젝트와의 일관성 유지 (코딩 스타일, 네이밍 컨벤션, 아키텍처 패턴)
- 이미 사용된 라이브러리와 디펜던시 재활용
- 기존 코드와의 호환성 보장
- 프로젝트 전체 구조에 맞는 개선
- 기존 설정값이나 환경변수 활용
"""

        # 개선 히스토리에서 학습
        improvement_history_guidance = ""
        if session_id and session_id in self.improvement_history:
            recent_iterations = self.improvement_history[session_id][-3:]
            if recent_iterations:
                common_patterns = self._analyze_improvement_patterns(recent_iterations)
                improvement_history_guidance = f"""
**이전 개선 패턴 분석:**
{common_patterns}

**반복되는 이슈 방지:**
- 이전에 발견된 공통 문제점들을 미리 고려
- 성공적이었던 개선 방향성 참고
- 반복되는 실수 패턴 회피
"""

        improvement_prompt = f"""
다음 코드를 종합적으로 개선해주세요:

**원본 요청:** {original_request}
**언어:** {language}
**프레임워크:** {framework or "없음"}

{context_guidance}

{improvement_history_guidance}

**현재 코드:**
```{language}
{original_response}
```

**발견된 문제점들:**
{chr(10).join(f"- {issue}" for issue in reflection_result.issues)}

**개선 제안사항:**
{chr(10).join(f"- {suggestion}" for suggestion in reflection_result.suggestions)}

**현재 점수:** {reflection_result.score}/10
**목표 점수:** 8.0+ (production-ready 수준)

위의 모든 정보를 종합하여 개선된 완전한 코드를 작성해주세요.

개선 시 우선순위:
1. **컨텍스트 일관성**: 기존 프로젝트와의 연계성 및 호환성
2. **문제점 해결**: 발견된 모든 이슈 완전 해결
3. **품질 향상**: 코드 구조, 가독성, 유지보수성 개선
4. **에러 처리**: 견고한 예외 처리 및 에러 복구
5. **베스트 프랙티스**: 업계 표준 및 권장사항 적용
6. **완전성**: 실제 사용 가능한 수준의 구현

개선된 완전한 코드만 출력하고 추가 설명은 최소화해주세요.
        """

        try:
            improved_response = self.llm.invoke(improvement_prompt)
            return improved_response

        except Exception as e:
            logger.error(f"개선된 응답 생성 실패: {e}")
            return original_response

    def _analyze_improvement_patterns(self, recent_iterations: List[ImprovementIteration]) -> str:
        """최근 개선 반복에서 패턴 분석"""
        if not recent_iterations:
            return "개선 히스토리 없음"

        all_issues = []
        all_suggestions = []

        for iteration in recent_iterations:
            all_issues.extend(iteration.reflection_result.issues)
            all_suggestions.extend(iteration.reflection_result.suggestions)

        issue_counts = {}
        suggestion_counts = {}

        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_suggestions = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        pattern_analysis = "반복되는 주요 이슈:\n"
        for issue, count in top_issues:
            pattern_analysis += f"- {issue} (발생 {count}회)\n"

        pattern_analysis += "\n자주 제안되는 개선사항:\n"
        for suggestion, count in top_suggestions:
            pattern_analysis += f"- {suggestion} (제안 {count}회)\n"

        return pattern_analysis

    def _extract_reflection_from_text(self, text: str) -> Dict[str, Any]:
        """텍스트에서 reflection 정보 추출 (JSON 파싱 실패 시 백업)"""
        import re

        score_match = re.search(r'score["\s:]*(\d+\.?\d*)', text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 5.0

        return {
            "score": score,
            "overall_issues": ["텍스트 파싱으로 추출된 이슈"],
            "improvement_suggestions": ["상세 분석을 위해 JSON 형식 응답 필요"],
            "overall_assessment": "부분적 평가 완료"
        }

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

    async def _should_perform_web_search(self, description: str, language: str, framework: Optional[str]) -> bool:
        """웹 검색이 필요한지 판단"""
        if not self.enable_web_search:
            return False

        # 웹 검색이 필요한 키워드들
        web_search_keywords = [
            "최신", "latest", "newest", "current", "2024", "2025",
            "업데이트", "update", "버전", "version",
            "새로운", "new", "트렌드", "trend",
            "API", "라이브러리", "library", "패키지", "package",
            "프레임워크", "framework", "도구", "tool",
            "설치", "install", "setup", "configuration",
            "에러", "error", "문제", "issue", "해결", "solution"
        ]

        description_lower = description.lower()

        # 키워드 기반 1차 판단
        keyword_score = sum(1 for keyword in web_search_keywords if keyword in description_lower)

        if keyword_score >= 2:
            return True

        # LLM을 이용한 정교한 판단
        judgment_prompt = f"""
        다음 코드 생성 요청에 대해 웹 검색이 필요한지 판단해주세요:

        요청: {description}
        언어: {language}
        프레임워크: {framework or "없음"}

        웹 검색이 필요한 경우:
        - 최신 기술, 라이브러리, API 정보가 필요한 경우
        - 특정 에러나 문제 해결법이 필요한 경우
        - 설치/설정 방법이 필요한 경우
        - 업데이트된 문법이나 방법론이 필요한 경우

        0.0 (불필요) ~ 1.0 (매우 필요) 사이의 점수만 출력하세요.
        """

        try:
            judgment_response = self.llm.invoke(judgment_prompt)
            score = float(judgment_response.strip())
            return score >= self.web_search_threshold
        except:
            return keyword_score >= 1

    async def _perform_web_search(self, keywords: List[str]) -> str:
        """Google Custom Search API를 사용한 웹 검색"""
        try:

            if not self.google_api_key or not self.search_engine_id:
                logger.error("Google Search API 설정이 완료되지 않았습니다.")
                return "Google Search API 설정이 완료되지 않았습니다."

            import aiohttp
            import urllib.parse

            # 검색 쿼리 최적화
            search_query = " ".join(keywords)
            encoded_query = urllib.parse.quote(search_query)

            # Google Custom Search API URL
            search_url = (
                f"https://www.googleapis.com/customsearch/v1"
                f"?key={self.google_api_key}"
                f"&cx={self.search_engine_id}"
                f"&q={encoded_query}"
                f"&num={min(self.max_search_results, 10)}"
            )

            logger.info(f"검색 URL: {search_url[:100]}...")  # API 키 노출 방지를 위해 처음 100글자만

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    logger.info(f"응답 상태 코드: {response.status}")

                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"응답 데이터 키: {list(data.keys())}")

                        # 검색 결과 처리
                        results = []
                        items = data.get('items', [])
                        logger.info(f"검색된 아이템 수: {len(items)}")

                        if not items:
                            logger.warning("검색 결과가 비어있습니다.")
                            logger.info(f"전체 응답 데이터: {data}")
                            return "검색 결과를 찾을 수 없습니다."

                        for i, item in enumerate(items):
                            title = item.get('title', '')
                            snippet = item.get('snippet', '')
                            link = item.get('link', '')

                            logger.info(f"결과 {i + 1}: 제목='{title[:50]}...', 링크='{link}'")

                            # 결과 포맷팅
                            result_text = f"**{title}**\n{snippet}"
                            if link:
                                result_text += f"\n🔗 {link}"

                            results.append(result_text)

                        if results:
                            search_info = data.get('searchInformation', {})
                            total_results = search_info.get('totalResults', '0')
                            search_time = search_info.get('searchTime', '0')

                            logger.info(f"검색 완료 - 총 {len(results)}개 결과 반환")

                            final_result = f"""
    🔍 **웹 검색 결과** (총 {total_results}개 결과, {search_time}초)

    {chr(10).join(f"{i + 1}. {result}" for i, result in enumerate(results))}
    """
                            logger.info(f"최종 결과 길이: {len(final_result)} 문자")
                            return final_result
                        else:
                            logger.warning("결과 리스트가 비어있습니다.")
                            return "검색 결과를 찾을 수 없습니다."

                    elif response.status == 403:
                        error_data = await response.json()
                        logger.error(f"403 에러: {error_data}")
                        error_message = error_data.get('error', {}).get('message', '')
                        if 'quota' in error_message.lower():
                            return "Google Search API 할당량이 초과되었습니다."
                        else:
                            return f"Google Search API 접근 권한 오류: {error_message}"

                    else:
                        response_text = await response.text()
                        logger.error(f"예상치 못한 응답 상태: {response.status}, 내용: {response_text[:200]}")
                        return f"웹 검색 중 오류가 발생했습니다. 상태 코드: {response.status}"

        except Exception as e:
            logger.error(f"Google 웹 검색 실패: {e}", exc_info=True)
            return f"웹 검색을 수행할 수 없습니다: {str(e)}"

    def _get_optimized_query(self, description: str, language: str) -> List[str]:
        """LLM을 이용해 description에서 검색 키워드 추출"""
        try:
            description_optimized_query = f"""
    다음 코드 생성 요청에서 웹 검색에 필요한 핵심 키워드만 추출해주세요.

    요청: {description}
    프로그래밍 언어: {language}

    규칙:
    1. 기술적 용어와 핵심 개념만 포함
    2. 불필요한 조사나 부사 제거
    3. 영어 기술 용어 우선 사용
    4. 최대 5개의 키워드만 선택
    5. 각 키워드는 1-3단어로 구성
    6. 반드시 JSON 배열 형태로만 응답: ["키워드1", "키워드2", "키워드3"]

    예시:
    - 요청: "Spring Boot에서 JWT 토큰 인증을 구현하는 방법을 알려주세요"
    - 응답: ["Spring Boot", "JWT", "authentication", "token", "security"]

    응답 형식: ["키워드1", "키워드2", ...]
    """

            # LLM 호출
            response = self.llm.invoke(description_optimized_query)
            logger.info(f"LLM 키워드 추출 응답: {response}")

            # 응답에서 JSON 배열 추출
            keywords = self._parse_keywords_from_response(response)

            # 기본 키워드 추가 (언어명)
            if language and language not in keywords:
                keywords.insert(0, language)

            logger.info(f"추출된 키워드: {keywords}")
            return keywords[:5]  # 최대 5개로 제한

        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            # 실패시 기본 키워드 반환
            return self._get_fallback_keywords(description, language)

    def _parse_keywords_from_response(self, response: str) -> List[str]:
        """LLM 응답에서 키워드 배열 파싱"""
        import json
        import re

        try:
            # 1차 시도: 직접 JSON 파싱
            if response.strip().startswith('[') and response.strip().endswith(']'):
                return json.loads(response.strip())

            # 2차 시도: JSON 배열 패턴 찾기
            json_pattern = r'\[([^\]]+)\]'
            matches = re.findall(json_pattern, response)

            if matches:
                # 가장 긴 매치를 선택 (가장 완전한 배열일 가능성)
                json_str = f"[{max(matches, key=len)}]"
                return json.loads(json_str)

            # 3차 시도: 따옴표로 둘러싸인 단어들 추출
            quoted_pattern = r'"([^"]+)"'
            keywords = re.findall(quoted_pattern, response)

            if keywords:
                return keywords

            # 4차 시도: 쉼표로 구분된 단어들 (따옴표 제거)
            if ',' in response:
                # 배열 표시자 제거
                cleaned = re.sub(r'[\[\]"]', '', response)
                keywords = [kw.strip() for kw in cleaned.split(',') if kw.strip()]
                return keywords[:5]

            # 5차 시도: 공백으로 구분된 중요한 단어들
            words = response.split()
            # 길이가 2글자 이상인 단어들만 선택
            keywords = [word.strip('.,[]"') for word in words if len(word.strip('.,[]"')) > 1]
            return keywords[:5]

        except Exception as e:
            logger.error(f"키워드 파싱 실패: {e}")
            return []

    def _get_fallback_keywords(self, description: str, language: str) -> List[str]:
        """LLM 실패시 기본 키워드 추출"""
        import re

        keywords = []

        # 언어 추가
        if language:
            keywords.append(language)

        # 기술 용어 패턴 추출
        tech_patterns = [
            r'\b[A-Z][a-z]*[A-Z][a-zA-Z]*\b',  # CamelCase (Spring Boot, JWT 등)
            r'\b[A-Z]{2,}\b',  # 대문자 약어 (API, REST, JWT 등)
            r'\b\w+(?:\.js|\.py|\.java|\.go)\b',  # 파일 확장자
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, description)
            keywords.extend(matches)

        # 일반적인 프로그래밍 키워드
        common_terms = {
            '인증': 'authentication',
            '토큰': 'token',
            '데이터베이스': 'database',
            '서버': 'server',
            '클라이언트': 'client',
            '테스트': 'test',
            '구현': 'implementation',
            '예제': 'example',
        }

        for korean, english in common_terms.items():
            if korean in description:
                keywords.append(english)

        # 중복 제거하고 최대 5개
        unique_keywords = list(dict.fromkeys(keywords))  # 순서 유지하며 중복 제거
        return unique_keywords[:5]

    # 편의 메서드들
    def set_improvement_enabled(self, enabled: bool):
        """Self-improvement 기능 활성화/비활성화"""
        self.enable_self_improvement = enabled
        logger.info(f"Self-improvement 모드: {'활성화' if enabled else '비활성화'}")

    def set_quality_threshold(self, threshold: float):
        """품질 임계값 설정 (0-10)"""
        self.min_acceptable_score = max(0.0, min(10.0, threshold))
        logger.info(f"품질 임계값 설정: {self.min_acceptable_score}")

    def set_max_iterations(self, max_iter: int):
        """최대 반복 횟수 설정"""
        self.max_iterations = max(1, min(10, max_iter))
        logger.info(f"최대 반복 횟수 설정: {self.max_iterations}")

    def get_improvement_history(self, session_id: str) -> List[ImprovementIteration]:
        """세션별 개선 히스토리 조회"""
        return self.improvement_history.get(session_id, [])

    def get_improvement_statistics(self, session_id: str) -> Dict[str, Any]:
        """개선 통계 정보"""
        history = self.get_improvement_history(session_id)
        if not history:
            return {}

        total_iterations = len(history)
        avg_initial_score = sum(iter.reflection_result.score for iter in history) / total_iterations
        common_issues = {}

        for iteration in history:
            for issue in iteration.reflection_result.issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1

        return {
            "total_requests": len(set(iter.timestamp // 3600 for iter in history)),
            "total_iterations": total_iterations,
            "average_initial_score": round(avg_initial_score, 1),
            "most_common_issues": sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:5],
            "improvement_rate": round((total_iterations - len(history)) / max(total_iterations, 1) * 100, 1)
        }


# 싱글톤 인스턴스
ollama_service = OllamaService()
