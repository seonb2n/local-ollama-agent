"""
Ollama 서비스 - LLM 모델과의 통신을 담당
"""
import os
import sys
import time

import aiohttp
import logging
from typing import List, Optional, Dict, Any, Tuple
from langchain_community.llms import Ollama
import json
from .context_manager import context_manager
from .dto.self_improvements import ImprovementIteration, ReflectionResult

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
        self.min_acceptable_score = 9.0  # 최소 허용 점수
        self.improvement_history: Dict[str, List[ImprovementIteration]] = {}
        self.enable_self_improvement = True

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

        # 2. 요청 유형 분석 및 프롬프트 생성
        enhanced_prompt = self._build_context_aware_prompt(
            description, language, framework, context_info
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
            context_info: str
    ) -> str:
        """컨텍스트를 고려한 프롬프트 구성"""

        is_modification_request = self._is_code_modification_request(description)

        if context_info and is_modification_request:
            # 기존 코드 수정 요청
            return f"""
이전 대화 컨텍스트:
{context_info}

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
