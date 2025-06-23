"""
웹 검색 서비스 - Google Custom Search API를 통한 최신 정보 검색
"""
import os
import json
import logging
import urllib.parse
import aiohttp
from typing import List, Optional
from dotenv import load_dotenv
from .ollama_service import ollama_service

logger = logging.getLogger(__name__)


class WebSearchService:
    """웹 검색 전용 서비스"""

    def __init__(self):
        load_dotenv('.env.local')

        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.enable_web_search = True
        self.web_search_threshold = 0.7
        self.max_search_results = 3
        self.ollama_service = ollama_service

    async def should_perform_web_search(self, description: str, language: str, framework: Optional[str]) -> bool:
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
            judgment_response = await self.ollama_service.generate_response(judgment_prompt)
            score = float(judgment_response.strip())
            return score >= self.web_search_threshold
        except:
            return keyword_score >= 1

    async def perform_web_search(self, keywords: List[str]) -> str:
        """Google Custom Search API를 사용한 웹 검색"""
        try:
            if not self.google_api_key or not self.search_engine_id:
                logger.error("Google Search API 설정이 완료되지 않았습니다.")
                return "Google Search API 설정이 완료되지 않았습니다."

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

            logger.info(f"검색 URL: {search_url[:100]}...")  # API 키 노출 방지

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
                            return final_result
                        else:
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

    def get_optimized_query(self, description: str, language: str) -> List[str]:
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

            # LLM 호출 (동기 방식으로 호출)
            response = self.ollama_service.llm.invoke(description_optimized_query)
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

    def set_web_search_enabled(self, enabled: bool):
        """웹 검색 기능 활성화/비활성화"""
        self.enable_web_search = enabled
        logger.info(f"웹 검색 모드: {'활성화' if enabled else '비활성화'}")

    def set_search_threshold(self, threshold: float):
        """웹 검색 임계값 설정 (0-1)"""
        self.web_search_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"웹 검색 임계값 설정: {self.web_search_threshold}")

    def set_max_search_results(self, max_results: int):
        """최대 검색 결과 수 설정"""
        self.max_search_results = max(1, min(10, max_results))
        logger.info(f"최대 검색 결과 수 설정: {self.max_search_results}")


# 싱글톤 인스턴스
web_search_service = WebSearchService()