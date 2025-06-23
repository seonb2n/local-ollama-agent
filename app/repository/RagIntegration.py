import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from typing import List, Optional, Dict, Any, Tuple
from txtai import Embeddings, RAG
import logging
import json

logger = logging.getLogger(__name__)


class RAGIntegration:
    """FastAPI와 LLM을 위한 RAG 통합 클래스"""

    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = None
        self.rag_pipeline = None
        self.embeddings_model = embeddings_model
        self.knowledge_base = []
        self.is_initialized = False

    async def initialize(self, ollama_model: str = "deepseek-coder-v2:16b-lite-instruct-q4_K_M"):
        """RAG 시스템 초기화"""
        try:
            # 임베딩 데이터베이스만 초기화 (RAG 파이프라인 제거)
            self.embeddings = Embeddings(
                path=self.embeddings_model,
                content=True
            )

            # 기본 프로그래밍 지식 베이스 로드
            await self._load_programming_knowledge()

            # ⭐️ RAG 파이프라인 제거, 수동으로 구현
            self.ollama_model = ollama_model
            self.is_initialized = True
            logger.info("✅ RAG 시스템 초기화 완료")

        except Exception as e:
            logger.error(f"❌ RAG 시스템 초기화 실패: {e}")
            raise

    async def search_knowledge(self, keyword: List[str], max_results: int = 3) -> str:
        """지식 베이스에서 관련 문서 검색"""
        if not self.is_initialized:
            return ""

        try:
            # 벡터 검색 수행
            optimized_query = " ".join(keyword)
            search_results = self.embeddings.search(optimized_query, max_results)

            if not search_results:
                # 키워드 검색 실패 시 원본 쿼리로 재시도
                logger.info("키워드 검색 실패, 원본 쿼리로 재시도")
                search_results = self.embeddings.search(keyword, max_results)

            return self._format_search_results(search_results)

        except Exception as e:
            logger.error(f"RAG 지식 검색 실패: {e}")
            return ""

    def _format_search_results(self, search_results) -> str:
        """검색 결과를 읽기 쉬운 형태로 포맷팅"""
        if not search_results:
            return ""

        formatted_results = []

        for i, result in enumerate(search_results):
            try:
                # txtai 검색 결과 형태에 따라 처리
                if isinstance(result, tuple):
                    # (id, score, text, metadata) 또는 (id, score) 형태
                    if len(result) >= 3:
                        doc_id, score, text = result[0], result[1], result[2]
                        metadata = result[3] if len(result) > 3 else {}
                    else:
                        doc_id, score = result[0], result[1]
                        text = f"문서 ID: {doc_id}"
                        metadata = {}

                elif isinstance(result, dict):
                    # dict 형태 결과
                    doc_id = result.get('id', f'doc_{i}')
                    score = result.get('score', 0.0)
                    text = result.get('text', result.get('content', 'No content available'))
                    metadata = result.get('metadata', {})

                else:
                    # 기타 형태 - 단순 문자열이나 예상치 못한 형태
                    doc_id = f'result_{i}'
                    score = 1.0 - (i * 0.1)  # 순서 기반 가상 점수
                    text = str(result)
                    metadata = {}

                # 메타데이터에서 정보 추출
                topic = metadata.get('topic', 'general')
                level = metadata.get('level', 'intermediate')
                category = metadata.get('category', '')
                version = metadata.get('version', '')

                # 제목 생성
                title_parts = [topic.upper()]
                if version:
                    title_parts.append(f"v{version}")
                if category:
                    title_parts.append(f"({category})")

                title = " ".join(title_parts)

                # 텍스트 길이 제한 (너무 길면 자르기)
                max_text_length = 200
                if len(text) > max_text_length:
                    text = text[:max_text_length] + "..."

                # 결과 포맷팅
                formatted_result = f"""
    {i + 1}. **{title}** [{level}]
      {text}
      📊 관련도: {score:.3f}
    """

                formatted_results.append(formatted_result)

            except Exception as e:
                logger.error(f"검색 결과 포맷팅 실패 (결과 {i}): {e}")
                # 에러 발생 시 기본 포맷
                formatted_results.append(f"""
    {i + 1}. **검색 결과**
      {str(result)[:100]}...
      📊 관련도: 0.500
    """)

        return "\n".join(formatted_results)

    async def _load_programming_knowledge(self):
        """프로그래밍 관련 지식 베이스 로드 (벡터 DB 연동)"""

        # 지속성 있는 임베딩 저장소 설정
        index_path = "./rag_data/programming_embeddings"

        try:
            # 기존 인덱스가 있으면 로드
            if os.path.exists(index_path):
                logger.info("📂 기존 지식 베이스 로드 중...")
                self.embeddings.load(index_path)
                logger.info("✅ 기존 지식 베이스 로드 완료")
                return

        except Exception as e:
            logger.warning(f"기존 인덱스 로드 실패, 새로 생성: {e}")

        # 새로운 지식 베이스 생성
        logger.info("🔄 새로운 지식 베이스 생성 중...")

        programming_knowledge = [
            {
                "id": "java_24_late_barrier_expansion",
                "text": "Java 24의 Late Barrier Expansion for G1은 G1 가비지 컬렉터의 성능을 개선하는 기술입니다. 메모리 배리어 삽입을 런타임까지 지연시켜 컴파일 시점의 최적화 기회를 늘립니다. -XX:+UseLateBarrierExpansion 플래그로 활성화할 수 있으며, 고성능 애플리케이션에서 GC 오버헤드를 줄이는 데 효과적입니다.",
                "metadata": {"topic": "java", "level": "advanced", "version": "24"}
            },
            {
                "id": "java_gc_tuning",
                "text": "Java GC 튜닝을 위해서는 힙 사이즈 설정(-Xms, -Xmx), GC 알고리즘 선택(-XX:+UseG1GC, -XX:+UseZGC), GC 로깅(-Xlog:gc), 그리고 애플리케이션별 최적화가 필요합니다. G1GC에서는 -XX:MaxGCPauseMillis로 목표 일시정지 시간을 설정할 수 있습니다.",
                "metadata": {"topic": "java", "level": "advanced", "category": "performance"}
            },
            {
                "id": "java_jvm_monitoring",
                "text": "JVM 성능 모니터링을 위해서는 JProfiler, VisualVM, JConsole 등의 도구를 사용할 수 있습니다. JFR(Java Flight Recorder)를 통해 저오버헤드로 성능 데이터를 수집하고, jstat, jmap, jstack 명령어로 실시간 JVM 상태를 확인할 수 있습니다.",
                "metadata": {"topic": "java", "level": "intermediate", "category": "monitoring"}
            },
            {
                "id": "java_memory_management",
                "text": "Java 메모리 관리에서 힙은 Young Generation(Eden, Survivor 공간)과 Old Generation으로 나뉩니다. 객체는 Eden에서 생성되어 GC를 거쳐 Old Generation으로 승격됩니다. OutOfMemoryError 해결을 위해서는 힙 덤프 분석과 메모리 누수 찾기가 중요합니다.",
                "metadata": {"topic": "java", "level": "intermediate", "category": "memory"}
            },
            {
                "id": "java_performance_testing",
                "text": "Java 성능 테스트를 위해서는 JMH(Java Microbenchmark Harness)를 사용하여 정확한 벤치마크를 작성할 수 있습니다. @Benchmark 어노테이션으로 테스트 메서드를 지정하고, @BenchmarkMode로 측정 방식을 설정합니다. Warmup과 측정 반복 횟수 설정이 정확한 결과를 위해 중요합니다.",
                "metadata": {"topic": "java", "level": "advanced", "category": "testing"}
            },
            {
                "id": "java_latest_features",
                "text": "최신 Java 버전들의 주요 기능: Java 21 LTS의 Virtual Threads, Pattern Matching, Record Patterns; Java 22의 Unnamed Variables, String Templates; Java 23의 Primitive Types in Patterns; Java 24 Preview의 Late Barrier Expansion. --enable-preview 플래그로 미리보기 기능을 사용할 수 있습니다.",
                "metadata": {"topic": "java", "level": "advanced", "category": "features"}
            },
            {
                "id": "jvm_flags_optimization",
                "text": "JVM 최적화를 위한 주요 플래그들: -XX:+UseStringDeduplication (문자열 중복제거), -XX:+UseCompressedOops (압축 포인터), -XX:+TieredCompilation (계층 컴파일), -XX:ReservedCodeCacheSize (코드 캐시 크기), -XX:+UseNUMA (NUMA 최적화). 프로덕션 환경에서는 충분한 테스트 후 적용해야 합니다.",
                "metadata": {"topic": "java", "level": "expert", "category": "optimization"}
            },
            {
                "id": "g1gc_configuration",
                "text": "G1GC 설정 가이드: -XX:+UseG1GC로 활성화, -XX:MaxGCPauseMillis=200으로 목표 일시정지 시간 설정, -XX:G1HeapRegionSize로 리전 크기 조정, -XX:G1NewSizePercent와 -XX:G1MaxNewSizePercent로 Young Generation 비율 설정. 대용량 힙(4GB 이상)에서 효과적입니다.",
                "metadata": {"topic": "java", "level": "advanced", "category": "gc"}
            }
        ]

        # ⭐️ 핵심 수정: 메타데이터를 JSON 문자열로 변환
        processed_data = []
        for item in programming_knowledge:
            if item.get("text"):  # 빈 항목 제외
                metadata = item.get("metadata", {})
                # dict를 JSON 문자열로 변환
                metadata_json = json.dumps(metadata) if metadata else None

                processed_data.append((
                    item["id"],
                    item["text"],
                    metadata_json  # JSON 문자열로 전달
                ))

        # 임베딩 인덱스 생성 및 디스크에 저장
        self.embeddings.index(processed_data)

        # 벡터 DB에 지속적으로 저장
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.embeddings.save(index_path)

        logger.info(f"💾 지식 베이스 생성 및 저장 완료: {len(processed_data)}개 문서")

    async def _fetch_additional_knowledge(self) -> List[dict]:
        """추가 지식을 웹에서 수집 (선택사항)"""
        additional_docs = []

        # 여기에 웹 크롤링이나 API 호출 로직 추가 가능
        # 예: Python 공식 문서, FastAPI 문서 등

        return additional_docs

    async def should_use_rag(self, query: str) -> bool:
        """RAG 사용 여부 판단"""
        # 프로그래밍 관련 키워드 확인
        programming_keywords = [
            "python", "fastapi", "docker", "async", "await", "api", "함수", "클래스",
            "라이브러리", "설치", "사용법", "예제", "코드", "오류", "에러", "해결",
            "최적화", "성능", "구현", "방법", "how to", "사용방법"
        ]

        query_lower = query.lower()
        keyword_found = any(keyword in query_lower for keyword in programming_keywords)

        # 질문 형태인지 확인
        question_indicators = ["?", "어떻게", "무엇", "왜", "언제", "어디서", "how", "what", "why", "when"]
        is_question = any(indicator in query_lower for indicator in question_indicators)

        return keyword_found or is_question

    async def get_rag_response(self, query: str) -> str:
        """RAG를 사용하여 응답 생성"""
        if not self.is_initialized:
            raise Exception("RAG 시스템이 초기화되지 않았습니다.")

        try:
            # RAG 파이프라인을 통해 응답 생성
            response = self.rag_pipeline(query)
            return response

        except Exception as e:
            logger.error(f"RAG 응답 생성 실패: {e}")
            return f"RAG 시스템에서 오류가 발생했습니다: {str(e)}"

    async def add_document(self, text: str, metadata: dict = None):
        """새 문서를 지식 베이스에 추가"""
        if not self.is_initialized:
            return

        try:
            doc_id = f"custom_{len(self.knowledge_base)}"
            self.knowledge_base.append({"id": doc_id, "text": text, "metadata": metadata or {}})

            # 기존 인덱스에 문서 추가 (upsert)
            self.embeddings.upsert([(doc_id, text, metadata or {})])

            logger.info(f"📄 새 문서 추가됨: {doc_id}")

        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
