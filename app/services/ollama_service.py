"""
Ollama LLM 통신 전용 서비스 - LLM 모델과의 통신만 담당
"""
import os
import sys
import aiohttp
import logging
from dotenv import load_dotenv
from langchain_community.llms import Ollama

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ..config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Ollama LLM 통신 전용 서비스 클래스"""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.default_model = settings.default_model
        self.backup_model = settings.backup_model
        self.llm = None

        load_dotenv('.env.local')

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

    async def get_available_models(self) -> list[str]:
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

    async def generate_response(self, prompt: str) -> str:
        """프롬프트를 받아 LLM 응답 생성"""
        if not self.llm:
            await self.initialize()

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            raise

    def get_current_model(self) -> str:
        """현재 사용 중인 모델명 반환"""
        return self.default_model if self.llm else None


# 싱글톤 인스턴스
ollama_service = OllamaService()