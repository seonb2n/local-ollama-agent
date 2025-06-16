import os

class Settings:
    """설정 클래스 - 간단한 버전"""

    def __init__(self):
        # Ollama 설정
        self.ollama_base_url = "http://localhost:11434"
        self.default_model = "deepseek-coder-v2:16b-lite-instruct-q4_K_M"
        self.backup_model = "llama3.1:8b"

        # FastAPI 설정
        self.app_name = "Code Generator Agent"
        self.app_version = "1.0.0"
        self.debug = True
        self.host = "0.0.0.0"
        self.port = 8000

        # 파일 설정
        self.generated_code_path = "./generated_code"
        self.max_file_size = 10485760  # 10MB

        # API 설정
        self.max_request_time = 300  # 5분
        self.enable_cors = True


# 싱글톤 설정 인스턴스
settings = Settings()


# 생성된 코드 저장 폴더 확인/생성
def ensure_directories():
    """필요한 디렉토리들을 생성합니다."""
    os.makedirs(settings.generated_code_path, exist_ok=True)
    os.makedirs("logs", exist_ok=True)


# 앱 시작시 실행
ensure_directories()