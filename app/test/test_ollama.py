"""
Ollama 연결 및 모델 테스트 스크립트
"""
import requests
import json
import time

# Ollama API 설정
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "deepseek-coder-v2:16b-lite-instruct-q4_K_M"


def test_ollama_connection():
    """Ollama 서버 연결 테스트"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama 서버 연결 성공!")
            print("📋 사용 가능한 모델:")
            for model in models.get('models', []):
                print(f"  - {model['name']}")
            return True
        else:
            print(f"❌ Ollama 서버 응답 오류: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama 서버에 연결할 수 없습니다. 'ollama serve' 명령으로 서버를 시작해주세요.")
        return False
    except Exception as e:
        print(f"❌ 연결 테스트 실패: {e}")
        return False


def test_model_generation():
    """모델 코드 생성 테스트"""
    print(f"\n🤖 모델 테스트 시작: {MODEL_NAME}")

    # 테스트 프롬프트
    prompt = """
파이썬으로 주식 가격을 조회하는 간단한 함수를 작성해주세요.
함수명은 get_stock_price이고, 종목 코드를 입력받아서 현재가를 반환해야 합니다.
yfinance 라이브러리를 사용해주세요.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        print("⏳ 코드 생성 중... (30초-2분 정도 소요될 수 있습니다)")
        start_time = time.time()

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120  # 2분 타임아웃
        )

        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            generated_code = result.get('response', '')

            print(f"✅ 코드 생성 성공! (소요시간: {end_time - start_time:.1f}초)")
            print("\n" + "=" * 50)
            print("📝 생성된 코드:")
            print("=" * 50)
            print(generated_code)
            print("=" * 50)
            return True
        else:
            print(f"❌ 코드 생성 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("⏰ 요청 시간 초과 (2분). 모델이 너무 오래 걸리고 있습니다.")
        return False
    except Exception as e:
        print(f"❌ 코드 생성 오류: {e}")
        return False


def test_langchain_integration():
    """LangChain 연동 테스트"""
    try:
        from langchain_community.llms import Ollama

        print(f"\n🔗 LangChain 연동 테스트")

        # LangChain Ollama 객체 생성
        llm = Ollama(model=MODEL_NAME)

        # 간단한 테스트
        prompt = "파이썬 리스트를 딕셔너리로 변환하는 함수를 작성해주세요."

        print("⏳ LangChain으로 코드 생성 중...")
        start_time = time.time()

        response = llm.invoke(prompt)

        end_time = time.time()

        print(f"✅ LangChain 연동 성공! (소요시간: {end_time - start_time:.1f}초)")
        print("\n" + "=" * 30)
        print("📝 LangChain 응답:")
        print("=" * 30)
        print(response)
        print("=" * 30)
        return True

    except ImportError:
        print("❌ LangChain 라이브러리가 설치되지 않았습니다.")
        print("설치 명령: pip install langchain langchain-community")
        return False
    except Exception as e:
        print(f"❌ LangChain 연동 실패: {e}")
        return False


def main():
    """모든 테스트 실행"""
    print("🚀 Ollama 모델 테스트 시작")
    print(f"📍 대상 모델: {MODEL_NAME}")
    print(f"🌐 Ollama URL: {OLLAMA_URL}")
    print("-" * 60)

    # 1. 연결 테스트
    if not test_ollama_connection():
        print("\n❌ Ollama 서버 연결에 실패했습니다. 테스트를 중단합니다.")
        return

    # 2. 모델 생성 테스트
    if not test_model_generation():
        print("\n❌ 모델 코드 생성에 실패했습니다.")
        return

    # 3. LangChain 연동 테스트
    if not test_langchain_integration():
        print("\n❌ LangChain 연동에 실패했습니다.")
        return

    print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    print("✅ Ollama 서버 연결됨")
    print("✅ 모델 코드 생성 작동함")
    print("✅ LangChain 연동 성공")
    print("\n다음 단계: FastAPI 에이전트 구현을 진행할 수 있습니다!")


if __name__ == "__main__":
    main()