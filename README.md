# 🤖 LLM Agent 코드 생성기

> 로컬 LLM을 활용한 지능형 코드 생성 에이전트

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.ai)


![Desktop View](/static/img.png){: width="600" height="400" }

## ✨ 주요 특징

- 🧠 **로컬 LLM 기반**: Ollama를 활용한 완전한 오프라인 코드 생성
- 🔄 **Self-Improvement**: 생성된 코드를 자동으로 분석하고 개선
- 📚 **RAG 시스템**: 기술 문서와 코드 패턴을 학습하여 더 나은 코드 생성
- 🌐 **웹 검색 연동**: 최신 기술 정보를 반영한 코드 생성
- 💬 **컨텍스트 관리**: 세션별 대화 히스토리와 프로젝트 컨텍스트 유지
- 🎨 **직관적인 UI**: 채팅 기반의 사용자 친화적 웹 인터페이스


### 핵심 컴포넌트

- **CodeGenerationFacade**: 전체 코드 생성 프로세스를 조율하는 퍼사드 패턴
- **Ollama Service**: 로컬 LLM과의 통신 및 모델 관리
- **Context Service**: 세션별 대화 컨텍스트 및 프로젝트 히스토리 관리
- **RAG Integration**: 벡터 데이터베이스를 활용한 지식 검색 시스템
- **Improvement Service**: 생성된 코드의 품질 분석 및 자동 개선
- **Web Search Service**: 실시간 기술 정보 수집 및 통합

### 필수 요구사항

- Python 3.8+
- [Ollama](https://ollama.ai) 설치

### 개발 환경 설정

1. **개발 의존성 설치**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **pre-commit 훅 설정**
   ```bash
   pre-commit install
   ```

3. **코드 포맷팅**
   ```bash
   black app/
   isort app/
   flake8 app/
   ```

### API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`



---
