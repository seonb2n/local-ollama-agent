from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 웹 UI 라우터 (prefix 없이 루트 경로 사용)
ui_router = APIRouter(tags=["Web UI"])

@ui_router.get("/", response_class=HTMLResponse)
async def get_web_ui(request: Request):
    """웹 UI 메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@ui_router.get("/ui", response_class=HTMLResponse)
async def get_web_ui_alt(request: Request):
    """웹 UI 대체 경로"""
    return templates.TemplateResponse("index.html", {"request": request})

# 필요한 디렉토리 생성
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)