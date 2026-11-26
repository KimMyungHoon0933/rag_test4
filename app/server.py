"""교재 484페이지: Ollama를 /llm API로 제공하는 LangServe 서버."""
import os
import secrets

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from langserve import add_routes

from app.model import DEFAULT_BASE_URL, DEFAULT_DOCUMENT_MODEL, DEFAULT_MODEL, make_llm

load_dotenv()


def verify_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("RAG_API_KEY", "")
    if expected and not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="API key가 올바르지 않습니다.")


app = FastAPI(title="Ollama LangServe", version="1.0")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    # 서버 자체의 상태. 실제 모델 응답 여부는 /llm/invoke로 확인합니다.
    return {"status": "ok", "model": model_name}


model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
llm = make_llm(model_name, base_url)
document_llm = make_llm(
    os.getenv("OLLAMA_DOCUMENT_MODEL", DEFAULT_DOCUMENT_MODEL), base_url, document_reply=True,
)


# Streamlit의 Python 서버가 호출하므로 브라우저용 CORS 설정은 필요 없습니다.
add_routes(app, llm, path="/llm", dependencies=[Depends(verify_key)])
add_routes(app, document_llm, path="/rag", dependencies=[Depends(verify_key)])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
