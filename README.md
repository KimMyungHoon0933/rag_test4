# 479~505페이지 실습: 원격 LLM + RAG

개인 문서를 로컬 Ollama로 검색하고 질문할 수 있는 Streamlit 앱입니다. 원격 연결이 필요한 경우 교재의 Ollama → LangServe → ngrok 구성을 사용할 수 있습니다.

## 포함 파일과 교재 대응

| 파일 | 역할 | PDF 페이지 |
|---|---|---|
| app/server.py | Ollama 모델을 /llm API로 제공 | 480~485 |
| document_utils.py | PDF/DOCX/PPTX 텍스트 및 출처 추출 | 493~494 |
| rag_test4.py | 토큰 분할, 한국어 임베딩, FAISS MMR 검색, 채팅/스트리밍 | 493~500 |
| requirements*.txt | 클라이언트와 서버 설치 목록 | 481, 502 |
| .streamlit/secrets.toml.example | Cloud에서 원격 서버 연결 설정 | 501~505 |

PEFT 학습은 479페이지 이전에 완료되었다는 전제입니다. 이 프로젝트에는 학습 코드나 모델 가중치가 포함되지 않습니다. 이미 Ollama에 등록한 맞춤형 모델이 있으면 그 이름을 사용하세요. 없으면 기본 llama3 모델로 연결을 먼저 확인할 수 있습니다.

## 1. 설치 (Windows PowerShell)

Python 3.11과 Ollama를 설치한 다음, 압축을 푼 `remote_rag` 폴더에서 실행합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-server.txt
```

처음에는 PyTorch 및 임베딩 모델 다운로드로 시간과 디스크 공간이 필요합니다.
기본 모델과 로컬 주소를 사용하면 별도 설정 파일이 필요 없습니다. 모델이나 주소를 바꾸려면 `.env.example`을 `.env`로 복사해 수정하세요. 환경 변수는 `.env`보다 우선하며 Streamlit은 그다음 Cloud Secrets를 읽습니다.

## 2. Ollama 모델 준비

```powershell
ollama list
ollama show Llama3_ko_8b_Q8:latest
```

로컬 Streamlit과 LangServe의 기본 모델은 문서 답변과 일반 답변 모두 `Llama3_ko_8b_Q8:latest`입니다. 모델 등록을 다시 해야 할 때는 프로젝트의 `model` 폴더에서 `ollama create Llama3_ko_8b_Q8:latest -f llm`을 실행합니다.

로컬 모델을 따로 지정할 때는 문서 답변에 `RAG_DOCUMENT_MODEL`, 일반 답변에 `RAG_LOCAL_MODEL`을 사용합니다. 선택형 LangServe 서버에서는 문서 답변 `/rag`을 `OLLAMA_DOCUMENT_MODEL`로, 일반 답변 `/llm`을 `OLLAMA_MODEL`로 지정합니다.

Ollama 앱이 실행 중이어야 합니다. Ollama 서버가 실행되지 않은 경우 별도 터미널에서 `ollama serve`를 실행합니다. 이미 실행 중이면 추가 실행하지 않습니다.

## 3. LangServe 실행 (선택)

로컬 Streamlit은 `LANGSERVE_URL`이 비어 있으면 Ollama에 직접 연결합니다. LangServe 경로를 사용할 때는 `.env`에서 `LANGSERVE_URL=http://127.0.0.1:8000/llm/`을 설정한 뒤 아래 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

- API 문서: http://127.0.0.1:8000/docs
- 서버 상태: http://127.0.0.1:8000/health
- 일반 답변 경로 http://127.0.0.1:8000/llm/
- 문서 답변 경로 http://127.0.0.1:8000/rag/

`/health`는 서버 프로세스 상태만 확인합니다. 모델 연결은 다음 호출로 확인합니다. API 키를 설정했다면 아래 헤더 값도 바꿉니다.

```powershell
$body = @{ input = '안녕하세요. 한 문장으로 자기소개해 주세요.' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/llm/invoke' -Method Post -ContentType 'application/json; charset=utf-8' -Headers @{'X-API-Key'=''} -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

## 4. Streamlit 실행

프로젝트 폴더에서 아래 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run rag_test4.py --server.address 127.0.0.1
```

http://127.0.0.1:8501 에서 질문합니다. 프로젝트의 `document` 폴더에 있는 PDF/DOCX/PPTX는 세션을 열 때 자동으로 준비됩니다. 업로드한 파일은 기본 문서 대신 자동으로 검색합니다. 업로드 파일을 지우면 기본 문서 검색으로 돌아갑니다.

채팅은 매 질문에서 문서를 먼저 살펴봅니다. 모델이 제시한 원문 구절과 문서 번호를 확인할 수 있으면 자연스러운 답변과 **답변 출처**를 표시합니다. 근거를 확인하지 못한 질문은 같은 채팅에서 일반 답변으로 이어집니다. 문서 답변에는 직전 대화의 일부가 함께 전달됩니다.

## 5. ngrok 연결 (로컬 확인 후 선택)

ngrok을 설치하고 본인 계정의 인증 토큰과 도메인을 사용합니다. 교재의 토큰/URL은 사용하지 않습니다.

1. `.env`의 `RAG_API_KEY`에 임의의 긴 문자열을 넣고 LangServe를 재시작합니다.
2. 아래 명령을 별도 터미널에서 실행합니다.

```powershell
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
ngrok http 8000
```

발급된 HTTPS 주소 뒤에 `/llm/`을 붙여 `.env`의 `LANGSERVE_URL`을 수정합니다. Streamlit도 재시작합니다.
예: `LANGSERVE_URL=https://YOUR-DOMAIN.ngrok-free.app/llm/`
고정 도메인을 계정에 할당했다면 `ngrok http 8000 --url https://YOUR-DOMAIN.ngrok-free.app`을 사용합니다.

ngrok은 8000번 LangServe에 연결합니다. Ollama 11434번 포트를 직접 연결할 필요가 없습니다.

## 6. Streamlit Community Cloud 배포

1. 앱 소스를 본인 GitHub 저장소에 올립니다. 개인 문서가 있는 `document/`, `.env`, 실제 `secrets.toml`, 가상 환경은 제외합니다.
2. Streamlit Community Cloud에서 저장소를 선택하고 실행 파일을 `rag_test4.py`, Python을 3.11로 지정합니다.
3. 앱 설정의 Secrets에 아래 내용을 본인 값으로 입력합니다.

```toml
LANGSERVE_URL = "https://YOUR-DOMAIN.ngrok-free.app/llm/"
RAG_API_KEY = "LOCAL_SERVER와_동일한_키"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
```

Cloud에는 requirements.txt의 클라이언트 패키지가 설치됩니다. 서버 패키지는 로컬 PC에서만 필요합니다.
실제 배포는 수행하지 않았습니다. 사용 중에는 로컬 PC, Ollama, LangServe, ngrok이 모두 켜져 있어야 합니다.
Streamlit 쪽에서 임베딩을 계산하므로 Cloud 메모리 한도와 최초 다운로드 시간을 고려해 작은 문서부터 확인하세요.
선택된 문서의 검색 결과와 질문은 원격 LLM 서버에 전달됩니다. 기본 일반 답변 URL이 `/llm/`으로 끝나면 문서 답변은 같은 서버의 `/rag/`로 요청합니다. 별도 URL은 `RAG_DOCUMENT_URL`로 지정할 수 있습니다.

## 구현 차이와 제한

- 교재의 LangChain 0.1 대신 LangServe 0.3.1 및 LangChain 0.3 계열을 사용합니다. requirements는 호환 범위이며 전체 환경의 고정 lock 파일은 아닙니다.
- `ChatOllama`는 `langchain_ollama`, 임베딩은 `langchain_huggingface`에서 가져옵니다.
- PDF는 페이지별, PPTX는 슬라이드별 출처를 보존합니다. DOCX 표도 추출합니다.
- 교재의 900/100 토큰 분할, ko-sroberta 임베딩, FAISS MMR 검색을 유지합니다. 임베딩 모델 자체의 최대 입력 길이에 따라 긴 조각 뒷부분이 잘릴 수 있습니다.
- 스캔 PDF, 이미지 속 코드/텍스트는 OCR하지 않습니다. 이 교재 PDF의 코드 스크린샷도 텍스트 추출만으로는 검색되지 않을 수 있습니다.
- 문서 인덱스와 대화는 세션 메모리에만 보관됩니다. 기본 문서는 새 세션에서 자동으로 처리되며 업로드 문서는 세션마다 처리할 수 있습니다.
- 인증 키가 빈 값이면 로컬 실습용 무인증입니다. ngrok 연결 시 서버와 클라이언트 양쪽에 같은 키를 설정합니다.

## 검증

Python 문법 및 PDF/DOCX/PPTX 텍스트 추출은 동봉된 tests/test_documents.py로 검사합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

기본 문서 인덱싱과 일반 질문 모드의 실제 Ollama 응답을 Streamlit 테스트에서 확인했습니다. Cloud 배포는 수행하지 않았습니다.

## 공식 참고 자료

- LangServe: https://github.com/langchain-ai/langserve
- ChatOllama: https://docs.langchain.com/oss/python/integrations/chat/ollama
- ngrok: https://ngrok.com/docs/start

LangServe는 현재 보관(archive)된 프로젝트이지만 교재 실습의 구조를 재현하기 위해 사용했습니다.