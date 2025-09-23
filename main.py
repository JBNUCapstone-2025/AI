# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ✅ 주석 해제: llm_utils에서 함수 가져오기
from llm_utils import get_llm_answer

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    sentence: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # ✅ 프런트 입력을 LLM에 전달
    answer = get_llm_answer(request.sentence)
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"Hello": "World"}
