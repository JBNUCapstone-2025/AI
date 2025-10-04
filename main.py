# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ✅ llm_utils에서 함수 가져오기
from llm_utils import extract_emotion, get_embedding, generate_character_response
from vector_db import find_dissimilar_emotion_key, get_random_content

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
    character: str = "강아지"  # 기본값은 강아지

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    1. 문장에서 감정 추출
    2. 감정 벡터를 만들어 벡터 DB에서 반대 감정 찾기
    3. 반대 감정 기반 콘텐츠 추천
    4. 캐릭터 말투로 응답 생성
    """
    # 1. 감정 추출
    emotion = extract_emotion(request.sentence)

    # 2. 감정 임베딩 생성 후 벡터 DB에서 반대 감정 찾기
    emotion_vector = get_embedding(emotion)
    if emotion_vector is None:
        return {"answer": "감정 분석에 실패했어요. 다시 시도해주세요."}

    opposite_emotion = find_dissimilar_emotion_key(emotion_vector)

    # 3. 반대 감정 기반으로 콘텐츠 추천
    recommended_content = get_random_content(opposite_emotion)

    # 4. 캐릭터 말투로 응답 생성
    answer = generate_character_response(
        character=request.character,
        user_emotion=emotion,
        content=recommended_content
    )

    return {
        "answer": answer,
        "detected_emotion": emotion,
        "recommended_emotion": opposite_emotion,
        "content": recommended_content
    }

@app.get("/")
def read_root():
    return {"Hello": "World"}
