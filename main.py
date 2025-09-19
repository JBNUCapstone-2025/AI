import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from dotenv import load_dotenv

import llm_utils
import vector_db

# .env 파일에서 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

class UserRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 API 키 유무를 확인합니다."""
    if not os.getenv("GOOGLE_API_KEY"):
        print("경고: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        print(".env 파일을 생성하고 API 키를 입력해주세요.")
    if not vector_db.IS_DB_READY:
        print("="*50)
        print("경고: 벡터 데이터베이스 파일이 없습니다.")
        print("애플리케이션을 실행하기 전에, 터미널에서 다음 명령어를 실행해주세요:")
        print("python populate_db.py")
        print("="*50)


@app.post("/api/v1/comfort")
async def get_comfort(request: UserRequest):
    """
    사용자 입력을 받아 감정을 분석하고, 위로의 메시지와 콘텐츠를 추천합니다.
    """
    if not vector_db.IS_DB_READY:
        raise HTTPException(status_code=500, detail="서버가 아직 준비되지 않았습니다. 관리자에게 문의하세요.")

    if not request.text:
        raise HTTPException(status_code=400, detail="입력 텍스트가 없습니다.")

    # 1. 감정 추출
    user_emotion = llm_utils.extract_emotion(request.text)
    if not user_emotion:
        raise HTTPException(status_code=500, detail="감정을 분석하지 못했습니다.")

    # 2. 감정 벡터화
    emotion_vector = llm_utils.get_embedding(user_emotion, task_type="RETRIEVAL_QUERY")
    if emotion_vector is None:
        raise HTTPException(status_code=500, detail="감정을 벡터로 변환하지 못했습니다.")

    # 3. 유사도가 낮은 감정 키(콘텐츠 추천용) 검색
    dissimilar_emotion_key = vector_db.find_dissimilar_emotion_key(np.array(emotion_vector))

    # 4. 콘텐츠 랜덤 선택
    random_content = vector_db.get_random_content(dissimilar_emotion_key)
    if "error" in random_content:
        raise HTTPException(status_code=500, detail=random_content["error"])

    # 5. 최종 메시지 생성
    final_message = llm_utils.generate_comforting_message(user_emotion, random_content)

    return {
        "user_emotion": user_emotion,
        "recommended_content_emotion": dissimilar_emotion_key,
        "content": random_content,
        "message": final_message
    }

@app.get("/")
def read_root():
    return {"message": "감정 분석 및 위로 메시지 API에 오신 것을 환영합니다. POST /api/v1/comfort 로 요청하세요."}

