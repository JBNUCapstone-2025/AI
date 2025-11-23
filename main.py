# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ✅ AI 핵심 기능 import (정리된 구조)
from ai_core.llm import (
    extract_emotion,
    extract_recent_emotion,
    get_embedding,
    generate_empathetic_response,
    generate_recommendation_response
)
from ai_core.vector_db import get_recommendation_by_emotion
from ai_core.recommendation import format_recommendation

# Backend 기능 import
from app.api import auth, diary

app = FastAPI(
    title="ICSYF AI Integrated API",
    description="감정 기반 정서 관리 플랫폼 통합 API (AI + Backend)",
    version="2.0.0"
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Backend API 라우터 등록
app.include_router(auth.router)
app.include_router(diary.router)

class ChatRequest(BaseModel):
    sentence: str
    character: str = "강아지"  # 기본값은 강아지
    conversation_history: list[dict[str, str]] = [] # 이전 대화


class RecommendRequest(BaseModel):
    type: str  # 도서, 음악, 식사
    character: str = "강아지"
    conversation_history: str = ""

class DiaryRequest(BaseModel):
    diary: str
    class_type: str = "일반"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    1. 문장에서 감정 추출
    2. 공감 기능이 강화된 응답 생성
    3. 감정 벡터를 만들어 벡터 DB에서 반대 감정 찾기
    4. 반대 감정 기반 콘텐츠 추천
    5. 캐릭터 말투로 응답 생성
    """
    # 멀티턴 구현

    # sentence, character, conversation_hisotry

    # request.conversation_history(이전 대화 기록) 있으면 해당 기록 사용
    # 없으면 빈 리스트
    if request.conversation_history:
        chat_history = request.conversation_history
    else:
        chat_history = []
    
    '''
    for i in chat_history:
        if i['role'] == 'user':
            full_conversation = " ".join(i['content'])
    '''

    # 1. 감정 추출 (llm_utils.py)
    emotion = extract_emotion(request.sentence)

    # 현재 사용자의 질문 chat_history에 저장 
    chat_history.append({"role" : "user", "content":request.sentence})

    # 2. 공감 기능 강화 - 먼저 사용자의 감정에 공감 (llm_utils.py)
    empathy_response = generate_empathetic_response(
        character=request.character,
        user_sentence=request.sentence,
        user_emotion=emotion,
        chat_history = chat_history
    )

    # assistant의 문장 chat_history에 저장
    chat_history.append({"role" : "assistant", "content": empathy_response})

    return {
        "answer": empathy_response,
        "detected_emotion": emotion,
        "conversation_history" : chat_history
    }


@app.post("/api/recommend")
async def recommend(request: RecommendRequest): 
    """
    RAG 기반 지능형 추천 시스템
    1. 전체 대화 기록에서 최근 감정 분석
    2. 벡터 DB를 활용한 반대 감정 찾기
    3. 대화 내용과 가장 관련성 높은 콘텐츠 추천
    4. 캐릭터 말투로 응답 생성
    """

    
    # 1. 전체 대화에서 최근 감정 추출
    conversation = request.conversation_history or "평범한 하루"

    # 최근 감정 추출 (여러 감정이 있을 경우 가장 최근 것 선택) (llm_utils.py)
    recent_emotion = extract_recent_emotion(conversation)

    # rag
    # 기분과 대화에 따른 추천 (3개)(vector_db.py)

    selected = get_recommendation_by_emotion(recent_emotion, conversation, k=3)

    if not selected:
        return {
            "answer": f"{request.type} 추천 데이터가 없습니다.",
            "recommendation_data": {"error": "데이터 없음"}
        }
    
    recommendation_data = {
        "category": request.type,
        "current_emotion": recent_emotion,
        "recommended_emotion": recent_emotion,
        "recommendation": selected[0],
        "all_recommendations": selected,
    }

    # 4. 추천 정보 포맷팅
    # ex) 도서, recommendation_data
    formatted_rec = format_recommendation(request.type, recommendation_data)

    # 5. 캐릭터 말투로 추천 메시지 생성
    answer = generate_recommendation_response(
        character=request.character,
        category=request.type,
        recommendation_data=recommendation_data,
        formatted_recommendation=formatted_rec
    )

    return {
        "answer": answer,
        # 검색한 전체 3개 
        # 1개 -> selected[0]
        "recommendation_data": recommendation_data
    }


@app.post("/api/analyze-diary")
async def analyze_diary(request: DiaryRequest):
    """
    일기 분석 및 감정 기반 지능형 추천
    1. 일기에서 감정 추출
    2. 감정 벡터를 만들어 반대 감정 찾기
    3. 일기 내용과 가장 관련성 높은 콘텐츠를 의미 기반으로 추천
    """

    # 1. 감정 추출
    emotion = extract_emotion(request.diary)

    # 2. 감정 임베딩 생성 후 벡터 DB에서 반대 감정 찾기
    emotion_vector = get_embedding(emotion)
    if emotion_vector is None:
        return {"error": "감정 분석에 실패했습니다."}

    # opposite_emotion = find_dissimilar_emotion_key(emotion_vector)

    # 3. 의미 기반 스마트 추천
    from ai_core.recommendation import get_smart_recommendation

    # 일기 내용과 가장 관련성 높은 콘텐츠 추천
    selected_books = get_smart_recommendation(
        user_text=request.diary,
        emotion=emotion,
        category="도서",
        top_k=2
    )

    selected_music = get_smart_recommendation(
        user_text=request.diary,
        emotion=emotion,
        category="음악",
        top_k=2
    )

    selected_food = get_smart_recommendation(
        user_text=request.diary,
        emotion=emotion,
        category="식사",
        top_k=2
    )

    # 4. 감정에 따른 메시지 생성
    emotion_messages = {
        "행복": "오늘 정말 좋은 하루를 보내셨네요! 이 기분을 더 오래 간직할 수 있는 콘텐츠를 추천해드려요.",
        "슬픔": "힘든 하루였군요. 위로가 되는 콘텐츠로 마음을 다독여보세요.",
        "분노": "화가 많이 나셨나봐요. 스트레스를 해소할 수 있는 콘텐츠를 준비했어요.",
        "평온": "평온한 하루를 보내셨네요. 이 평화로움을 유지할 수 있는 콘텐츠예요.",
        "불안": "불안한 마음이 느껴지네요. 마음을 진정시킬 수 있는 콘텐츠를 추천드려요."
    }

    message = emotion_messages.get(emotion, "오늘 하루의 감정을 바탕으로 추천을 준비했어요.")

    return {
        "emotion": emotion,
        "emotion": emotion,
        "message": message,
        "books": selected_books,
        "music": selected_music,
        "food": selected_food
    }


@app.get("/")
def read_root():
    return {
        "message": "✅ ICSYF AI 통합 서버가 성공적으로 실행되었습니다!",
        "version": "2.0.0",
        "features": [
            "AI 챗봇 (감정 분석 + 캐릭터 대화)",
            "AI 추천 시스템 (도서, 음악, 식사)",
            "사용자 인증 (회원가입, 로그인)",
            "다이어리 관리 (작성, 조회, 수정, 삭제)",
            "AI 일기 분석 및 추천"
        ]
    }