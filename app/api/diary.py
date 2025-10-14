from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
from app.db.database import get_db
from app.schemas.diary import (
    DiaryCreateRequest, DiaryCreateResponse, DiaryResponse,
    DiaryUpdateRequest, DiaryListResponse, DiaryCalendarResponse
)
from app.crud.diary import (
    create_diary, get_diary_by_id, get_diary_by_date,
    get_diaries_by_user, get_diaries_by_month,
    update_diary, delete_diary
)
from app.services.auth import verify_token

# AI 기능 import
from ai_core.llm import extract_emotion, get_embedding
from ai_core.vector_db import find_dissimilar_emotion_key
from ai_core.recommendation import get_smart_recommendation

router = APIRouter(prefix="/diary", tags=["Diary"])
security = HTTPBearer()


def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
    """JWT 토큰에서 현재 사용자 ID 추출"""
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 토큰입니다"
        )
    user_id = payload.get("userId")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰 페이로드가 유효하지 않습니다"
        )
    return user_id


@router.post("/", response_model=DiaryCreateResponse, status_code=status.HTTP_201_CREATED)
def create_diary_endpoint(
    diary: DiaryCreateRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """일기 작성 + AI 감정 분석 및 추천"""
    # 같은 날짜에 이미 일기가 있는지 확인
    existing_diary = get_diary_by_date(db, user_id, diary.diary_date)
    if existing_diary:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{diary.diary_date} 날짜에 이미 일기가 존재합니다"
        )

    # 일기 생성
    new_diary = create_diary(db, user_id, diary)

    # AI 감정 분석
    try:
        emotion = extract_emotion(diary.content)
        emotion_vector = get_embedding(emotion)

        if emotion_vector is not None:
            opposite_emotion = find_dissimilar_emotion_key(emotion_vector)

            # 스마트 추천
            books = get_smart_recommendation(
                user_text=diary.content,
                emotion=opposite_emotion,
                category="도서",
                top_k=2
            )

            music = get_smart_recommendation(
                user_text=diary.content,
                emotion=opposite_emotion,
                category="음악",
                top_k=2
            )

            food = get_smart_recommendation(
                user_text=diary.content,
                emotion=opposite_emotion,
                category="식사",
                top_k=2
            )

            # DB 업데이트
            new_diary.emotion = {emotion: 1.0}
            new_diary.recommend_content = {
                "books": books,
                "music": music,
                "food": food
            }
            db.commit()
            db.refresh(new_diary)
    except Exception as e:
        print(f"AI 분석 에러: {e}")
        # AI 분석 실패해도 일기는 저장됨

    return {"message": "일기가 작성되었습니다", "diary": new_diary}


@router.get("/list", response_model=List[DiaryListResponse])
def get_diary_list(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """일기 목록 조회 (최신순)"""
    # 파라미터 검증
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="skip은 0 이상이어야 합니다"
        )
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit은 1에서 1000 사이여야 합니다"
        )

    diaries = get_diaries_by_user(db, user_id, skip, limit)
    return diaries


@router.get("/calendar/{year}/{month}", response_model=DiaryCalendarResponse)
def get_diary_calendar(
    year: int,
    month: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """월별 일기 조회 (달력용)"""
    # 년도 검증
    if year < 1900 or year > 2100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="년도는 1900에서 2100 사이여야 합니다"
        )
    # 월 검증
    if month < 1 or month > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="월은 1에서 12 사이여야 합니다"
        )

    diaries = get_diaries_by_month(db, user_id, year, month)
    return {"year": year, "month": month, "diaries": diaries}


@router.get("/by-date/{diary_date}", response_model=DiaryResponse)
def get_diary_by_date_endpoint(
    diary_date: str,  # YYYY-MM-DD 형식
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """특정 날짜의 일기 조회"""
    try:
        date_obj = datetime.strptime(diary_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요"
        )

    diary = get_diary_by_date(db, user_id, date_obj)
    if not diary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{diary_date} 날짜의 일기를 찾을 수 없습니다"
        )
    return diary


@router.get("/{diary_id}", response_model=DiaryResponse)
def get_diary_detail(
    diary_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """일기 상세 조회"""
    diary = get_diary_by_id(db, diary_id, user_id)
    if not diary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="일기를 찾을 수 없습니다"
        )
    return diary


@router.put("/{diary_id}", response_model=DiaryResponse)
def update_diary_endpoint(
    diary_id: int,
    diary_update: DiaryUpdateRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """일기 수정"""
    # 날짜를 변경하는 경우, 해당 날짜에 이미 다른 일기가 있는지 확인
    if diary_update.diary_date is not None:
        existing_diary = get_diary_by_date(db, user_id, diary_update.diary_date)
        # 같은 날짜에 다른 일기가 있고, 그게 현재 수정하려는 일기가 아닌 경우
        if existing_diary and existing_diary.diary_id != diary_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{diary_update.diary_date} 날짜에 이미 일기가 존재합니다"
            )

    updated_diary = update_diary(db, diary_id, user_id, diary_update)
    if not updated_diary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="일기를 찾을 수 없습니다"
        )
    return updated_diary


@router.delete("/{diary_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_diary_endpoint(
    diary_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """일기 삭제"""
    success = delete_diary(db, diary_id, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="일기를 찾을 수 없습니다"
        )
    return None
