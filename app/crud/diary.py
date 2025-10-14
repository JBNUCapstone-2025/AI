from sqlalchemy.orm import Session
from sqlalchemy import extract
from app.db.models import Diary
from app.schemas.diary import DiaryCreateRequest, DiaryUpdateRequest
from datetime import date
from typing import List, Optional


def create_diary(db: Session, user_id: int, diary: DiaryCreateRequest) -> Diary:
    """일기 생성"""
    db_diary = Diary(
        user_id=user_id,
        title=diary.title,
        content=diary.content,
        diary_date=diary.diary_date
    )
    db.add(db_diary)
    db.commit()
    db.refresh(db_diary)
    return db_diary


def get_diary_by_id(db: Session, diary_id: int, user_id: int) -> Optional[Diary]:
    """일기 ID로 조회 (본인 일기만)"""
    return db.query(Diary).filter(
        Diary.diary_id == diary_id,
        Diary.user_id == user_id
    ).first()


def get_diary_by_date(db: Session, user_id: int, diary_date: date) -> Optional[Diary]:
    """특정 날짜의 일기 조회"""
    return db.query(Diary).filter(
        Diary.user_id == user_id,
        Diary.diary_date == diary_date
    ).first()


def get_diaries_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Diary]:
    """사용자의 모든 일기 조회 (최신순)"""
    return db.query(Diary).filter(
        Diary.user_id == user_id
    ).order_by(Diary.diary_date.desc()).offset(skip).limit(limit).all()


def get_diaries_by_month(db: Session, user_id: int, year: int, month: int) -> List[Diary]:
    """월별 일기 조회 (달력용)"""
    return db.query(Diary).filter(
        Diary.user_id == user_id,
        extract('year', Diary.diary_date) == year,
        extract('month', Diary.diary_date) == month
    ).order_by(Diary.diary_date.asc()).all()


def update_diary(db: Session, diary_id: int, user_id: int, diary_update: DiaryUpdateRequest) -> Optional[Diary]:
    """일기 수정"""
    db_diary = get_diary_by_id(db, diary_id, user_id)
    if not db_diary:
        return None

    # 수정할 필드만 업데이트
    if diary_update.title is not None:
        db_diary.title = diary_update.title
    if diary_update.content is not None:
        db_diary.content = diary_update.content
    if diary_update.diary_date is not None:
        db_diary.diary_date = diary_update.diary_date

    db.commit()
    db.refresh(db_diary)
    return db_diary


def delete_diary(db: Session, diary_id: int, user_id: int) -> bool:
    """일기 삭제"""
    db_diary = get_diary_by_id(db, diary_id, user_id)
    if not db_diary:
        return False

    db.delete(db_diary)
    db.commit()
    return True
