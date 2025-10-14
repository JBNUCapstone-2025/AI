from sqlalchemy import Column, Integer, String, DateTime, Enum, JSON, Text, Date, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base
import enum

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    MEMBER = "member"

class User(Base):
    __tablename__ = "User"

    # 기본 키
    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # 로그인 정보
    username = Column(String(50), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)  # bcrypt 해싱된 비밀번호

    # 개인 정보
    person_name = Column(String(50), nullable=False)
    nick_name = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    phone = Column(String(20), nullable=False)

    # 타임스탬프
    create_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    update_date = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 감정 데이터 (월별 결산용 JSON 형식)
    emotion = Column(JSON, nullable=True)

    # 권한
    role = Column(Enum(UserRole), default=UserRole.MEMBER, nullable=False)

    # 관계 설정
    diaries = relationship("Diary", back_populates="user", cascade="all, delete-orphan")


class Diary(Base):
    __tablename__ = "Diary"

    # 기본 키
    diary_id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # 외래 키
    user_id = Column(Integer, ForeignKey("User.user_id"), nullable=False, index=True)

    # 일기 내용
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)

    # AI 분석 결과 (AI API 연동 전까지 null)
    emotion = Column(JSON, nullable=True)  # {"기쁨": 0.8, "설렘": 0.3}
    recommend_content = Column(JSON, nullable=True)  # {"책": [...], "음악": [...]}

    # 일기 날짜 (달력 표시용)
    diary_date = Column(Date, nullable=False, index=True)

    # 타임스탬프
    create_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    update_date = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 관계 설정
    user = relationship("User", back_populates="diaries")
