from sqlalchemy.orm import Session
from app.db.models import User
from app.schemas.user import UserSignupRequest
from app.services.auth import get_password_hash


def get_user_by_username(db: Session, username: str):
    """username으로 유저 조회"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str):
    """email으로 유저 조회"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_nickname(db: Session, nick_name: str):
    """nickname으로 유저 조회"""
    return db.query(User).filter(User.nick_name == nick_name).first()


def create_user(db: Session, user_data: UserSignupRequest):
    """새로운 유저 생성"""
    hashed_password = get_password_hash(user_data.password)

    db_user = User(
        username=user_data.username,
        password=hashed_password,
        person_name=user_data.person_name,
        nick_name=user_data.nick_name,
        email=user_data.email,
        phone=user_data.phone
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user
