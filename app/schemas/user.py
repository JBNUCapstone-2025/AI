from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, Dict
from datetime import datetime
from enum import Enum


# 권한 Enum
class RoleEnum(str, Enum):
    admin = "admin"
    member = "member"


# 회원가입 요청 스키마
class UserSignupRequest(BaseModel):
    username: str = Field(..., min_length=4, max_length=50, description="로그인 ID")
    password: str = Field(..., min_length=8, description="로그인 비밀번호")
    person_name: str = Field(..., min_length=2, max_length=50, description="사람 이름")
    nick_name: str = Field(..., min_length=2, max_length=50, description="닉네임")
    email: EmailStr = Field(..., description="이메일")
    phone: str = Field(..., pattern=r'^010-\d{4}-\d{4}$', description="휴대폰 번호 (010-1234-5678)")

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('비밀번호는 최소 8자 이상이어야 합니다')
        return v


# 로그인 요청 스키마
class UserLoginRequest(BaseModel):
    username: str = Field(..., description="로그인 ID")
    password: str = Field(..., description="로그인 비밀번호")


# 로그인 응답 스키마
class UserLoginResponse(BaseModel):
    access_token: str = Field(..., description="JWT 액세스 토큰")
    token_type: str = Field(default="bearer", description="토큰 타입")
    user: "UserResponse"


# 유저 정보 응답 스키마 (비밀번호 제외)
class UserResponse(BaseModel):
    user_id: int
    username: str
    person_name: str
    nick_name: str
    email: str
    phone: str
    create_date: datetime
    update_date: datetime
    emotion: Optional[Dict] = None
    role: RoleEnum

    class Config:
        from_attributes = True  # SQLAlchemy 모델을 Pydantic 모델로 변환


# 회원가입 응답 스키마
class UserSignupResponse(BaseModel):
    message: str = Field(default="회원가입이 완료되었습니다")
    user: UserResponse
