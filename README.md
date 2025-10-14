# ICSYF AI 통합 서버

감정 기반 정서 관리 플랫폼의 AI 및 Backend 통합 서버입니다.

## 주요 기능

### 1. AI 기능
- **챗봇**: 감정 분석 기반 캐릭터 대화 (강아지, 고양이, 토끼)
- **추천 시스템**: RAG 기반 지능형 콘텐츠 추천 (도서, 음악, 식사)
- **일기 분석**: 감정 추출 및 맞춤형 콘텐츠 추천

### 2. Backend 기능
- **사용자 인증**: 회원가입, 로그인 (JWT 토큰)
- **다이어리 관리**: CRUD 작업 + AI 자동 분석
- **데이터베이스**: MySQL 연동

## 프로젝트 구조

```
ai/
├── app/
│   ├── api/          # API 라우터
│   │   ├── auth.py   # 인증 API
│   │   └── diary.py  # 다이어리 API (AI 통합)
│   ├── crud/         # 데이터베이스 작업
│   │   ├── user.py
│   │   └── diary.py
│   ├── db/           # 데이터베이스 설정
│   │   ├── database.py
│   │   └── models.py
│   ├── schemas/      # Pydantic 스키마
│   │   ├── user.py
│   │   └── diary.py
│   └── services/     # 비즈니스 로직
│       └── auth.py
├── main.py           # FastAPI 애플리케이션
├── llm_utils.py      # AI 유틸리티
├── vector_db.py      # 벡터 DB
├── content_recommender.py  # 콘텐츠 추천
└── requirements.txt  # 의존성
```

## 설치 및 실행

### 1. 가상환경 활성화
```bash
source cahtbot/bin/activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정
`.env` 파일을 생성하고 다음 내용을 설정:
```env
# Database
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=capstone
DB_USER=root
DB_PASSWORD=your_password

# JWT
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=10080

# OpenAI
OPENAI_API_KEY=your-openai-api-key
```

### 4. 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 실행되면 다음 URL에서 접근 가능:
- API 문서: http://localhost:8000/docs
- 서버 상태: http://localhost:8000/

## API 엔드포인트

### 인증 API
- `POST /auth/signup` - 회원가입
- `POST /auth/login` - 로그인

### 다이어리 API
- `POST /diary/` - 일기 작성 (AI 자동 분석 포함)
- `GET /diary/list` - 일기 목록 조회
- `GET /diary/{diary_id}` - 일기 상세 조회
- `GET /diary/by-date/{date}` - 날짜별 일기 조회
- `GET /diary/calendar/{year}/{month}` - 월별 일기 조회
- `PUT /diary/{diary_id}` - 일기 수정
- `DELETE /diary/{diary_id}` - 일기 삭제

### AI API
- `POST /api/chat` - AI 챗봇 대화
- `POST /api/recommend` - AI 콘텐츠 추천
- `POST /api/analyze-diary` - 일기 감정 분석

## 사용 예시

### 1. 회원가입
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123",
    "person_name": "홍길동",
    "nick_name": "길동이",
    "email": "test@example.com",
    "phone": "010-1234-5678"
  }'
```

### 2. 로그인
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123"
  }'
```

### 3. 일기 작성 (AI 자동 분석)
```bash
curl -X POST "http://localhost:8000/diary/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "title": "좋은 하루",
    "content": "오늘은 정말 기분 좋은 하루였다.",
    "diary_date": "2025-10-15"
  }'
```

### 4. AI 챗봇 대화
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "오늘 너무 기분이 좋아!",
    "character": "강아지"
  }'
```

## 주요 변경사항

### Backend 통합
- Backend의 모든 기능을 AI 서버로 병합
- `/app` 디렉토리에 Backend 코드 구조 통합
- 단일 서버에서 모든 기능 제공

### AI 기능 강화
- 일기 작성 시 자동으로 AI 감정 분석 및 추천 실행
- 감정 벡터 DB를 활용한 반대 감정 기반 추천
- 의미 기반 스마트 추천 시스템

## 주의사항

1. **포트 충돌**: 기존 Backend 서버(8000번 포트)가 실행 중이면 종료해야 함
2. **데이터베이스**: MySQL 서버가 실행 중이어야 함
3. **OpenAI API**: 환경변수에 유효한 API 키 설정 필요

## 프론트엔드 연동

프론트엔드는 이제 단일 서버(AI 통합 서버)만 호출하면 됩니다:
- 기존: Frontend → Backend → (X)
- 변경: Frontend → AI 통합 서버 (포트 8000)

모든 API 엔드포인트는 `http://localhost:8000`에서 제공됩니다.
