# AI 폴더 구조 정리 완료

## 새로운 폴더 구조

```
ai/
├── main.py                    # FastAPI 메인 애플리케이션
├── requirements.txt           # Python 의존성
├── .env                       # 환경 변수
├── README.md                  # 프로젝트 문서
│
├── app/                       # Backend 기능
│   ├── api/                   # API 라우터
│   │   ├── auth.py           # 인증 API
│   │   └── diary.py          # 다이어리 API
│   ├── crud/                  # DB CRUD 작업
│   │   ├── user.py
│   │   └── diary.py
│   ├── db/                    # 데이터베이스
│   │   ├── database.py       # DB 연결
│   │   └── models.py         # SQLAlchemy 모델
│   ├── schemas/               # Pydantic 스키마
│   │   ├── user.py
│   │   └── diary.py
│   └── services/              # 비즈니스 로직
│       └── auth.py           # JWT 인증
│
├── ai_core/                   # AI 핵심 기능 (정리됨!)
│   ├── llm/                   # 언어 모델
│   │   ├── __init__.py
│   │   └── llm_utils.py      # 감정 분석, 응답 생성
│   ├── recommendation/        # 추천 시스템
│   │   ├── __init__.py
│   │   ├── content_recommender.py    # 스마트 추천
│   │   └── rag_recommender.py        # RAG 추천
│   └── vector_db/             # 벡터 데이터베이스
│       ├── __init__.py
│       └── vector_db.py      # 감정 벡터 검색
│
├── data/                      # 데이터 파일 (정리됨!)
│   ├── __init__.py
│   ├── recommendation_data.py  # 추천 데이터
│   ├── emotion_data.pkl        # 감정 벡터
│   └── vector_db.faiss         # FAISS 인덱스
│
├── scripts/                   # 유틸리티 스크립트 (정리됨!)
│   ├── __init__.py
│   ├── deom_data.py           # 데모 데이터 생성
│   └── mk_data_db.py          # DB 초기화
│
└── prompt/                    # 프롬프트 템플릿
    └── characters.py          # 캐릭터 정의
```

## 주요 변경사항

### 1. 기능별 분리
**이전**: 모든 파일이 루트에 섞여있음
```
❌ llm_utils.py, vector_db.py, content_recommender.py, rag_recommender.py (루트)
```

**변경 후**: 기능별 폴더로 명확히 분리
```
✅ ai_core/llm/llm_utils.py
✅ ai_core/vector_db/vector_db.py
✅ ai_core/recommendation/content_recommender.py
✅ ai_core/recommendation/rag_recommender.py
```

### 2. 데이터 파일 분리
**이전**: 데이터 파일이 코드와 섞여있음
```
❌ recommendation_data.py, emotion_data.pkl, vector_db.faiss (루트)
```

**변경 후**: 데이터 전용 폴더로 분리
```
✅ data/recommendation_data.py
✅ data/emotion_data.pkl
✅ data/vector_db.faiss
```

### 3. 스크립트 분리
**이전**: 유틸리티 스크립트가 루트에 있음
```
❌ deom_data.py, mk_data_db.py (루트)
```

**변경 후**: 스크립트 전용 폴더로 분리
```
✅ scripts/deom_data.py
✅ scripts/mk_data_db.py
```

### 4. 모듈 Import 개선

**이전**:
```python
from llm_utils import extract_emotion
from vector_db import find_dissimilar_emotion_key
from content_recommender import get_smart_recommendation
```

**변경 후**:
```python
from ai_core.llm import extract_emotion
from ai_core.vector_db import find_dissimilar_emotion_key
from ai_core.recommendation import get_smart_recommendation
```

## 장점

### 1. 명확한 구조
- 각 모듈의 역할이 폴더 이름으로 명확히 표현됨
- 새로운 개발자도 쉽게 이해 가능

### 2. 유지보수 용이
- 관련 파일들이 한 곳에 모여있어 수정 편리
- 의존성 관리가 명확해짐

### 3. 확장성
- 새로운 AI 기능 추가 시 `ai_core/` 내에 폴더 추가
- 새로운 데이터 타입 추가 시 `data/` 내에 파일 추가

### 4. 테스트 용이
- 각 모듈을 독립적으로 테스트 가능
- Import 경로가 명확하여 mocking 쉬움

## 사용 예시

### main.py에서 AI 기능 사용
```python
from ai_core.llm import extract_emotion, generate_empathetic_response
from ai_core.vector_db import find_dissimilar_emotion_key
from ai_core.recommendation import get_smart_recommendation

# 감정 추출
emotion = extract_emotion("오늘 기분이 좋아요!")

# 벡터 DB에서 반대 감정 찾기
opposite = find_dissimilar_emotion_key(emotion_vector)

# 스마트 추천
recommendations = get_smart_recommendation(
    user_text="행복한 하루",
    emotion=opposite,
    category="도서"
)
```

### app/api/diary.py에서 AI 기능 사용
```python
from ai_core.llm import extract_emotion, get_embedding
from ai_core.vector_db import find_dissimilar_emotion_key
from ai_core.recommendation import get_smart_recommendation

# 일기 작성 시 자동으로 AI 분석
emotion = extract_emotion(diary.content)
recommendations = get_smart_recommendation(diary.content, emotion, "도서")
```

## 주의사항

1. **경로 참조**: 모든 데이터 파일 경로가 `data/` 폴더 기준으로 수정됨
2. **Import 경로**: 기존 코드에서 import 경로 전부 수정 완료
3. **서버 재시작**: 변경사항 적용을 위해 서버 재시작 필요

## 확인 방법

서버가 정상 작동하는지 확인:
```bash
curl http://localhost:8000/
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"sentence": "안녕하세요!", "character": "강아지"}'
```

## 정리 완료 체크리스트

- ✅ ai_core 폴더 생성 및 파일 이동
- ✅ data 폴더 생성 및 파일 이동
- ✅ scripts 폴더 생성 및 파일 이동
- ✅ 모든 __init__.py 파일 생성
- ✅ import 경로 전부 수정
- ✅ 서버 정상 작동 확인
- ✅ API 테스트 통과
