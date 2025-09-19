import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

# API 키 설정
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 벡터화할 감정 키와 추천 콘텐츠 데이터
EMOTION_DATA = {
    "행복": {
        "movies": ["이터널 선샤인", "어바웃 타임", "라라랜드"],
        "music": ["Pharrell Williams - Happy", "아이유 - Blueming"],
        "books": ["어린 왕자", "데일 카네기 인간관계론"]
    },
    "슬픔": {
        "movies": ["조제, 호랑이 그리고 물고기들", "캐롤", "맨체스터 바이 더 씨"],
        "music": ["Adele - Someone Like You", "김광석 - 서른 즈음에"],
        "books": ["상실의 시대", "1984"]
    },
    "분노": {
        "movies": ["달콤한 인생", "존 윅", "악마를 보았다"],
        "music": ["Rage Against The Machine - Killing In The Name", "에픽하이 - Born Hater"],
        "books": ["정의란 무엇인가", "총, 균, 쇠"]
    },
    "평온": {
        "movies": ["리틀 포레스트", "패터슨", "원스"],
        "music": ["Debussy - Clair de Lune", "Norah Jones - Don't Know Why"],
        "books": ["월든", "느리게 사는 즐거움"]
    },
    "불안": {
        "movies": ["버드맨", "블랙 스완", "더 파더"],
        "music": ["Radiohead - Creep", "검정치마 - Antifreeze"],
        "books": ["변신", "인간 실격"]
    }
}

def get_embedding(text):
    """텍스트를 임베딩 벡터로 변환합니다."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None

def create_vector_db():
    """감정 데이터를 기반으로 Faiss 벡터 DB를 생성하고 저장합니다."""
    emotions = list(EMOTION_DATA.keys())
    vectors = []
    valid_emotions = []

    print("감정 키를 벡터화하는 중...")
    for emotion in emotions:
        embedding = get_embedding(emotion)
        if embedding:
            vectors.append(embedding)
            valid_emotions.append(emotion)
        else:
            print(f"'{emotion}'을(를) 벡터화하지 못했습니다.")

    if not vectors:
        print("벡터화할 데이터가 없습니다. API 키 또는 네트워크를 확인하세요.")
        return

    dimension = len(vectors[0])
    vector_matrix = np.array(vectors).astype('float32')

    # Faiss 인덱스 생성 (L2 정규화 및 IndexFlatIP 사용)
    faiss.normalize_L2(vector_matrix)
    index = faiss.IndexFlatIP(dimension)
    index.add(vector_matrix)

    print(f"Faiss 인덱스 생성 완료. {index.ntotal}개의 벡터가 추가되었습니다.")

    # 인덱스 및 데이터 저장
    faiss.write_index(index, "vector_db.faiss")
    with open("emotion_data.pkl", "wb") as f:
        pickle.dump({
            "emotions": valid_emotions,
            "data": EMOTION_DATA
        }, f)

    print("벡터 DB 파일(vector_db.faiss) 및 데이터 파일(emotion_data.pkl)이 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    create_vector_db()
