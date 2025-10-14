import pickle
import random
import numpy as np
import faiss
import os

# 데이터 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Faiss 인덱스와 감정 데이터 로드
try:
    index = faiss.read_index(os.path.join(DATA_DIR, "vector_db.faiss"))
    with open(os.path.join(DATA_DIR, "emotion_data.pkl"), "rb") as f:
        db_data = pickle.load(f)
    
    EMOTIONS = db_data["emotions"]
    EMOTION_DATA = db_data["data"]
    IS_DB_READY = True
    print("벡터 DB가 성공적으로 로드되었습니다.")
except FileNotFoundError:
    IS_DB_READY = False
    print("오류: vector_db.faiss 또는 emotion_data.pkl 파일을 찾을 수 없습니다.")
    print("먼저 populate_db.py 스크립트를 실행하여 DB를 생성해주세요.")


def find_dissimilar_emotion_key(vector: np.ndarray) -> str:
    """
    주어진 벡터와 코사인 유사도가 가장 낮은 감정 키를 찾습니다.
    Faiss의 IndexFlatIP는 내적(dot product)을 계산하므로, 정규화된 벡터들 사이에서는
    내적이 코사인 유사도와 같습니다. 따라서 가장 작은 값을 찾으면 됩니다.
    """
    if not IS_DB_READY:
        raise ConnectionError("벡터 DB가 준비되지 않았습니다.")

    # 입력 벡터 정규화
    query_vector = np.array([vector]).astype('float32')
    faiss.normalize_L2(query_vector)

    # 가장 낮은 유사도(가장 먼 거리)를 가진 벡터 1개를 검색
    # k=len(EMOTIONS)로 전체를 검색한 뒤, 첫번째(자기 자신)를 제외하고 선택할 수도 있음
    # 여기서는 가장 낮은 하나만 찾습니다.
    distances, indices = index.search(query_vector, k=1)
    
    # IndexFlatIP는 최대 내적을 찾으므로, k를 늘려 가장 낮은 값을 찾아야 합니다.
    # 모든 벡터와의 거리를 계산하고 가장 작은 값을 선택합니다.
    distances, indices = index.search(query_vector, k=len(EMOTIONS))

    # 검색 결과는 유사도가 높은 순으로 정렬되므로, 가장 마지막 인덱스가 유사도가 가장 낮은 벡터입니다.
    dissimilar_index = indices[0][-1]

    return EMOTIONS[dissimilar_index]

def get_random_content(emotion_key: str) -> dict:
    """주어진 감정 키에 해당하는 콘텐츠 중 하나를 무작위로 선택합니다."""
    if not IS_DB_READY:
        raise ConnectionError("벡터 DB가 준비되지 않았습니다.")
        
    content_pool = EMOTION_DATA.get(emotion_key, {})
    if not content_pool:
        return {"error": "추천할 콘텐츠가 없습니다."}

    content_type = random.choice(list(content_pool.keys()))
    content_item = random.choice(content_pool[content_type])
    
    return {content_type: content_item}
