import pickle
import random
import numpy as np
import os
import faiss
from langchain_chroma import Chroma
from ai_core.llm.llm_utils import embedding_model
from typing import List, Dict, Any
from langchain_core.documents import Document

# 데이터 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORDB_DIR = os.path.join(BASE_DIR, "ai_core", "vector_db", "chroma_vectordb")

# 기쁨, 설렘, 보통, 슬픔, 불안, 분노


# 랭체인 기반 벡터db (Chroma) 
try:
    vectordb = Chroma(
        persist_directory = VECTORDB_DIR,
        embedding_function = embedding_model,
    )

    print("VECTORDB_DIR:", VECTORDB_DIR)
    print("FILES:", os.listdir(VECTORDB_DIR))
    print("INDEX EXISTS:", os.path.exists(os.path.join(VECTORDB_DIR, "index")))

    IS_DB_READY = True

    print("Chroma 벡터 DB가 성공적으로 로드되었습니다.")

    try:
        total = vectordb._collection.count()
        print(f"📌 DEBUG: Chroma 컬렉션 문서 개수 = {total}")
        sample = vectordb.similarity_search("테스트", k=1)
        print("📌 DEBUG: 샘플 검색 결과 =", sample)
    except Exception as inner_e:
        print("📌 DEBUG: vectordb 상태 확인 중 에러:", inner_e)

    with open(os.path.join(DATA_DIR, "emotion_data.pkl"), "rb") as f:
        db_data = pickle.load(f)

    EMOTIONS = db_data["emotions"]
    EMOTION_DATA = db_data["data"]
 
except Exception as e:
    vectordb = None
    IS_DB_READY = False
    print("오류: Chroma 벡터db를 찾을 수 없습니다. 먼저 mk_data_db.py를 실행하여 벡터DB를 생성해주세요.")
    print(f"상세 오류: {e}")


EMOTION_GROUP: Dict[str, str] = {
    "기쁨": "joy",
    "설렘": "excitement",
    "보통": "normal",
    "슬픔": "sadness",
    "분노": "anger",
    "불안": "anxiety",
}


# 검색 함수
# 감정, 대화내용에 대한 관련 문서 추천
# (main.py) recent_emotion, conversation, k=3
def get_recommendation_by_emotion(emotion_query: str, conversation: str = "", k: int = 3)-> List[Document]:
    
    if not IS_DB_READY:
        raise RuntimeError("벡터DB가 준비되지 않았습니다.")

    emotion_group = EMOTION_GROUP.get(emotion_query, emotion_query)

    # 검색 
    if conversation.strip():
        query = conversation
        print("query : ",query)

        docs : List[Document] = vectordb.similarity_search(query, k=10, filter={"emotion_group":emotion_group})
    else:
        query = emotion_group
        print("query : ", query)

        docs : List[Document] = vectordb.similarity_search(query, k=10, filter={"emotion_group":emotion_group})
        docs = random.shuffle(docs)


    # query = conversation if conversation.strip() else emotion_group
    # print("query : ", query)

    # query와 유사한 문서 찾기 (rag)
    # docs : List[Document] = vectordb.similarity_search(query, k=10, filter={"emotion_group":emotion_group})

    print(f"📌 DEBUG: emotion_query={emotion_query}, emotion_group={emotion_group}, 결과 개수={len(docs)}")

    # random.shuffle(docs)
    
    '''
    filtered = []
    for i in docs:
            if i.metadata.get("emotion_group") == emotion_query:
                filtered.append(i)
            else:
                filtered = docs
    
    random.shuffle(filtered)
    '''

    return docs




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
    distances, indices = vectordb.search(query_vector, k=1)
    
    # IndexFlatIP는 최대 내적을 찾으므로, k를 늘려 가장 낮은 값을 찾아야 합니다.
    # 모든 벡터와의 거리를 계산하고 가장 작은 값을 선택합니다.
    distances, indices = vectordb.search(query_vector, k=len(EMOTIONS))

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
