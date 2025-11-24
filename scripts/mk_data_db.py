import os
import json 
from langchain_core.documents import Document
from langchain_chroma import Chroma
from ai_core.llm.llm_utils import embedding_model
from typing import List, Dict, Any
import shutil

# 벡터db 생성 및 저장


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

JSON_DIR = os.path.join(BASE_DIR, "data2")
VECTORDB_DIR = os.path.join(BASE_DIR, "ai_core", "vector_db", "chroma_vectordb")

def load_book_docs_from_dir(directory: str) -> list[Document]:
    docs : List[Document] = [] 

    # 디렉토리 안의 모든 파일 확인
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
    
            full_path = os.path.join(directory, filename)

            # 파일명에서 emotion_group 추출 (예: anger.json → anger)
            

            with open(full_path, "r", encoding="utf-8") as f:
                payload: Dict[str, Any] = json.load(f)


            emotion = payload.get("emotion")
            emotion_kr = payload.get("emotion_kr")

            for b in payload.get("books", []):
                page_content = f"""
                [BOOK]

                감정: {emotion_kr} ({emotion})

                제목: {b.get('title', '')}
                부제: {b.get('subtitle', '')}
                저자: {b.get('author', '')}
                출판사: {b.get('publisher', '')}
                가격: {b.get('price', '')}
                """.strip()

                metadata = {
                    "emotion": emotion,
                    "emotion_kr": emotion_kr,
                    "category": "도서",  
                    "title": b.get("title", ""),
                    "author": b.get("author", ""),
                    "publisher": b.get("publisher", ""),
                    "detail_url": b.get("detail_url", ""),
                    "cover_image_url": b.get("cover_image_url", ""),
                }

                docs.append(Document(page_content=page_content, metadata=metadata))


            for m in payload.get("musics", []):
                page_content = f"""
                [MUSIC]

                감정: {emotion_kr} ({emotion})

                제목: {m.get('title', '')}
                아티스트: {m.get('artist', '')}
                앨범: {m.get('album', '')}
                장르: {m.get('genre', '')}
                """.strip()

                metadata = {
                    "emotion": emotion,
                    "emotion_kr": emotion_kr,
                    "category": "음악",
                    "title": m.get("title", ""),
                    "artist": m.get("artist", ""),
                    "album": m.get("album", ""),
                    "detail_url": m.get("detail_url", ""),
                    "cover_image_url": m.get("cover_url", ""),
                }
                
                docs.append(Document(page_content = page_content, metadata = metadata))

    return docs

        

def build_vectordb(docs):
    
    # shutil.rmtree(VECTORDB_DIR)
    os.makedirs(VECTORDB_DIR, exist_ok=True)

    # VectorDB 생성
    vectordb = Chroma(embedding_function = embedding_model, persist_directory = VECTORDB_DIR)
    vectordb.add_documents(docs)

    print(f"Chroma VectorDB 생성 완료: {VECTORDB_DIR}")
    print(f"저장된 문서 수 : ", len(docs))

    return vectordb

if __name__ == "__main__":
    # json 파일을 document 형식으로 변환 
    # # [Document(metadata={'product_id', 'title','detail_url', 'tags': [], 'emotion_group'}, page_content='제목, 저자, 출판사, 키워드/태그,출간일,가격, 상세보기)]
    docs = load_book_docs_from_dir(JSON_DIR) 

    print("📌 DEBUG: docs count =", len(docs))
    if len(docs) > 0:
        print("📌 DEBUG: sample doc =", docs[0])
    vectordb = build_vectordb(docs)

# ===============================================
# test_docs = vectordb.similarity_search("너무 화가나고 기분이 안좋아. 왜케 나를 화나게 하는걸까?", k=1)
# print("TEST 결과 : ", test_docs)