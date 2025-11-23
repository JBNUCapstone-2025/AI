import os
import json 
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from ai_core.llm.llm_utils import embedding_model

# 벡터db 생성 및 저장


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

JSON_DIR = os.path.join(BASE_DIR, "data2")
VECTORDB_DIR = os.path.join(BASE_DIR, "ai_core", "vector_db", "emotion_vectordb")

def load_book_docs_from_dir(directory: str) -> list[Document]:
    docs = []

    # 디렉토리 안의 모든 파일 확인
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            full_path = os.path.join(directory, filename)

            # 파일명에서 emotion_group 추출 (예: anger.json → anger)
            emotion_group = os.path.splitext(filename)[0]

            with open(full_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            for item in raw:
                tags = ", ".join(item.get("tags", []))

                page_content = f"""
                제목: {item["title"]}
                부제: {item.get("subtitle", "")}
                저자: {item["author"]}
                출판사: {item["publisher"]}
                키워드/태그: {tags}
                출간일: {item["pub_date"]}
                가격: {item["price"]}
                상세보기: {item["detail_url"]}
                """.strip()

                docs.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "product_id": item["product_id"],
                            "title": item["title"],
                            "author": item["author"],
                            "publisher": item["publisher"],
                            "subtitle": item.get("subtitle", ""),
                            "detail_url": item["detail_url"],
                            "tags": item.get("tags", []),
                            "emotion_group": emotion_group,  # ⭐ 추가된 부분
                        },
                    )
                )

    return docs

def build_vectordb(docs):

    # VectorDB 생성
    vectordb = FAISS.from_documents(docs, embedding_model)
    vectordb.save_local(VECTORDB_DIR)
    print(f"VectorDB 생성 완료: {VECTORDB_DIR}")
    
    return vectordb

if __name__ == "__main__":
    # json 파일을 document 형식으로 변환 
    # # [Document(metadata={'product_id', 'title','detail_url', 'tags': [], 'emotion_group'}, page_content='제목, 저자, 출판사, 키워드/태그,출간일,가격, 상세보기)]
    docs = load_book_docs_from_dir(JSON_DIR) 

    print("📌 DEBUG: docs count =", len(docs))
    if len(docs) > 0:
        print("📌 DEBUG: sample doc =", docs[0])
    vectordb = build_vectordb(docs)

    