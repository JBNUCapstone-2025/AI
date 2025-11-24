# rag_recommender.py
import random
import os
import sys
from typing import Dict, List, Any, Literal
from langchain_core.documents import Document
from ai_core.llm.llm_utils import embedding_model
import json

# 프로젝트 루트를 sys.path에 추가
# os.path.abspath(_file_) : 이 파일의 절대 경로를 구함.
# os.path.dirname() : 파일 경로에서 상위 폴더 경로만 추출
# 프로젝트의 루트 폴더 의미 -> os.path.dirname() 3번 썼기 때문
# /home/user/project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 파이썬이 모듈을 import할 때 검색하는 경로 목록
# BASE_DIR을 맨 앞(index 0)에 추가하면, python이 import할 때 프로젝트 루트부터 먼저 검색함.
sys.path.insert(0, BASE_DIR)

#from ai_core.llm.llm_utils import get_embedding
#from data.recommendation_data import get_recommendation_data
'''
def get_rag_recommendation(conversation_history: str, category: str) -> Dict:
    """
    대화 기록을 분석하여 RAG 기반으로 추천을 제공합니다.

    Args:
        conversation_history: 최근 대화 내용
        category: 추천 카테고리 (도서, 음악, 식사)

    Returns:
        추천 정보 딕셔너리
    """
    # 1. 대화에서 감정 추출 (대화 전체를 임베딩)
    emotion_vector = get_embedding(conversation_history)

    if emotion_vector is None:
        # 기본 감정으로 fallback
        opposite_emotion = "평온"
        current_emotion = "불안"
    else:
        # 2. 벡터 DB에서 반대 감정 찾기
        opposite_emotion = find_dissimilar_emotion_key(emotion_vector)
        # 현재 감정은 벡터 DB의 모든 감정 중 가장 유사한 것
        from vector_db import EMOTIONS, index
        import numpy as np
        import faiss

        query_vector = np.array([emotion_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, k=1)
        current_emotion = EMOTIONS[indices[0][0]]

    # 3. 반대 감정 기반으로 추천 데이터 가져오기
    recommendations = get_recommendation_data(opposite_emotion, category)

    if not recommendations:
        return {
            "error": f"{category} 추천 데이터가 없습니다.",
            "current_emotion": current_emotion,
            "recommended_emotion": opposite_emotion
        }

    # 4. 랜덤으로 하나 선택 (또는 여러 개)
    selected = random.choice(recommendations)

    return {
        "category": category,
        "current_emotion": current_emotion,
        "recommended_emotion": opposite_emotion,
        "recommendation": selected,
        "all_recommendations": recommendations[:3]  # 상위 3개 반환
    }
'''
def build_item_candidates(docs: List[Document], category: Literal["도서", "음악"],) -> List[Dict[str, Any]]:
    candidates = List[Dict[str, Any]] = []

    for doc in docs:
        payload = json.loads(doc.page_content)
        emotion = payload.get("emotion")
        emotion_kr = payload.get("emotion_kr")

        if category == "도서":
            for b in payload.get("books", []):
                candidates.append({
                    "type": "book",
                    "emotion": emotion,
                    "emotion_kr": emotion_kr,
                    "title": b.get("title", ""),
                    "author": b.get("author", ""),
                    "publisher": b.get("publisher", ""),
                    "subtitle": b.get("subtitle", ""),
                    "detail_url": b.get("detail_url", ""),
                    "cover_image_url": b.get("cover_image_url", ""),
                    "price": b.get("price", ""),
                    "tags": b.get("tags", []),
                })
        elif category == "음악":
            for m in payload.get("music", []):
                candidates.append({
                    "type": "music",
                    "emotion": emotion,
                    "emotion_kr": emotion_kr,
                    "title": m.get("title", ""),
                    "artist": m.get("artist", ""),
                    "album": m.get("album", ""),
                    "genre": m.get("genre", ""),
                    "detail_url": m.get("detail_url", ""),
                    "cover_image_url": m.get("cover_url", ""),
                    "dj_tags": m.get("dj_tags", []),
                })

    return candidates


# 도서 
def format_book_recommendation(data: Dict) -> str:
    """도서 추천 정보를 포맷팅합니다."""
    rec = data["recommendation"]
    title = rec.metadata.get("title", "")
    author = rec.metadata.get("author", "")
    description = rec.page_content if hasattr(rec, "page_content") else ""

    result = f"📚 {title}"
    if author:
        result += f"\n저자: {author}"
    if description:
        result += f"\n{description}"

    # 추가 추천도 포함
    if "all_recommendations" in data and len(data["all_recommendations"]) > 1:
        result += "\n\n다른 추천도서:"
        for book in data["all_recommendations"][1:]:
            b_title = book.metadata.get("title", "") if hasattr(book,"metadata") else ""
            b_author = book.metadata.get("author","") if hasattr(book, "metadata") else ""
            result += f"\n {b_title} - {b_author}"

    return result


def format_music_recommendation(data: Dict) -> str:
    """음악 추천 정보를 포맷팅합니다."""
    rec = data["recommendation"]
    title = rec.metadata.get("title", "")
    artist = rec.metadata.get("artist", "")
    album = rec.metadata.get("album", "")

    result = f"🎵 {title}"
    if artist:
        result += f"\n아티스트: {artist}"
    if album:
        result += f"\n{album}"

    # 추가 추천도 포함
    if "all_recommendations" in data and len(data["all_recommendations"]) > 1:
        result += "\n\n다른 추천곡:"
        for music in data["all_recommendations"][1:]:
            result += f"\n• {music.metadata.get('title', '')} - {music.metadata.get('artist', '')}"

    return result


def format_food_recommendation(data: Dict) -> str:
    """식사 추천 정보를 포맷팅합니다."""
    rec = data["recommendation"]
    name = rec.get("name", "")
    description = rec.get("description", "")
    category_type = rec.get("category", "")

    result = f"🍽️ {name}"
    if category_type:
        result += f" ({category_type})"
    if description:
        result += f"\n{description}"

    # 추가 추천도 포함
    if "all_recommendations" in data and len(data["all_recommendations"]) > 1:
        result += "\n\n다른 추천메뉴:"
        for food in data["all_recommendations"][1:]:
            result += f"\n• {food.get('name', '')} - {food.get('description', '')}"

    return result

# main.py 
# 도서, {}
def format_recommendation(category: str, data: Dict) -> str:
    """카테고리에 따라 추천 정보를 포맷팅합니다."""
    if "error" in data:
        return data["error"]

    formatters = {
        "도서": format_book_recommendation,
        "음악": format_music_recommendation,
        "식사": format_food_recommendation
    }

    # 도서 
    formatter = formatters.get(category)
    if formatter:
        return formatter(data)

    return str(data)
