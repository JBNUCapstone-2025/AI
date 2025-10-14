# rag_recommender.py
import random
import os
import sys
from typing import Dict, List

# 프로젝트 루트를 sys.path에 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ai_core.llm.llm_utils import get_embedding
from ai_core.vector_db.vector_db import find_dissimilar_emotion_key
from data.recommendation_data import get_recommendation_data

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


def format_book_recommendation(data: Dict) -> str:
    """도서 추천 정보를 포맷팅합니다."""
    rec = data["recommendation"]
    title = rec.get("title", "")
    author = rec.get("author", "")
    description = rec.get("description", "")

    result = f"📚 {title}"
    if author:
        result += f"\n저자: {author}"
    if description:
        result += f"\n{description}"

    # 추가 추천도 포함
    if "all_recommendations" in data and len(data["all_recommendations"]) > 1:
        result += "\n\n다른 추천도서:"
        for book in data["all_recommendations"][1:]:
            result += f"\n• {book.get('title', '')} - {book.get('author', '')}"

    return result


def format_music_recommendation(data: Dict) -> str:
    """음악 추천 정보를 포맷팅합니다."""
    rec = data["recommendation"]
    title = rec.get("title", "")
    artist = rec.get("artist", "")
    description = rec.get("description", "")

    result = f"🎵 {title}"
    if artist:
        result += f"\n아티스트: {artist}"
    if description:
        result += f"\n{description}"

    # 추가 추천도 포함
    if "all_recommendations" in data and len(data["all_recommendations"]) > 1:
        result += "\n\n다른 추천곡:"
        for music in data["all_recommendations"][1:]:
            result += f"\n• {music.get('title', '')} - {music.get('artist', '')}"

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


def format_recommendation(category: str, data: Dict) -> str:
    """카테고리에 따라 추천 정보를 포맷팅합니다."""
    if "error" in data:
        return data["error"]

    formatters = {
        "도서": format_book_recommendation,
        "음악": format_music_recommendation,
        "식사": format_food_recommendation
    }

    formatter = formatters.get(category)
    if formatter:
        return formatter(data)

    return str(data)
