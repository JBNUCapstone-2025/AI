# content_recommender.py
# 의미 기반 콘텐츠 추천 시스템

import numpy as np
from typing import List, Dict
from llm_utils import get_embedding
from recommendation_data import get_recommendation_data


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_content_embedding(content: Dict, category: str) -> str:
    """
    콘텐츠 정보를 텍스트로 변환합니다.
    """
    if category == "도서":
        title = content.get("title", "")
        author = content.get("author", "")
        description = content.get("description", "")
        return f"{title} {author} {description}"

    elif category == "음악":
        title = content.get("title", "")
        artist = content.get("artist", "")
        description = content.get("description", "")
        return f"{title} {artist} {description}"

    elif category == "식사":
        name = content.get("name", "")
        description = content.get("description", "")
        category_type = content.get("category", "")
        return f"{name} {description} {category_type}"

    return ""


def rank_contents_by_similarity(
    user_text: str,
    contents: List[Dict],
    category: str,
    top_k: int = 3
) -> List[Dict]:
    """
    사용자 입력과 콘텐츠 간의 의미적 유사도를 계산하여 상위 K개를 반환합니다.
    """
    if not contents or not user_text:
        return contents[:top_k]

    # 사용자 텍스트 임베딩
    user_embedding = get_embedding(user_text)
    if user_embedding is None:
        # 임베딩 실패 시 첫 K개 반환
        return contents[:top_k]

    user_vec = np.array(user_embedding)

    # 각 콘텐츠의 유사도 계산
    scored_contents = []

    for content in contents:
        content_text = get_content_embedding(content, category)
        if not content_text:
            scored_contents.append((content, 0.0))
            continue

        content_embedding = get_embedding(content_text)
        if content_embedding is None:
            scored_contents.append((content, 0.0))
            continue

        content_vec = np.array(content_embedding)
        similarity = calculate_cosine_similarity(user_vec, content_vec)
        scored_contents.append((content, similarity))

    # 유사도 기준 내림차순 정렬
    scored_contents.sort(key=lambda x: x[1], reverse=True)

    # 상위 K개 반환
    return [content for content, score in scored_contents[:top_k]]


def get_smart_recommendation(
    user_text: str,
    emotion: str,
    category: str,
    top_k: int = 3
) -> List[Dict]:
    """
    감정 기반으로 콘텐츠를 필터링한 후,
    사용자 입력과 가장 관련성 높은 콘텐츠를 추천합니다.
    """
    # 1. 감정에 맞는 콘텐츠 풀 가져오기
    contents = get_recommendation_data(emotion, category)

    if not contents:
        return []

    # 2. 사용자 입력과의 유사도 기반으로 랭킹
    ranked_contents = rank_contents_by_similarity(
        user_text=user_text,
        contents=contents,
        category=category,
        top_k=top_k
    )

    return ranked_contents
