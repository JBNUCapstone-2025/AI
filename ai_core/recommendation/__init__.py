# ai_core/recommendation/__init__.py
"""
추천 시스템 모듈
- 스마트 콘텐츠 추천
- RAG 기반 추천
"""

from .content_recommender import get_smart_recommendation
from .rag_recommender import get_rag_recommendation, format_recommendation

__all__ = [
    'get_smart_recommendation',
    'get_rag_recommendation',
    'format_recommendation'
]
