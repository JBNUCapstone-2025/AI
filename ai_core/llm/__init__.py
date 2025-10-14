# ai_core/llm/__init__.py
"""
LLM (Large Language Model) 관련 기능
- 감정 분석
- 캐릭터 응답 생성
- 공감 응답 생성
"""

from .llm_utils import (
    extract_emotion,
    extract_recent_emotion,
    get_embedding,
    generate_character_response,
    generate_empathetic_response,
    generate_recommendation_response
)

__all__ = [
    'extract_emotion',
    'extract_recent_emotion',
    'get_embedding',
    'generate_character_response',
    'generate_empathetic_response',
    'generate_recommendation_response'
]
