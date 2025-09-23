# llm_utils.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# API 키 설정
load_dotenv()
genai.configure(api_key="에이피아이키 하드코딩")

# LLM 모델 설정
llm = genai.GenerativeModel('gemini-1.5-flash')

def get_embedding(text, task_type="RETRIEVAL_QUERY"):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None

def extract_emotion(user_input: str) -> str:
    prompt = f"""
    다음 문장에서 가장 두드러지는 핵심 감정 한 가지를 '행복', '슬픔', '분노', '평온', '불안' 중에서 하나만 골라주세요.
    다른 설명 없이 감정 단어만 응답해야 합니다.

    문장: "{user_input}"
    감정:
    """
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"감정 추출 중 오류 발생: {e}")
        return "평온"

def generate_comforting_message(user_emotion: str, content: dict) -> str:
    content_type = list(content.keys())[0]
    content_name = content[content_type]
    prompt = f"""
    사용자는 현재 '{user_emotion}'의 감정을 느끼고 있습니다.
    이 사용자에게 따뜻한 위로와 공감의 말을 전해주세요.
    그리고 사용자의 현재 감정과 다른 새로운 경험을 할 수 있도록, '{content_name}'({content_type})을(를) 추천해주세요.
    추천하는 이유를 자연스럽게 설명하며 메시지를 마무리해주세요.
    응답은 한국어로, 친근하고 다정한 말투로 작성해주세요.
    """
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"메시지 생성 중 오류 발생: {e}")
        return "괜찮아요, 모든 게 다 잘 될 거예요. 오늘 하루도 정말 고생 많으셨어요."

# ✅ 프런트에서 온 문장을 받아 LLM에 보내고 결과를 반환하는 접착 함수
def get_llm_answer(user_sentence: str) -> str:
    try:
        # 간단 버전: 바로 생성
        prompt = f"다음 문장에 대해 공감하고 짧게 답해주세요(한국어): \"{user_sentence}\""
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"LLM 응답 생성 중 오류: {e}")
        return "잠시 문제가 발생했어요. 다시 시도해 주세요."
