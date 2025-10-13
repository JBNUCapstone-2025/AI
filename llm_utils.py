# llm_utils.py
import os
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# ✅ .env 불러오기
load_dotenv()

# ✅ 환경변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# ✅ OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# ✅ Google AI 초기화
genai.configure(api_key=google_api_key)


# 🔹 임베딩 함수 (Google 임베딩 사용 - 벡터 DB와 동일한 모델)
def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        return result['embedding']
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None


# 🔹 감정 추출 함수
def extract_emotion(user_input: str) -> str:
    prompt = f"""
    다음 문장에서 가장 두드러지는 핵심 감정 한 가지를 
    '행복', '슬픔', '분노', '평온', '불안' 중에서 하나만 골라주세요.
    다른 설명 없이 감정 단어만 응답해야 합니다.

    문장: "{user_input}"
    감정:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"감정 추출 중 오류 발생: {e}")
        return "평온"


# 🔹 위로 메시지 생성 함수
def generate_comforting_message(user_emotion: str, content: dict) -> str:
    content_type = list(content.keys())[0]
    content_name = content[content_type]

    prompt = f"""
    사용자는 현재 '{user_emotion}'의 감정을 느끼고 있습니다.
    이 사용자에게 따뜻한 위로와 공감의 말을 전해주세요.
    그리고 사용자의 현재 감정과 다른 새로운 경험을 할 수 있도록,
    '{content_name}'({content_type})을(를) 추천해주세요.
    추천하는 이유를 자연스럽게 설명하며 메시지를 마무리해주세요.
    응답은 한국어로, 친근하고 다정한 말투로 작성해주세요.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"메시지 생성 중 오류 발생: {e}")
        return "괜찮아요, 모든 게 다 잘 될 거예요. 오늘 하루도 정말 고생 많으셨어요."


# 🔹 캐릭터별 응답 생성 함수
def generate_character_response(character: str, user_emotion: str, content: dict) -> str:
    """
    캐릭터 말투를 반영한 위로 메시지를 생성합니다.
    """
    from prompt.characters import get_character_prompt

    # 콘텐츠 정보 추출
    if "error" in content:
        content_description = "추천할 콘텐츠가 없어요."
    else:
        content_type = list(content.keys())[0]
        content_name = content[content_type]
        content_description = f"{content_name} ({content_type})"

    # 캐릭터 프롬프트 가져오기
    character_prompt = get_character_prompt(character)

    # 전체 프롬프트 구성
    system_prompt = character_prompt
    user_prompt = f"""
    사용자는 현재 '{user_emotion}'의 감정을 느끼고 있습니다.
    당신의 캐릭터에 맞는 말투로 사용자를 따뜻하게 위로하고,
    '{content_description}'을(를) 추천해주세요.

    캐릭터의 특징을 잘 살려서 자연스럽고 진정성 있는 메시지를 작성해주세요.
    응답은 한국어로 3-5문장 정도로 작성해주세요.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"캐릭터 응답 생성 중 오류 발생: {e}")
        return "괜찮아요, 모든 게 다 잘 될 거예요. 오늘 하루도 정말 고생 많으셨어요."


# 🔹 간단 응답 함수
def get_llm_answer(user_sentence: str) -> str:
    try:
        prompt = f"다음 문장에 대해 공감하고 짧게 답해주세요(한국어): \"{user_sentence}\""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM 응답 생성 중 오류: {e}")
        return "잠시 문제가 발생했어요. 다시 시도해 주세요."
