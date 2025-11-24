# llm_utils.py
import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from prompt.characters import get_character_prompt

from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


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


# 허깅페이스 임베딩 모델 정의 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# =============================================
# 랭체인 chatopenai

llm = ChatOpenAI(
    model = "gpt-4o-mini"
)
empathetic_llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.8
)

# extract_recent_emotion()
recent_emotion_llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.3
)

# generate_recommendation_response()
recommend_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# ==================================================


# 🔹 임베딩 함수 (Google 임베딩 사용 - 벡터 DB와 동일한 모델)
# txt를 임베딩 벡터로 반환하는 함수
def get_embedding(text):
    try:
        # 최근 감정 벡터화 
        # 허깅페이스 임베딩 모델
        result = embedding_model.embed_query(text)

        return result # 응답에서 임베딩 배열을 꺼내 반환
    
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None

# ===================================================

# 🔹 감정 추출 함수 (main.py)
def extract_emotion(user_input: str) -> str:
    
    messages = [("user", f"""
        다음 문장에서 가장 두드러지는 핵심 감정 한 가지를
        '기쁨', '설렘', '보통', '분노', '슬픔', '불안' 중에서 하나만 골라주세요.
        다른 설명 없이 감정 단어만 응답해야 합니다.

        문장: "{user_input}"
        감정:
        """)]
    
    try: 
        # open ai의 clint 사용 -> 채팅 completions api 호출
        response = llm.invoke(messages)
      
        return response.content.strip()
    
    except Exception as e:
        print(f"감정 추출 중 오류 발생: {e}")
        return "보통"


recent_emotion_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    다음은 사용자가 작성한 대화 내역입니다.
    전체 대화를 읽고, 가장 최근에 표현된 감정을 파악해주세요.

    중요한 규칙:
    1. "고마워", "감사", "ㅋㅋ", "ㅎㅎ" 같은 짧은 인사말이나 반응은 무시하세요.
    2. 실제 감정이 담긴 의미 있는 문장만 분석하세요.
    3. 여러 감정이 섞여 있다면, 시간 순서상 가장 마지막에 나타난 감정을 선택하세요.
    4. '기쁨', '설렘', '보통', '슬픔', '분노' '불안' 중에서 하나만 골라주세요.
    5. 다른 설명 없이 감정 단어만 응답해야 합니다.
"""), ("user", """다음은 사용자가 작성한 대화 내역입니다.
       
       {conversation_history}

       위 대화에서 가장 최근에 드러난 감정 하나를 골라주세요."""),
])

recent_emotion_chain = recent_emotion_prompt | recent_emotion_llm

# 🔹 전체 대화에서 최근 감정 추출 함수 (개선 버전)
# 대화 전체를 분석하여 가장 최근에 표현된 마지막 감정을 반환 
def extract_recent_emotion(conversation_history: str) -> str:
    """
    대화 전체를 분석하되, 여러 감정이 있을 경우 가장 최근 감정을 선택합니다.
    짧은 인사말이나 간단한 응답은 무시합니다.
    """
    '''
    prompt = f"""
    다음은 사용자가 작성한 대화 내역입니다.
    전체 대화를 읽고, 가장 최근에 표현된 감정을 파악해주세요.

    중요한 규칙:
    1. "고마워", "감사", "ㅋㅋ", "ㅎㅎ" 같은 짧은 인사말이나 반응은 무시하세요.
    2. 실제 감정이 담긴 의미 있는 문장만 분석하세요.
    3. 여러 감정이 섞여 있다면, 시간 순서상 가장 마지막에 나타난 감정을 선택하세요.
    4. '기쁨', '설렘', '보통', '슬픔', '분노' '불안' 중에서 하나만 골라주세요.
    5. 다른 설명 없이 감정 단어만 응답해야 합니다.

    대화 내역:
    {conversation_history}

    가장 최근 감정:
    """
    '''
    # 대화 내역이 비어있거나 너무 짧으면 기본값(평온) 반환
    if not conversation_history or len(conversation_history.strip()) < 5:
        return "보통"
    
    # 유효한 감정인지 확인
    valid_emotions = ["기쁨", "설렘", "보통", "슬픔", "분노", "불안"]

    try:
        response = recent_emotion_chain.invoke({
          "conversation_history":conversation_history
        })

        emotion = response.content.strip()

        # 추출된 최근 감정이 기쁨, 설렘, 보통, 슬픔, 분노, 불안 안에 있으면
        # 해당 감정 반환. 아니면 기본값(보통)
        if emotion in valid_emotions:
            return emotion
        
        return "보통"
    
    except Exception as e:
        print(f"최근 감정 추출 중 오류 발생: {e}")
        return "보통"


# 🔹 위로 메시지 생성 함수
def generate_comforting_message(user_emotion: str, content: dict) -> str:
    # content 딕셔너리의 첫 키/ 값을 취해서 추천 콘텐츠 정보를 만듦
    # ex) "movie" : "너의 이름은" 과 같은 구조
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


def to_langchain_history(chat_history: List[Dict]) -> List[BaseMessage]:
    # chat_history를 langchain의 human message, ai message로 변환

    langchain_messge: List[BaseMessage] = []

    for i in chat_history:
        role = i.get("role")
        content = i.get("content","")
        if role == "user":
            langchain_messge.append(HumanMessage(content = content))
        elif role == "assistant":
            langchain_messge.append(AIMessage(content = content))
    
    return langchain_messge

# =============================================

# 프롬프트
empathetic_prompt = ChatPromptTemplate.from_messages([
    ("system", """{character_prompt}
     중요한 규칙:
    1. 사용자의 감정을 먼저 인정하고 공감해주세요
    2. 사용자의 경험을 소중하게 여기는 태도를 보여주세요
    3. 판단하지 말고, 있는 그대로 받아들여주세요
    4. 따뜻하고 진심 어린 위로를 전해주세요
    5. 캐릭터의 말투를 유지하면서도 진정성을 잃지 마세요
     """), MessagesPlaceholder(variable_name="chat_history"),

     ("user", """사용자가 이렇게 말했습니다: "{user_sentence}"

    감정 분석 결과: {user_emotion}

    당신의 캐릭터 특성을 살려서, 사용자에게 진심으로 공감하고 위로해주세요.
    사용자의 감정을 충분히 이해하고 있다는 것을 보여주며,
    따뜻한 말로 응답해주세요.

    응답은 3-5문장 정도로 작성해주세요.
    """),
])

# 체인 구성 
empathetic_chain = empathetic_prompt | empathetic_llm

# 🔹 공감 기능 강화된 응답 함수
def generate_empathetic_response(character: str, user_sentence: str, user_emotion: str, chat_history: list[dict] = None) -> str:
    """
    사용자의 말에 깊이 공감하는 응답을 생성합니다.
    """

    # 캐릭터 프롬프트 가져오기
    # 강아지 -> 강아지 프롬프트 
    character_prompt = get_character_prompt(character)

    lc_history = to_langchain_history(chat_history or [])
    '''
    # 공감 중심의 프롬프트 구성
    system_prompt = character_prompt + """
    중요한 규칙:
    1. 사용자의 감정을 먼저 인정하고 공감해주세요
    2. 사용자의 경험을 소중하게 여기는 태도를 보여주세요
    3. 판단하지 말고, 있는 그대로 받아들여주세요
    4. 따뜻하고 진심 어린 위로를 전해주세요
    5. 캐릭터의 말투를 유지하면서도 진정성을 잃지 마세요
    """
    
    # 시스템에게 전달한 메세지 (시스템, 대화 기록, 현재 사용자의 문장 순서)
    messages = [] 
    messages.append({"role" : "system", "content":system_prompt})

    # 이전 대화 있으면 넣어주기 
    if chat_history:
        messages.extend(chat_history)


    user_prompt = f"""
    사용자가 이렇게 말했습니다: "{user_sentence}"

    감정 분석 결과: {user_emotion}

    당신의 캐릭터 특성을 살려서, 사용자에게 진심으로 공감하고 위로해주세요.
    사용자의 감정을 충분히 이해하고 있다는 것을 보여주며,
    따뜻한 말로 응답해주세요.

    응답은 3-5문장 정도로 작성해주세요.
    """

    messages.append({"role":"user", "content": user_prompt})
'''

    try:
        response = empathetic_chain.invoke({
            "character_prompt": character_prompt,
            "chat_history": lc_history,
            "user_sentence": user_sentence,
            "user_emotion": user_emotion,
        })

        return response.content.strip()
    
    except Exception as e:
        print(f"공감 응답 생성 중 오류 발생: {e}")
        return "괜찮아요, 당신의 이야기를 듣고 있어요. 함께 있어줄게요."

# ===============================================================

recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", """ {character_prompt}
     당신은 사용자의 감정 상태를 파악하고, 그에 맞는 추천을 해주는 역할입니다.
     추천할 때는:
     1. 사용자의 현재 감정을 먼저 공감해주세요
     2. 왜 이 추천이 도움이 될지 설명해주세요
     3. 캐릭터의 특성을 살려 자연스럽게 추천해주세요
     """), ("user", """사용자의 현재 감정: {current_emotion}
     추천 카테고리: {category}

     다음은 추천할 콘텐츠 정보입니다:

     {formatted_recommendation}

     위 정보를 바탕으로, 캐릭터의 말투를 살려서 자연스럽게 추천 멘트를 작성해주세요.

     요구사항:
     - 4~6문장 정도로 작성
     - 첫 문장에서는 사용자의 감정을 공감해주기
     - 중간에는 왜 이 콘텐츠가 지금의 감정에 도움이 되는지 설명
     - 마지막에는 따뜻한 응원 또는 격려 한 마디로 마무리
     - 한국어로 작성 """)
])

recommend_chain = recommend_prompt | recommend_llm


# 🔹 추천 응답 생성 함수
def generate_recommendation_response(character: str, category: str, recommendation_data: dict, formatted_recommendation: str) -> str:
    """
    RAG 기반 추천을 캐릭터 말투로 전달합니다.
    """

    # 캐릭터 프롬프트 가져오기
    character_prompt = get_character_prompt(character)

    # 감정 정보 추출
    current_emotion = recommendation_data.get("current_emotion", "")
    recommended_emotion = recommendation_data.get("recommended_emotion", "")
    '''
    system_prompt = character_prompt + """

    당신은 사용자의 감정 상태를 파악하고, 그에 맞는 추천을 해주는 역할입니다.
    추천할 때는:
    1. 사용자의 현재 감정을 먼저 공감해주세요
    2. 왜 이 추천이 도움이 될지 설명해주세요
    3. 캐릭터의 특성을 살려 자연스럽게 추천해주세요
    """

    user_prompt = f"""
    사용자의 현재 감정: {current_emotion}
    추천 카테고리: {category}

    다음 추천 정보를 바탕으로, 캐릭터의 말투를 살려서 자연스럽게 추천해주세요:

    {formatted_recommendation}

    사용자에게 이 추천이 왜 좋은지, 어떤 도움이 될지 함께 설명해주세요.
    응답은 4-6문장 정도로 작성해주세요.
    """
    '''

    try:
        response = recommend_chain.invoke({
            "character_prompt": character_prompt,
            "current_emotion": current_emotion, 
            "category": category,
            "formatted_recommendation": formatted_recommendation,
        })

        return response.content.strip()
    
    except Exception as e:
        print(f"추천 응답 생성 중 오류 발생: {e}")
        return f"{formatted_recommendation}\n\n이 추천이 도움이 되었으면 좋겠어요!"


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
