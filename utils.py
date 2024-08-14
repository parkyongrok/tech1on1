# -------------------------------------------------------------------------
# 참고: 이 코드의 일부는 다음 GitHub 리포지토리에서 참고하였습니다:
# https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407
# 해당 리포지토리의 라이센스에 따라 사용되었습니다.
# -------------------------------------------------------------------------

import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

def load_model(model_name: str) -> ChatOpenAI:
    """
    주어진 모델 이름을 기반으로 ChatOpenAI 모델을 로드합니다.

    Args:
        model_name (str): 사용할 모델의 이름.

    Returns:
        ChatOpenAI: 로드된 ChatOpenAI 모델.
    """
    # load_dotenv()
    
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = st.screts["OPENAI_API_KEY"]
    
    if not OPENAI_API_KEY:
        raise ValueError("API Key is not set in environment variables.")
    if not model_name:
        raise ValueError("Model name must be provided.")
    
    try:
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    return llm 


def load_prompt() -> str:
    """
    프롬프트 파일을 로드합니다.
    
    Returns:
        str: 로드된 프롬프트 내용.
    """
    with open(f"prompt/prompt.prompt", "r", encoding="utf-8") as file:
        prompt = file.read().strip()
    return prompt


def set_memory() -> ConversationBufferMemory:
    """
    대화 히스토리를 저장하기 위한 메모리를 설정합니다.

    Returns:
        ConversationBufferMemory: 초기화된 대화 메모리.
    """
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def initialize_chain(llm: ChatOpenAI, memory: ConversationBufferMemory) -> LLMChain:
    """
    주어진 LLM과 메모리를 기반으로 체인을 초기화합니다.

    Args:
        llm (ChatOpenAI): 사용할 언어 모델.
        memory (ConversationBufferMemory): 대화 메모리.

    Returns:
        LLMChain: 초기화된 LLM 체인.
    """
    system_prompt = load_prompt()
    custom_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=custom_prompt, verbose=True, memory=memory)
    return chain


def generate_message(chain: LLMChain, user_input: str) -> str:
    """
    사용자 입력을 기반으로 메시지를 생성합니다.

    Args:
        chain (LLMChain): 사용할 체인.
        user_input (str): 사용자의 입력.

    Returns:
        str: 생성된 응답 메시지.
    """
    result = chain({"input": user_input})
    response_content = result["text"]
    return response_content
