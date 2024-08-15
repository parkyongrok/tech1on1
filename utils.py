import os
import streamlit as st
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
from transformers import pipeline


def load_model(model_name: str) -> ChatOpenAI:
    """
    ChatOpenAI모델(gpt-4o-mini)을 로드합니다.
    OPENAI_API_KEY 를 환경변수 파일에서 불러옵니다.

    Args:
        model_name (str): 사용할 모델의 이름.

    Returns:
        ChatOpenAI: 로드된 ChatOpenAI 모델.
    """
    # # dotenv 이용해서 API_KEY 호출 -> local 환경에서 테스트시 사용    
    # load_dotenv()
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # st.secrets[] 이용해서 streamlit cloud 에서 API KEY 호출
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
     
    # ERROR CHECK : API KEY, model, llm 제대로 불러왔는지 확인
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
    LOAD promt/prompt.promt file
    나만의 친구에 어울리는 prompt 작성해둔 파일 호출
    
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
    
    
    result_openai = chain({"input": user_input})
    response_content = result_openai["text"]
    
    # 감정 분석 파이프라인 초기화
    sentiment_analysis = pipeline("sentiment-analysis",model="monologg/koelectra-base-finetuned-nsmc")
    # OPENAI 로 생성된 메세지 감정분석
    result_hugging = sentiment_analysis(response_content)
    # 영어로 나오는 결과 한글로 변경
    if result_hugging[0]['label'] == 'positive':
        result_sentiment_analysis = '(긍정적)'
    elif result_hugging[0]['label'] == 'negative':
        result_sentiment_analysis = '(부정적)'
    else:
        result_sentiment_analysis = '(애매해)'

    return response_content + result_sentiment_analysis
