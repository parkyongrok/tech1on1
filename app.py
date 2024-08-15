# IMPORT : Streamlit 라이브러리 및 기타 라이브러리
import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from utils import load_model, set_memory, initialize_chain, generate_message

# TITLE 설정
st.title("너만의 친구, 밍기뉴")
st.markdown("<br>", unsafe_allow_html=True)  # 브라우저에 줄 바꿈을 삽입합니다.

# GPT_4O_mini : 대화 생성 모델로 사용
st.session_state.model_name = "gpt-4o-mini"

# SESSION CHECK: CHAT START 여부 및 Initialization
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
    st.session_state.memory = None
    st.session_state.chain = None


# FUNCTION: start_chat() -> 대화를 시작하는 함수
def start_chat() -> None:    
    llm = load_model(st.session_state.model_name) # Load Model: gpt-4o-mini
    # initialization session state : .chat_started, .memory, .chain
    st.session_state.chat_started = True
    st.session_state.memory = set_memory()
    st.session_state.chain = initialize_chain(
        llm, st.session_state.memory
    )


# "Start Chat" 버튼을 클릭했을 때 start_chat 함수를 호출
if st.button("Start Chat"):
    start_chat()

st.markdown("<br>", unsafe_allow_html=True)  # 브라우저에 줄 바꿈을 삽입합니다.

# 채팅이 시작된 경우
if st.session_state.chat_started:
    # 메모리나 체인이 초기화되지 않은 경우를 대비해 다시 초기화합니다.
    if st.session_state.memory is None or st.session_state.chain is None:
        start_chat()  

    # 메모리에 저장된 모든 메시지를 화면에 표시
    for message in st.session_state.memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            role = 'user'  # HumanMessage는 "user" 역할로 설정
        elif isinstance(message, AIMessage):
            role = 'assistant'  # AIMessage는 "assistant" 역할로 설정
        else:
            continue
        with st.chat_message(role):  # 해당 역할로 메시지를 화면에 표시
            st.markdown(message.content)

    # 사용자가 입력한 새로운 메시지를 처리
    if prompt := st.chat_input():
        with st.chat_message("user"):  # 사용자 메시지를 화면에 표시
            st.markdown(prompt)

        with st.chat_message("assistant"):  # AI 응답을 화면에 표시
            message_placeholder = st.empty()  # 응답을 표시할 자리 확보
            full_response = ""
            response_content = generate_message(
                st.session_state.chain, prompt
            )  # AI 응답 생성

            # 응답을 단어 단위로 나누어 점진적으로 화면에 표시
            for chunk in response_content.split():
                full_response += chunk + " "
                time.sleep(0.05)  # 각 단어 사이에 지연을 추가하여 애니메이션 효과
                message_placeholder.markdown(full_response + "▌")  # 진행 중 표시
            message_placeholder.markdown(full_response.strip())  # 최종 응답 표시
