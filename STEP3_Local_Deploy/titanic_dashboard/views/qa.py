import streamlit as st
import importlib
from utils import rag
# Streamlit keeps imported modules between reruns. Refresh the edited helper too.
importlib.reload(rag)
build_store,ask=rag.build_store,rag.ask

st.title('🤖 분석 Q&A')
st.caption('타이타닉 분석 결과를 찾아보고, 근거 ID와 함께 답하는 챗봇입니다. 자료에 없는 내용은 확인할 수 없다고 답합니다.')
st.sidebar.markdown('<div style="height:36px"></div>',unsafe_allow_html=True)
def remember_api_key():
    st.session_state['qa_api_key_value']=st.session_state.get('qa_api_key_input','').strip()

st.sidebar.text_input('OpenAI API 키',type='password',key='qa_api_key_input',
    value=st.session_state.get('qa_api_key_value',''),on_change=remember_api_key,
    help='키를 붙여 넣은 뒤 Enter를 누르세요. 현재 접속 중에는 페이지를 이동해도 유지되며 파일에는 저장하지 않습니다.')
key=st.session_state.get('qa_api_key_value','').strip()
channels={'A':'A. 분석 사실 확인 (사실표)','B':'B. 분석 + 모델 안내 (사실표 + 모델 설명서)'}
channel=st.radio('책장 고르기',list(channels),format_func=channels.get,horizontal=True)
with st.expander('예시 질문'):
    st.markdown('''- 어린이 승객의 생존율은?
- 성인 승객의 생존율은?
- 어린이와 성인의 생존율 차이는?
- 생존 예측 모델은 어떤 정보를 사용하나요?
- 자료에 없는 승객들의 국적은?''')
if not key:
    st.markdown('<div class="titanic-guide">왼쪽에 OpenAI API 키를 넣으면 질문할 수 있습니다.</div>',unsafe_allow_html=True)
    st.stop()
st.caption('질문과 선택한 근거가 OpenAI API로 전송되며 비용이 발생할 수 있습니다. 키는 파일에 저장하지 않습니다.')
history=st.session_state.setdefault(f'qa_history_{channel}',[])
for message in history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
question=st.chat_input('질문을 입력하세요')
if question:
    history.append({'role':'user','content':question})
    with st.chat_message('user'):
        st.write(question)
    with st.chat_message('assistant'):
        try:
            identity=(key,channel,rag.OPENAI_URL)
            with st.spinner('근거를 찾는 중…'):
                if st.session_state.get('rag_identity')!=identity:
                    st.session_state['rag_store']=build_store(channel=='B',key)
                    st.session_state['rag_identity']=identity
                result=ask(question,st.session_state['rag_store'],key)
            st.write(result['answer'])
            with st.expander('찾아온 근거 보기'):
                for doc in result['found']:
                    st.markdown(f'**[{doc.metadata["id"]}]** {doc.page_content}')
            history.append({'role':'assistant','content':result['answer']})
        except Exception as error:
            st.error(rag.error_message(error))
