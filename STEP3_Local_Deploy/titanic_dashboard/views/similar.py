import streamlit as st
from utils.models import similar,load_model

st.title('비슷한 조건 찾기')
st.write('상품 추천의 KNN 방법을 승객 특징 검색에 적용했습니다. 이름·표 번호·정답은 거리 계산에 쓰지 않습니다.')
art=load_model('part6_similar.joblib')
pid=st.selectbox('자료 내 승객 번호',art['records'].PassengerId.tolist())
st.dataframe(art['records'].loc[art['records'].PassengerId==pid],hide_index=True,width='stretch')
st.subheader('가까운 기록 5개')
st.dataframe(similar(pid).round(3),hide_index=True,width='stretch')
st.caption('거리는 전처리한 특징 사이의 거리입니다. Survived는 비교 참고용 실제 기록이며 검색에 사용하지 않았습니다. 가까운 사람의 운명이 같다는 뜻은 아닙니다.')
