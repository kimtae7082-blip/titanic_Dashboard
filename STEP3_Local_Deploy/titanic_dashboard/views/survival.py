import streamlit as st
from utils.models import predict

st.title('생존 예측 비교')
st.write('STEP2 Gradio와 같은 함수로 로지스틱 회귀와 MLP 신경망을 비교합니다.')
with st.form('survival'):
    a,b=st.columns(2)
    pclass=a.selectbox('객실 등급',[1,2,3],index=2)
    sex=a.selectbox('성별',['female','male'])
    age=a.number_input('나이',0.0,100.0,25.0)
    sibsp=a.number_input('동승 형제·배우자 수',0,10,0)
    parch=b.number_input('동승 부모·자녀 수',0,10,0)
    fare=b.number_input('운임 · 원자료 단위',0.0,1000.0,15.0)
    embarked=b.selectbox('승선 항구',['S','C','Q'])
    submit=st.form_submit_button('두 모델로 추정')
if submit:
    a,b=st.columns(2)
    for col,key,title in [(a,'logistic','로지스틱 회귀'),(b,'neural','신경망 MLP')]:
        p=predict(key,pclass,sex,age,sibsp,parch,fare,embarked)
        col.metric(title,f'{p:.1%}')
    st.info('모델의 생존 추정 확률입니다. 실제 개인의 생존 여부나 확정된 운명이 아닙니다.')
