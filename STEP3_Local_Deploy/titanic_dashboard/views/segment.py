import streamlit as st
import plotly.express as px
from utils.models import load_model

st.title('승객 특징 군집')
art=load_model('part8_segment.joblib')
st.write(f'후보 2~5개 중 실루엣 점수로 선택한 군집 수: {art["k"]}')
st.caption('생존 정답을 제외한 특징으로 비슷한 집단을 만듭니다. 군집 번호에는 우열이 없습니다.')
table=art['profiles'].copy()
table['사후 생존율(%)']=100*table.pop('사후 생존율')
st.dataframe(table.round(1),hide_index=True,width='stretch')
fig=px.bar(table,x='군집',y='인원',color='군집')
st.plotly_chart(fig,width='stretch')
with st.expander('후보별 실루엣 점수'):
    st.write(art['scores'])
st.caption('사후 생존율은 군집을 만든 뒤 정답과 대조한 기술 통계입니다. 군집을 생존 위험 등급으로 해석하면 안 됩니다.')
