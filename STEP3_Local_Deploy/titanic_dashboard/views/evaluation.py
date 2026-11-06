import pandas as pd
import streamlit as st
from utils.models import load_model
from utils.training import NAMES

st.title('모델 평가')
art=load_model('part9_comparison.joblib')
st.write(f'학습 {art["train_rows"]}명 · 최종 평가 {art["holdout_rows"]}명 · 같은 분할 기준 42')
st.success('학습용 5회 교차검증으로 선택: '+NAMES[art['selected']])
st.dataframe(art['comparison'].round(3),hide_index=True,width='stretch')
key=st.selectbox('평가 상세',list(NAMES),format_func=NAMES.get,index=list(NAMES).index(art['selected']))
model=load_model(f'part7_{key}.joblib')
st.caption(f'여성만 생존으로 예측하는 단순 규칙의 평가 정확도: {model["baseline_accuracy"]:.1%}')
st.dataframe(pd.DataFrame(model['matrix'],index=['실제 사망','실제 생존'],columns=['예측 사망','예측 생존']))
st.caption('정확도: 전체 중 맞힌 비율 · 정밀도: 생존 예측 중 실제 생존 비율 · 재현율: 실제 생존자를 찾아낸 비율. 최종 평가 점수로 모델을 재선택하지 않습니다. 확률 보정은 별도 검증하지 않았습니다.')
