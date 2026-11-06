import streamlit as st
from utils.data import load_data, inventory
from utils.models import load_model

st.markdown('''<section class="titanic-hero">
<p class="eyebrow">TITANIC / DATA &amp; AI</p>
<h1>승객 데이터에서<br>생존의 패턴까지</h1>
<p>타이타닉 승객의 생존 기록을 살펴보고,<br>나이·성별·객실 등급을 AI 모델로 탐색하세요.</p>
</section>''',unsafe_allow_html=True)
df=load_data()
a,b,c=st.columns(3)
a.metric('기록 인원',f'{len(df):,}명')
b.metric('생존 기록',f'{int(df.Survived.sum())}명')
c.metric('자료 내 생존율',f'{df.Survived.mean():.1%}')
st.markdown('<div class="titanic-guide">처음이라면 <strong>데이터 분석 → AI 모델 → 분석 Q&amp;A</strong> 순서로 살펴보세요.</div>',unsafe_allow_html=True)
st.subheader('무엇을 살펴볼까요?')
menus=[
    ('데이터 분석','필터·빈칸·집단별 생존율'),('생존 예측 비교','로지스틱 회귀와 신경망의 추정 비교'),
    ('비슷한 조건 찾기','이름을 제외한 KNN 유사 기록 5개'),('승객 특징 군집','K-Means 군집과 사후 통계'),
    ('모델 평가','5개 모델 CV·독립 평가'),('분석 Q&A','근거 ID가 있는 사실 조회와 선택적 RAG')]
columns=st.columns(2)
for index,(title,description) in enumerate(menus):
    with columns[index%2]:
        st.markdown(f'<div class="titanic-card"><strong>{title}</strong><p>{description}</p></div>',unsafe_allow_html=True)
st.subheader('사용한 자료')
st.dataframe(inventory(),hide_index=True,width='stretch')
st.caption('리뷰·상품·판매자 기능은 타이타닉 자료에 맞게 교체했습니다. LSTM은 MLP로, BERT는 표 데이터용 부스팅으로 대체했습니다. 두 원본을 합산하지 않습니다.')
