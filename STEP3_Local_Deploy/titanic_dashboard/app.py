import streamlit as st

st.set_page_config(page_title='타이타닉 데이터 연구실',page_icon='🚢',layout='wide')
st.markdown('''<style>
.stApp,[data-testid="stMain"],[data-testid="stHeader"]{background:#F5FBE6;color:#233D4D}
[data-testid="stMain"] p,[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3,[data-testid="stMain"] label,[data-testid="stMetricValue"]{color:#233D4D!important}
[data-testid="stSidebar"]{background:#233D4D}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] a{color:#F5FBE6!important}
[data-testid="stSidebar"] a[aria-current="page"]{background:#215E61;border-left:5px solid #FE7F2D;border-radius:9px}
[data-testid="stSidebar"] [data-testid="stAlert"]{background:#215E61;color:#F5FBE6}
[data-testid="stMetric"]{background:#fff;padding:24px;border:1px solid #dce5cc;border-radius:20px;border-top:5px solid #215E61}
[data-testid="stMetricValue"]{color:#215E61!important}
.titanic-hero{background:#215E61;padding:30px 42px 38px;border-radius:0 0 28px 28px;margin:0 0 32px}
[data-testid="stMain"] .titanic-hero .eyebrow{color:#FE7F2D!important;font-weight:800;letter-spacing:2px;font-size:15px;margin-bottom:20px}
[data-testid="stMain"] .titanic-hero h1{color:#F5FBE6!important;font-size:clamp(30px,3.4vw,48px);line-height:1.5;padding:0;margin:0 0 20px;word-break:keep-all}
[data-testid="stMain"] .titanic-hero p{color:#F5FBE6!important;font-size:18px;line-height:1.8;margin:0;word-break:keep-all}
.titanic-guide{background:#fff0df;border-left:5px solid #FE7F2D;padding:20px 24px;border-radius:14px;margin:28px 0;color:#233D4D}
.titanic-card{border:1px solid #cbd7bb;border-radius:12px;padding:24px;margin-bottom:18px;min-height:145px;background:#fbfdf5}
.titanic-card strong{color:#215E61;font-size:19px}
.stButton>button,.stDownloadButton>button{border-color:#215E61}
@media(max-width:600px){.titanic-hero{padding:26px 22px}.titanic-hero h1{font-size:30px}}
</style>''',unsafe_allow_html=True)
page=st.navigation({
    '소개':[st.Page('views/home.py',title='홈',icon='🏠'),st.Page('views/analysis.py',title='데이터 분석',icon='📊')],
    'AI 모델':[st.Page('views/survival.py',title='생존 예측 비교',icon='🔎'),st.Page('views/similar.py',title='비슷한 조건 찾기',icon='👥'),
                st.Page('views/segment.py',title='승객 특징 군집',icon='🧩'),st.Page('views/evaluation.py',title='모델 평가',icon='📈')],
    'RAG 챗봇':[st.Page('views/qa.py',title='분석 Q&A',icon='🤖')]})
if page.url_path != 'qa':
    st.sidebar.caption('TITANIC · DATA JOURNALISM LAB')
    st.sidebar.info('제공된 학습 자료의 기록입니다. 승객·승무원 전체 통계로 일반화할 수 없습니다.')
page.run()
