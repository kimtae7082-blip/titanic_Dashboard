import streamlit as st
import plotly.express as px
from utils.data import load_data, age_data, summary

st.title('데이터 분석')
source=st.selectbox('분석 자료',['train.csv','titanic.csv'])
df=load_data(source)
left,right=st.columns(2)
sex=left.multiselect('성별',['female','male'],default=['female','male'])
classes=right.multiselect('객실 등급',[1,2,3],default=[1,2,3])
filtered=df[df.Sex.isin(sex)&df.Pclass.isin(classes)]
a,b,c=st.columns(3)
a.metric('선택 인원',f'{len(filtered)}명')
b.metric('생존 기록',f'{int(filtered.Survived.sum())}명')
c.metric('선택 범위 생존율',f'{filtered.Survived.mean():.1%}' if len(filtered) else '자료 없음')
st.caption(f'원본 {df.shape[0]}행 × {df.shape[1]}열 · 전체 결측치 {int(df.isna().sum().sum())}개')
with st.expander('원본 열별 결측치'):
    st.dataframe(df.isna().sum().rename('빈칸 수'),width='stretch')
if filtered.empty:
    st.info('선택 조건에 해당하는 기록이 없습니다.')
else:
    label=st.selectbox('비교 기준',['나이 집단','성별','객실 등급'])
    if label=='나이 집단':
        st.caption(f'나이 미상 {filtered.Age.isna().sum()}명 제외. 18세 미만을 어린이로 분류합니다.')
        table=summary(age_data(filtered),'AgeGroup')
    else:
        table=summary(filtered,'Sex' if label=='성별' else 'Pclass')
    column=table.columns[0]
    table[column]=table[column].astype(str)
    fig=px.bar(table,x=column,y='생존율(%)',text_auto='.1f',color=column,color_discrete_sequence=['#178b87','#31678f','#d99550'])
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(fig,width='stretch')
    st.dataframe(table.round(1),hide_index=True,width='stretch')
    st.download_button('비교표 CSV 저장',table.to_csv(index=False).encode('utf-8-sig'),'titanic_comparison.csv','text/csv')
    st.caption('생존율 차이는 인과관계를 입증하지 않습니다. 나이 미상 제외와 성별·등급 차이를 고려하세요.')
