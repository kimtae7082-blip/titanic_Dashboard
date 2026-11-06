from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
FEATURES = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

def load_data(source='train.csv'):
    if source not in ['train.csv','titanic.csv','test.csv','gender_submission.csv']:
        raise ValueError('허용된 자료를 선택하세요.')
    df = pd.read_csv(DATA/source)
    names = {c.lower():c for c in FEATURES+['Survived','PassengerId']}
    df.columns = [names.get(c.lower(),c) for c in df.columns]
    return df

def inventory():
    return pd.DataFrame([{'파일':n,'행':len(load_data(n)),'열':len(load_data(n).columns),
                          '용도': '실제 생존 값 없음·평가 제외' if n in ['test.csv','gender_submission.csv'] else '별도 원본·합산 금지'}
                         for n in ['train.csv','titanic.csv','test.csv','gender_submission.csv']])

def age_data(df):
    known = df.dropna(subset=['Age']).copy()
    known['AgeGroup'] = known.Age.map(lambda n:'어린이(<18)' if n<18 else '성인(≥18)')
    return known

def summary(df, column):
    result = df.groupby(column,dropna=False).Survived.agg(인원='count',생존자='sum',생존율='mean').reset_index()
    result['생존율(%)'] = result.pop('생존율')*100
    return result

def export_facts():
    df=load_data()
    child=df[df.Age<18]
    adult=df[df.Age>=18]
    rows=[('F01','자료 규모',f'{df.shape[0]}행 × {df.shape[1]}열','train.csv 전체, 제목 행 제외'),
          ('F02','결측치',f'총 {df.isna().sum().sum()}개, 나이 {df.Age.isna().sum()}개','원본 빈칸 수'),
          ('F03','어린이 생존율',f'{child.Survived.mean():.1%} ({int(child.Survived.sum())}/{len(child)}명)','Age<18, 나이 미상 제외'),
          ('F04','성인 생존율',f'{adult.Survived.mean():.1%} ({int(adult.Survived.sum())}/{len(adult)}명)','Age≥18, 나이 미상 제외'),
          ('F05','생존율 차이',f'{100*(child.Survived.mean()-adult.Survived.mean()):.1f}%포인트','어린이−성인, 반올림 전 비율 차이'),
          ('F06','자료 내 전체 생존율',f'{df.Survived.mean():.1%} ({int(df.Survived.sum())}/{len(df)}명)','train.csv에 기록된 사람만, 전체 탑승자 아님')]
    facts=pd.DataFrame(rows,columns=['사실ID','내용','숫자','기준'])
    facts['출처']='제공된 train.csv에서 Python 재계산'
    facts.to_csv(DATA/'titanic_facts.csv',index=False,encoding='utf-8-sig')
    return facts
