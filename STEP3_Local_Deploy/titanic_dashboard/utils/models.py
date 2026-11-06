import joblib
import numpy as np
import pandas as pd
from utils.data import BASE, FEATURES

def load_model(name):
    path=BASE/'model'/name
    if not path.exists():
        raise FileNotFoundError('모델이 없습니다. 먼저 train_all.py 또는 STEP1 노트북을 실행하세요.')
    return joblib.load(path)

def predict(key,pclass,sex,age,sibsp,parch,fare,embarked):
    if key not in ['logistic','neural','boosting','tree','forest']:
        raise ValueError('알 수 없는 모델')
    if pclass not in [1,2,3] or sex not in ['female','male'] or embarked not in ['S','C','Q']:
        raise ValueError('등급·성별·항구 선택을 확인하세요.')
    nums=np.array([age,sibsp,parch,fare],dtype=float)
    if not np.isfinite(nums).all() or (nums<0).any() or age>100 or sibsp>10 or parch>10 or fare>1000 or int(sibsp)!=sibsp or int(parch)!=parch:
        raise ValueError('나이 0~100, 동승 인원 정수 0~10, 운임 0~1000 범위를 확인하세요.')
    row=pd.DataFrame([[pclass,sex,age,sibsp,parch,fare,embarked]],columns=FEATURES)
    return float(load_model(f'part7_{key}.joblib')['model'].predict_proba(row)[0,1])

def similar(passenger_id):
    art=load_model('part6_similar.joblib')
    records=art['records']
    hit=records.index[records.PassengerId==passenger_id]
    if not len(hit): return records.iloc[0:0]
    i=int(hit[0])
    distances,indices=art['model'].kneighbors(art['values'][[i]])
    matches=[(int(idx),float(dist)) for idx,dist in zip(indices[0],distances[0]) if int(idx)!=i][:5]
    result=records.iloc[[idx for idx,_ in matches]].copy()
    result['거리']=[distance for _,distance in matches]
    return result
