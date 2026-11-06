import json
import hashlib
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from utils.data import BASE, DATA, FEATURES, load_data, export_facts

MODEL=BASE/'model'
MODEL.mkdir(exist_ok=True)
NAMES={'logistic':'로지스틱 회귀','neural':'신경망 MLP','boosting':'Gradient Boosting','tree':'깊이 3 결정트리','forest':'랜덤포레스트'}

def prep():
    return ColumnTransformer([
        ('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),['Pclass','Age','SibSp','Parch','Fare']),
        ('category',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),['Sex','Embarked'])])

def split_data():
    df=load_data()
    if not df.Survived.isin([0,1]).all() or not df.PassengerId.is_unique:
        raise ValueError('생존 코드 또는 중복 번호를 확인하세요.')
    return train_test_split(df[FEATURES],df.Survived,test_size=.2,stratify=df.Survived,random_state=42)

def train_model(key):
    estimators={'logistic':LogisticRegression(solver='liblinear',max_iter=2000,random_state=42),
                'neural':MLPClassifier(hidden_layer_sizes=(16,8),early_stopping=True,max_iter=500,random_state=42),
                'boosting':GradientBoostingClassifier(n_estimators=70,max_depth=2,random_state=42),
                'tree':DecisionTreeClassifier(max_depth=3,random_state=42),
                'forest':RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=3,random_state=42,n_jobs=1)}
    x,y,yt,yv=split_data()
    model=make_pipeline(prep(),estimators[key])
    cv=cross_val_score(model,x,yt,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42),scoring='accuracy')
    model.fit(x,yt)
    pred=model.predict(y)
    probability=model.predict_proba(y)[:,1]
    metrics={'모델':NAMES[key],'CV 정확도':float(cv.mean()),'CV 표준편차':float(cv.std()),
             '최종 정확도':accuracy_score(yv,pred),'정밀도':precision_score(yv,pred,zero_division=0),
             '재현율':recall_score(yv,pred,zero_division=0),'F1':f1_score(yv,pred,zero_division=0),'ROC-AUC':roc_auc_score(yv,probability)}
    art={'model':model,'key':key,'metrics':metrics,'matrix':confusion_matrix(yv,pred,labels=[0,1]).tolist(),
         'train_ids':x.index.tolist(),'holdout_ids':y.index.tolist(),'features':FEATURES,
         'baseline_accuracy':accuracy_score(yv,(y.Sex=='female').astype(int))}
    joblib.dump(art,MODEL/f'part7_{key}.joblib')
    return art

def train_similarity():
    df=load_data()
    transformer=prep()
    values=transformer.fit_transform(df[FEATURES])
    nn=NearestNeighbors(n_neighbors=6).fit(values)
    art={'transformer':transformer,'model':nn,'records':df[['PassengerId']+FEATURES+['Survived']].copy(),'values':values}
    joblib.dump(art,MODEL/'part6_similar.joblib')
    return art

def train_segments():
    df=load_data()
    transformer=prep()
    values=transformer.fit_transform(df[FEATURES])
    candidates={}
    scores={}
    for k in range(2,6):
        model=KMeans(n_clusters=k,n_init=10,random_state=42).fit(values)
        candidates[k]=model
        scores[k]=float(silhouette_score(values,model.labels_,sample_size=500,random_state=42))
    chosen=max(scores,key=scores.get)
    labels=candidates[chosen].labels_
    table=df[['PassengerId']+FEATURES+['Survived']].copy()
    table['군집']=labels
    profiles=table.groupby('군집').agg(**{'인원':('PassengerId','size'),'평균 나이':('Age','mean'),'평균 운임':('Fare','mean'),'사후 생존율':('Survived','mean')}).reset_index()
    art={'transformer':transformer,'model':candidates[chosen],'k':chosen,'scores':scores,'profiles':profiles,'table':table}
    joblib.dump(art,MODEL/'part8_segment.joblib')
    return art

def train_all():
    artifacts=[train_model(key) for key in NAMES]
    chosen=max(artifacts,key=lambda a:a['metrics']['CV 정확도'])
    comparison=pd.DataFrame([art['metrics'] for art in artifacts])
    bundle={'selected':chosen['key'],'comparison':comparison,'train_rows':len(chosen['train_ids']),'holdout_rows':len(chosen['holdout_ids'])}
    joblib.dump(bundle,MODEL/'part9_comparison.joblib')
    train_similarity()
    train_segments()
    export_facts()
    manifest={'python':platform.python_version(),'sklearn':sklearn.__version__,'train_csv_sha256':hashlib.sha256((DATA/'train.csv').read_bytes()).hexdigest(),
              'selected_by':'training 5-fold CV accuracy','selected':chosen['key'],'features':FEATURES,'holdout_rows':len(chosen['holdout_ids'])}
    (MODEL/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return bundle
