"""Checks actual calculations, saved artifacts, notebooks, and Streamlit pages."""
from pathlib import Path
import sys
import json
import io
import contextlib
import os

ROOT=Path(__file__).resolve().parent
APP=ROOT/'STEP3_Local_Deploy'/'titanic_dashboard'
sys.path.insert(0,str(APP))
from utils.data import load_data,FEATURES
from utils.models import load_model,similar,predict
from utils.rag import local_answer,documents
from streamlit.testing.v1 import AppTest

df=load_data()
assert df.shape==(891,12) and int(df.isna().sum().sum())==866
assert int(df.Survived.sum())==342
assert not {'Survived','alive','Name'}.intersection(FEATURES)
comparison=load_model('part9_comparison.joblib')
for key in ['logistic','neural','boosting','tree','forest']:
    art=load_model(f'part7_{key}.joblib')
    assert not set(art['train_ids'])&set(art['holdout_ids'])
    assert len(art['train_ids'])==712 and len(art['holdout_ids'])==179
    assert 0<=predict(key,3,'female',25,0,0,15,'S')<=1
assert 1 not in similar(1).PassengerId.tolist()
assert '확인할 수 없' in local_answer('F99')
assert len(documents(True))==10
print('PASS: totals, leakage exclusion, split, predictions, KNN, facts and RAG documents')

os.chdir(APP)
at=AppTest.from_file(str(APP/'app.py')).run(timeout=60)
assert not at.exception,at.exception
for view in ['analysis','survival','similar','segment','evaluation','qa']:
    at.switch_page(f'views/{view}.py').run(timeout=60)
    assert not at.exception,at.exception
    if view=='analysis':
        at.selectbox[0].set_value('titanic.csv').run()
        assert not at.exception
        at.multiselect[0].set_value([]).run()
        assert not at.exception
    elif view=='survival':
        at.button[0].click().run()
        assert not at.exception and len(at.metric)==2
    elif view=='qa':
        assert at.sidebar.text_input[0].label == 'OpenAI API 키'
        assert at.radio[0].label == '책장 고르기'
        assert len(at.chat_input) == 0
        assert not at.exception
print('PASS: seven navigation pages, second CSV, empty filter, two predictions, sidebar API input')

# Run each learning notebook; captured text is actual execution output.
os.chdir(ROOT)
for path in sorted((ROOT/'STEP1_Model_Training').glob('*.ipynb')):
    notebook=json.loads(path.read_text(encoding='utf-8'))
    namespace={}
    count=0
    for cell in notebook['cells']:
        if cell['cell_type']!='code': continue
        count+=1
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(''.join(cell['source']),str(path),'exec'),namespace)
        cell['execution_count']=count
        cell['outputs']=[{'output_type':'stream','name':'stdout','text':output.getvalue().splitlines(keepends=True)}] if output.getvalue() else []
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS notebook:',path.name)
print('PASS: all 12 learning notebooks executed')
