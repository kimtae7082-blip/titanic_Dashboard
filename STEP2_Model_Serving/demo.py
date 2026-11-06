from pathlib import Path
import sys
APP = Path(__file__).resolve().parents[1] / 'STEP3_Local_Deploy' / 'titanic_dashboard'
sys.path.insert(0, str(APP))
from utils.models import predict

def analyze(pclass, sex, age, sibsp, parch, fare, embarked):
    outputs = []
    for name in ['logistic', 'neural']:
        p = predict(name, int(pclass), sex, float(age), int(sibsp), int(parch), float(fare), embarked)
        outputs.append({'생존 추정': p, '사망 추정': 1-p})
    return tuple(outputs)

def build_demo():
    import gradio as gr
    return gr.Interface(fn=analyze,
        inputs=[gr.Dropdown([1,2,3], value=3, label='객실 등급'), gr.Dropdown(['female','male'],value='female',label='성별'),
            gr.Number(value=25, minimum=0, maximum=100, label='나이'), gr.Number(value=0,minimum=0,maximum=10,precision=0,label='동승 형제·배우자'),
            gr.Number(value=0,minimum=0,maximum=10,precision=0,label='동승 부모·자녀'), gr.Number(value=15,minimum=0,maximum=1000,label='운임'),
            gr.Dropdown(['S','C','Q'],value='S',label='승선 항구')],
        outputs=[gr.Label(label='로지스틱 회귀'),gr.Label(label='신경망 MLP')],
        examples=[[3,'female',25,0,0,15,'S'],[1,'male',40,1,0,80,'C']],
        title='타이타닉 생존 예측 비교',description='학습 모델 두 개의 추정값입니다. 실제 생존 기록이나 확정된 개인의 운명이 아닙니다.')

if __name__ == '__main__':
    build_demo().launch(server_name='127.0.0.1', server_port=7862, share=False)
