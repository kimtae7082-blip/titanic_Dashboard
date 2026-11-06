"""저장된 두 모델의 예측 비교. 웹 서버 없이 실행합니다."""
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

if __name__ == '__main__':
    outputs = analyze(3, 'female', 25, 0, 0, 15, 'S')
    for title, result in zip(['로지스틱 회귀', '신경망 MLP'], outputs):
        print(f'{title}: 생존 추정 {result["생존 추정"]:.1%}, 사망 추정 {result["사망 추정"]:.1%}')
    print('입력값을 바꾸려면 대시보드의 생존 예측 비교 메뉴를 사용하세요.')
