# 타이타닉 STEP1 · STEP2 · STEP3

원본 Olist 프로젝트의 12개 학습 노트북 → Gradio 서빙 → 다중 페이지 Streamlit 구조를 타이타닉에 맞게 재구성했습니다. 원본 예제를 덮어쓰지 않았습니다.

## 시작
이 폴더의 실행.cmd를 더블클릭하세요. 처음에는 전용 환경·패키지 설치와 모델 학습을 수행합니다. 정상 결과는 Streamlit 홈 화면입니다. Gradio_실행.cmd는 별도 Gradio 화면을 엽니다. 서버 창을 닫으면 종료됩니다.

수동 실행(이 폴더의 PowerShell):
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe STEP3_Local_Deploy/titanic_dashboard/train_all.py
.venv\Scripts\python.exe -m streamlit run STEP3_Local_Deploy/titanic_dashboard/app.py
```
STEP1은 01→10 순서이며 07은 세 개 파일입니다. 각 노트북의 코드와 실제 출력으로 학습 과정을 확인하세요. STEP2는 저장된 로지스틱·신경망 모델을 읽습니다. STEP3은 같은 모델과 자료를 불러옵니다.

## 원본과의 대응
| Olist | Titanic |
|---|---|
| 01~03 소개·불러오기·이해 | 동일 단계, 제공 CSV 사용 |
| 04 여러 표 병합 | 분석 자료 준비, 중복 데이터 병합 금지 |
| 05 탐색 | 생존율·결측치·나이 비교 |
| 06 상품 KNN | 비슷한 승객 조건 검색 |
| 07-1 리뷰 로지스틱 | 생존 로지스틱 |
| 07-2 리뷰 LSTM | 표 데이터 MLP 신경망 |
| 07-3 리뷰 BERT | 표 데이터 Gradient Boosting (BERT 아님) |
| 08 판매자 군집 | 승객 특징 군집 |
| 09 판매자 이탈 | 생존 예측 모델 비교·선택 |
| 10 사업 전략 | 기사 사실표·모델 설명서 |
| STEP2 Gradio 모델 2개 | 로지스틱·MLP 비교 화면 |
| STEP3 app/views/utils/data/model | 동일 폴더 분리 방식 |
| RAG Q&A | 로컬 사실 조회 + 선택적 API RAG |

판매자·주문·리뷰·규정 데이터가 없으므로 관련 기능과 규정 PDF를 지어내지 않았습니다. API RAG는 키를 직접 입력할 때만 외부 서비스로 질문과 선택 근거를 전송합니다. 별도 비용이 발생할 수 있습니다. 기본 로컬 조회는 API 키가 필요 없습니다.

Python 3.12 환경에서 검증하며 3.14.7에서는 직접 검증하지 않았습니다. 학습 자료에 생존 정답이 있는 891명만 평가에 사용합니다. test.csv와 gender_submission.csv는 정답으로 사용하지 않습니다. 두 원본 CSV는 같은 기록의 다른 표현이므로 합산하지 않습니다. 이름·표 번호는 화면에 표시하지 않습니다. 원출처와 이용 조건은 제출 전 확인하세요.
