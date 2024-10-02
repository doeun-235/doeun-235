**KOR** · [ENG](./README_EN.md)

---

# :wave: Hi there, I'm Doeun Oh.

## Education

- M.S in Mathematics, 2021 (Yonsei University, Seoul)
- B.S in Mathematics & Philosophy, 2018 (Yonsei University,Seoul)

## Tech Stack

![c](https://img.shields.io/badge/C-a8b9cc?style=flat-square&logo=c&logoColor=black) ![python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python&logoColor=white) ![jupyter](https://img.shields.io/badge/Jupyter-f37626?style=flat-square&logo=jupyter&logoColor=white) ![mysql](https://img.shields.io/badge/mysql-4479A1?style=flat-square&logo=mysql&logoColor=white) ![matlab](https://img.shields.io/badge/MATLAB-0076a8?style=flat-square&logo=mathworks&logoColor=white)

![numpy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![matplotlib](https://img.shields.io/badge/matplotlib-11557c?style=flat-square) ![tensorflow](https://img.shields.io/badge/TensorFlow-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)

## 주요 경험

### [DACON 재정정보 AI 검색 알고리즘 경진대회 참여](https://github.com/theNocturni/WASSUP-DACON-FinAI)

- 24.07.24 - 24.08.21, 24.09.27 - (현재)
- **Libraries** : huggingface, langchain, peft, faiss, trl, pymupdf, gmft
- [주어진 재정정보 pdf 문서](https://dacon.io/competitions/official/236295/overview/description)를 바탕으로 질문에 답변하는 gemma2 기반 LLM 모델을 RAG, LoRA를 활용하여 학습.
- pymupdf와 gmft를 이용한 표 전처리, 코드 리팩토링 등에 기여
  - 표 전처리를 통해 문자 단위의 F1 score가 0.657에서 0.666으로 증가
- 경진대회 마감 이후, 성능 개선을 위한 실험 설계 및 실험 진행 중

### [알라딘 주간 베스트 셀러 및 중고 매장 도서 DB 구축](https://github.com/kdt-3-second-Project/aladin_usedbook)

- 24.07.11 - 24.07.22
- **Libraries** : NumPy, Pandas, Matplotlib, Beautifulsoup, re, Scikit-learn, xgboost, Mecab
- 알라딘 00년 1월 1주차 ~ 24년 7월 2주차의 베스트셀러 목록을 크롤링하여 141.5만 행의 DB 구축
- 주간 베스트 셀러 DB를 바탕으로, 78만 행의 알라딘 중고 매장의 중고 도서 DB 구축
- XGBoost Regressor를 이용하여 중고가 예측 모델 개발하고 두 가지 방법으로 평가
  - test 1 : 초기에 test set으로 나눈 데이터로 평가
  - test 2 : valid 및 test set 중 train set에 포함된 적 없는 종류의 도서에 한해서 평가

||RMSE|R2 score|
|:--:|--:|--:|
|test 1|811|0.95|
|test 2|1413|0.91|

<center><i><b>도표</b>. test1과 test 2에서의 XGBoost Regressor의 성적</i></center>

- 크롤링 코드 개발, DB 및 model의 prototype 개발, 실험 설계 및 진행 등에 기여하고 조장으로서 프로젝트 기획 및 진행

<!--각 프로젝트 별로 성장한 내용 혹은 느낀 점 정리해두면 좋을 것 같음-->

### [미국 대도시 보건 데이터셋을 기반으로 한 질병 발병 및 사망 통계 예측 AI 모델](https://github.com/WASSUP-AIModel-3rd-Project1/Project-1)

- 24.06.14 - 24.06.24
- **Libraries** : NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, Jit
- 미국 대도시 보건 데이터셋([BCHI Dataset](https://bigcitieshealthdata.org/))는 35개 대도시의 16종으로 층화된 인종 · 성별 인구 집단 별로 다양한 통계항목을 2010-2022 동한 집계한 데이터 셋
  - 통계 항목은 All Cancer Death, Lung Cancer Death, Diabetes Death, Drug Overdose Death 등 총 118 종으로 구성
    - e.g. *"Minneapolis에서 2015년에 인종 상관없이 여성에 대해 All Cancer Death를 조사한 결과, 십만명당 157명"*
  - 각 대도시는 '지역'/ '경제적 빈곤'/ '인구'/ '인구밀도'/ '인종별 거주지 분리 정도' 5가지 특성을 기준으로 분류 되어 있음
    - 35개 도시가 총 19종의 도시 유형으로 분류됨
    - e.g. *"Minneapolis의 도시 유형 : 중서부, 덜 빈곤한, 인구규모가 작은, 낮은 인구밀도, 인종 별 거주지 분리 정도가 낮은 도시"*
- BCHI Dataset의 다양한 통계 항목과 인종, 성별, 도시유형의 층화 정보를 이용하여 해당 집단의 특정 통계 항목의 값을 회귀 예측하는 프로젝트 진행
  - All Cancer Deaths, Lung Cancer Deathes 등 총 14가지 통계 항목에 대하여 회귀 예측 진행
  - e.g. *도시의 특성,인종,성별로 층화된 인구집단에 대하여, 층화된 정보 및 Adult Physical Inactivity, Diabetes, Teen Obesity, Adult Obesity, Population : Seniors, Income : Poverty in All Ages 등의 통계값를 이용하여, All Cancer Deaths 통계값을 예측*
  - 예측 방법으로 XGBoost Regressor, Random Forest Regressor, Multilayer Perceptron, k-NN Regressor을 사용
    - k-NN의 경우는 층화 항목에 대해서만 custom metric을 이용해 예측하고, 다른 참고 항목은 사용하지 않음
    - 기타 모델의 경우, 결측 값들을 제외하고 학습을 진행한 경우와 결측값을 k-NN을 이용한 예측값으로 보간한 뒤 진행한 경우의 성능을 비교함
    - 평가 metric으로 RMSE, MAPE, R2 score 등을 사용
    - 통계 항목 별로 차이가 있지만, k-NN, k-NN으로 결측을 보간한 XGBoost, k-NN으로 결측을 보간하지 않은 XGBoost 세 모델에서 성능이 제일 높게 나옴

|예측 목표 항목|참고 항목|
|-------|------------------|
| All Cancer Deaths|Adult Physical Inactivity, Diabetes, Teen Obesity, Adult Obesity, Population : Seniors, Income : Poverty in All Ages, e.t.c.|
| Colorectal Cancer Deaths|Teen Obesity, Adult Obesity, Health Insurance : Uninsured in All Ages, Births : Low Birthweight, Dietary Quality : Teen Soda, e.t.c.|

<center><i><b>도표.</b> 각 예측 목표 항목 별로 설정된 참고 항목 후보의 예시</i></center>

![결과비교](./imgs/6-2.rslt2.png)

<center><i><b>도표.</b> k-NN, k-NN 전처리를 사용한 XGBoost, 사용하지 않은 XGBoost 간의 성능 비교</i></center>

- 프로젝트 방향 결정을 위한 EDA, k-NN에서 사용한 custom metric 제시 및 구현, k-NN을 활용한 결측치 보간 제안, 코드 리팩토링 등에 기여. 조장으로서 프로젝트 방향 제시.

### [쿠커-스메일 모델 및 그 확장에 대한 시뮬레이션](https://github.com/doeun-235/Cucker-Smale-Model)

- 비선형 ODE 시스템인 쿠커-스메일 모델 및 그 확장들의 수치적 해를 구하는 시뮬레이션을 진행
- **석사 학위 논문** : ["Flocking Behavior in Stochastic Cucker-Smale Model with Formation Control on Symmetric Digraphs"](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=c40c7fb1b28114ebffe0bdc3ef48d419)  (개명 전 이름으로 표기됨)
  - 운동하는 물체들이 의도된 모양으로 군집을 이루도록 동기화 시킬 수 있는 상호작용의 예시가 될 수 있는 모델을 제시.
  - 상대위치 및 상대속도에 대한 함수로 표현되는 힘을 노이즈가 섞인 형태로 물체들 간에 주고받는 시스템.
  - 쿠커-스메일 모델을 확률 미분방정식으로 확장한 모델로, 에너지 관련 지표를 제시해 특정 조건에서 해의 존재성과 수렴성을 보임.
- **후속 연구 논문** : ["Controlled pattern formation of stochastic Cucker-Smale systems with network structures"](https://arxiv.org/abs/2105.07353)
  - 위 모델에서의 수렴 속도에 대한 이론적 · 수치적 추정
  - SCIE급 저널이자 SCOPUS 등재지인 ["Communications in Nonlinear Science and Numerical Simulation"](https://www.sciencedirect.com/science/article/pii/S1007570422001265?dgcid=coauthor)에 게재.
  - 모델 제안, 해의 존재성 및 수렴성 증명, 수치적 시뮬레이션 진행 및 이론에 부합되는지 검토 등에 기여

## 경력

### 주식회사 딥메트릭스

- Researcher / 22.06 - 23.05
- 서울대병원 인공호흡기 자율주행 AI 프로젝트 및 분당 서울대 병원 인공호흡기 자율주행 AI 프로젝트 참여
- 데이터 전처리 프로세스 구축, 유지보수 및 개선에 참여
<!--
- **분당서울대병원 인공호흡기 자율주행 AI** : 데이터 전처리 프로세스 구축 및 개선
  - 데이터 별 입력 주파수 및 값 분포를 조사하고, 병원과 협업하여 데이터 및 결측치 정의
  - 의료 지식에 데이터가 부합하는지 검토하여 전처리 프로세스의 문제점 발견 및 개선
- **서울대병원 인공호흡기 자율주행 AI** : 데이터 전처리 프로세스 유지 보수 및 개선
  - 데이터가 과하게 필터 되는 문제 발견 후, 의료 전문가와 협업하여 해결
-->

### 교수 경험

- 공학수학 조교 (연세대학교)
  - 2018-2020 (4학기)
  - 수학 이론 설명 및 문제풀이
    - 미적분학, 선형대수, 상미분방정식 및 편미분방정식, 복소해석 등.

<!--
**neulbo-187/neulbo-187** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
