**KOR** · [ENG](./README_EN.md)

---

# :wave: Hi there, I'm Doeun Oh.

## Tech Stack

![c](https://img.shields.io/badge/C-a8b9cc?style=flat-square&logo=c&logoColor=black) ![python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python&logoColor=white) ![jupyter](https://img.shields.io/badge/Jupyter-f37626?style=flat-square&logo=jupyter&logoColor=white) ![mysql](https://img.shields.io/badge/mysql-4479A1?style=flat-square&logo=mysql&logoColor=white) ![matlab](https://img.shields.io/badge/MATLAB-0076a8?style=flat-square&logo=mathworks&logoColor=white)

![numpy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![matplotlib](https://img.shields.io/badge/matplotlib-11557c?style=flat-square) ![tensorflow](https://img.shields.io/badge/TensorFlow-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)

# 주요 경험

## [DACON 재정정보 AI 검색 알고리즘 경진대회 참여](https://github.com/theNocturni/WASSUP-DACON-FinAI)

### 개요

- 24.07.24 - 24.08.21, 24.09.27 - (현재)
- **Libraries** : huggingface, langchain, peft, faiss, trl, pymupdf, gmft
- [주어진 재정정보 pdf 문서](https://dacon.io/competitions/official/236295/overview/description)를 바탕으로 질문에 답변하는 gemma2 기반 LLM 모델을 RAG, LoRA를 활용하여 학습.
- **대회 성적**
  - metric : 문장에서 문자 단위의 F1 score
  - Public 0.666, Private 0.673, 최종순위 38/359 (상위 10.58%)
- 경진대회 마감 이후, 성능 개선을 위한 실험 설계 및 실험 진행 중
  - 현재 성적 : Public 0.715, Private 0.693
  - public 기준 : 현재 25/359, priveate 기준 : 대회 종료 시점의 23등에 준하는 성적

### 기여

- pymupdf와 gmft를 이용한 표 전처리, 코드 리팩토링 등에 기여
  - 표 전처리를 통해 Public 기준, 0.657에서 0.666으로 증가하고, 이후 0.690으로 증가

## [알라딘 주간 베스트 셀러 및 중고 매장 도서 DB 구축](https://github.com/kdt-3-second-Project/aladin_usedbook)

### 개요

- 24.07.11 - 24.07.22, 24.10.19~24.10.23
- **Libraries** : NumPy, Pandas, Matplotlib, Beautifulsoup, re, Scikit-learn, xgboost, Mecab
- 알라딘 00년 1월 1주차 ~ 24년 7월 2주차의 베스트셀러 목록을 크롤링하여 141.5만 행의 DB 구축
  - 15.8만 여종의 도서에 대하여, 해당 주차에서의 순위 및 도서 관련 정보를 포함
- 주간 베스트 셀러 DB를 바탕으로, 78만 행의 알라딘 중고 매장의 중고 도서 DB 구축
  - 10.3만 여종의 역대 베스트셀러 도서에 대한 중고 도서 매물 데이터
- XGBoost Regressor를 이용하여 중고가 예측 모델 개발
  - cross validation과 grid search를 이용하여 486개의 조합 중 우수 hyperparameter 14개를 추림
    - Python API 및 cupy를 이용하여 GridSearchCV를 진행할 수 있는 함수를 만들어 연산 속도를 개선
  - 우수 hyperparameter로 학습한 모델들에 대해서는 두 가지 방법으로 평가
    - test 1 : 초기에 test set으로 나눈 데이터로 평가
    - test 2 : test set 중 train set에 포함된 적 없는 종류의 도서에 한해서 평가
- Best model
  - 독립변수 : 중고품질, 취급지점, 도서명, 도서명에 포함된 부가적 문구(양장본, 한정판 등), 저자, 기타 저자, 출판사, 출간일, 정가, 대분류
  - hyperparameter
    - *num_boost_round* : 2500
    - *learning_rate* : 0.3
    - *max_depth* : 6
    - *min_child_weight* : 4
    - *colsample_bytree* : 1
    - *subsample* : 1

  ![h5_rslt](./imgs/h5_rslt.png)
  <center><i><b>도표</b>. best model의 예측값 및 오차 분포와 성능</i></center>

  ![h5_fi](./imgs/h5_fi.png)
  <center><i><b>도표</b>. best model의 feature importance</i></center>

||RMSE|R2 score|N|
|:--:|--:|--:|--:|
|test 1|610.7|0.973|784,213|
|test 2|1,440|0.914|5,968|
|harmonic mean|857.8|0.943||

<center><i><b>도표</b>. test별 데이터셋의 크기 및 XGBoost Regressor에서의 최고 성적</i></center>

### 기여

- 조장으로서 프로젝트 기획 및 진행
- 크롤링 코드 개발, DB 및 model의 prototype 개발, 실험 설계, 진행 및 평가 등에 기여

### 배운 점

- 적절한 모듈화가 개발의 효율성 및 코드의 가독성에 주는 영향력을 체감함.
- 소수의 샘플로 빠른 개발을 진행하여, 현재의 방법론이 가능한지 혹은 적절한지 평가하는 것은 전략적으로 유효함.
  - 프로젝트의 방향성을 잡는데 도움이 되고, 좋은 baseline의 기준이 될 수 있음.
  - 빠르게 prototype를 개발하는데 도메인 지식 등을 이용해 휴리스틱한 판단을 하는 것은 유효한 도움이 될 수 있음.
  - 하지만 휴리스틱한 결정들에 대해서 체계적인 기준을 세우기 위해서는 예상보다 큰 노고가 들 수 있음.
- 모델이 접한적 없는 종류의 데이터에 대해 추가적인 test를 진행함으로써 모델의 학습 정도에 대해서 적극적으로 평가할 수 있음.
  - train set에 포함 된 적 없는 종류의 도서에 대해서만 추가적인 평가를 진행.
  - 해당 test에서도 성적이 큰 차이 나지 않게 잘 나오는 것을 확인함.
  - 도서 별 가격을 모델이 외운 것이 아니라 자연어 처리 결과를 모델이 반영하고 있음을 확인하고 있었음.
- 데이터 셋의 column 중 불명확한 것은 사용하지 않아도, 모델의 복잡도를 유효한 방향으로 높히면 성능이 좋고 더 강건한 모델을 개발할 수 있음을 확인.
  - 알라딘이 개발한 판매지수(SalesPoint)를 중고도서 예측에 이용하면, 단순한 모델로도 좋은 성능을 얻을 수 있다는 장점이 있었지만 단점도 있었음
  - best model에 쓰인 hyperparamter를 포함하여, 동일한 hyperparameter로 SalesPoint를 제외하고 학습시켰을 때 성능이 더 좋고 더 강건한 경우가 몇 있었음.
  - 추산법이 공개 되지 않아 불명확할 뿐 아니라, 중고가 예측의 성능을 더 고도화하는 단계에서는 방해가 될 수 있다고 판단.
- 간단한 모델로 리버스 엔지니어링을 진행하여, 시스템이 어느 정도로 복잡하거나 단순한지 평가해볼 수 있음.
  - 간단하고 기본적인 전처리만 진행한 상황에서, 모델이 처음 보는 종류의 데이터에 대해서도 XGBoost 만으로도 충분히 좋은 성능이 나올 수 있었음.
  - 알라딘에서 중고도서에 대해서 저자, 출판사, 중고 품질 등을 기준으로 가격을 책정하고 있으며, 책정 시스템이 아주 복잡하지는 않으리라 유추할 수 있었음.
- 연산량의 관점에서 grid search는 hyperparameter 탐색에 매우 비효율적.
  - grid search를 이용하면 hyperparameter에 따른 변화를 직접적으로 관찰할 수 있어 결과 분석과 앞으로의 방향 설정에 용이하다는 장점이 있지만, 연산량의 측면에서 지나치게 비효율적.
  - 모델에 맞게 hyperparameter의 탐색 순서를 설정하거나, Bayesian search 등을 활용하면 연산에 드는 자원 및 시간을 보다 효율적으로 사용할 수 있었을 것이라 기대.
- 몇 십만 개 단위의 데이터를 XGBoost에 적용하고자 하면, Sci-kit API 보다 Python API를 이용하는 것이 연산 속도의 면에서 더 빠를 수 있고, 특히 cupy를 이용해 gpu를 사용하면 연산속도를 비약적으로 빠르게 할 수 있음.

## [미국 대도시 보건 데이터셋을 기반으로 한 질병 발병 및 사망 통계 예측 AI 모델](https://github.com/WASSUP-AIModel-3rd-Project1/Project-1)

### 개요

- 24.06.14 - 24.06.24
- **Libraries** : NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, Jit
- 미국 대도시 보건 데이터셋([BCHI Dataset](https://bigcitieshealthdata.org/))은 35개 대도시의 16종으로 층화된 인종 · 성별 인구 집단 별로 다양한 통계항목을 2010-2022 동안 집계한 데이터 셋.
  - 통계 항목은 All Cancer Death, Lung Cancer Death, Diabetes Death, Drug Overdose Death 등 총 118 종으로 구성.
    - e.g. *"Minneapolis에서 2015년에 인종 상관없이 여성에 대해 All Cancer Death를 조사한 결과, 십만명당 157명"*
  - 각 대도시는 '지역'/ '경제적 빈곤'/ '인구'/ '인구밀도'/ '인종별 거주지 분리 정도' 5가지 특성을 기준으로 분류 되어 있음.
    - 35개 도시가 총 19종의 도시 유형으로 분류됨.
    - e.g. *"Minneapolis의 도시 유형 : 중서부, 덜 빈곤한, 인구규모가 작은, 낮은 인구밀도, 인종 별 거주지 분리 정도가 낮은 도시"*
- BCHI Dataset의 다양한 통계 항목과 인종, 성별, 도시유형의 층화 정보를 이용하여 해당 집단의 특정 통계 항목의 값을 회귀 예측하는 프로젝트 진행.
  - All Cancer Deaths, Lung Cancer Deathes 등 총 14가지 통계 항목에 대하여 회귀 예측 진행.
  - e.g. *도시의 특성,인종,성별로 층화된 인구집단에 대하여, 층화된 정보 및 Adult Physical Inactivity, Diabetes, Teen Obesity, Adult Obesity, Population : Seniors, Income : Poverty in All Ages 등의 통계값를 이용하여, All Cancer Deaths 통계값을 예측*
  - 예측 방법으로 XGBoost Regressor, Random Forest Regressor, Multilayer Perceptron, k-NN Regressor을 사용.
    - k-NN의 경우는 층화 항목에 대해 $L_p$ norm을 응용한 custom metric을 이용해 예측하고, 다른 참고 항목은 사용하지 않음.
    - 기타 모델의 경우, 결측 값들을 제외하고 학습을 진행한 경우와 결측값을 k-NN을 이용한 예측값으로 보간한 뒤 진행한 경우의 성능을 비교함.
    - 평가 metric으로 RMSE, MAPE, R2 score 등을 사용.
    - 통계 항목 별로 차이가 있지만, k-NN, k-NN으로 결측을 보간한 XGBoost, k-NN으로 결측을 보간하지 않은 XGBoost 세 모델에서 성능이 제일 높게 나옴.

|예측 목표 항목|참고 항목|
|-------|------------------|
| All Cancer Deaths|Adult Physical Inactivity, Diabetes, Teen Obesity, Adult Obesity, Population : Seniors, Income : Poverty in All Ages, e.t.c.|
| Colorectal Cancer Deaths|Teen Obesity, Adult Obesity, Health Insurance : Uninsured in All Ages, Births : Low Birthweight, Dietary Quality : Teen Soda, e.t.c.|

<center><i><b>도표.</b> 각 예측 목표 항목 별로 설정된 참고 항목 후보의 예시</i></center>

![결과비교](./imgs/6-2.rslt2.png)
<center><i><b>도표.</b> k-NN, k-NN 전처리를 사용한 XGBoost, 사용하지 않은 XGBoost 간의 성능 비교</i></center>

### 기여

- 조장으로서 프로젝트 방향 제시.
- 프로젝트 방향 결정을 위한 EDA, k-NN에서 사용한 custom metric 제시 및 구현, k-NN을 활용한 결측치 보간 제안, 코드 리팩토링 등에 기여.

### 배운 점

- 회귀 예측을 평가할 때, 평균 오차에 관한 score와 r2 score를 복합적으로 이용해야 함을 익힘.
  - r2 score가 좋을수록 x에서의 차이가 y값 예측에 잘 반영되고 있고, 평균 오차에 관한 score(RMSE, MAPE 등)가 좋을수록 실제값과 오차가 적은 것을 데이터를 통해 직접적으로 볼 수 있었음.
  - 일반적으로 평균 오차에 관한 score가 좋을 수록 r2 score도 좋았으나, 항상 그런 것은 아니었음.
- 데이터 셋 특성에 따라, k-NN을 적용하여 결측 보간을 하는 것이 유효할 수 있음.
  - 다만, 다른 보간 방법 혹은 데이터를 drop하는 것에 비해 항상 압도적으로 좋지는 않음.
    - 평균 오차에 관련된 score는 대개 좋아졌지만, r2 score는 나빠지는 경우들이 있었음.
  - 도메인 지식을 바탕으로 custom metric을 설계하는 것이 유효할 수 있음.
  - numpy 및 cython에 맞게 최적화를 시키지 않을 경우, custom metric을 scikit-learn 의 k-NN에 사용하면 속도가 매우 느림.
    - 약 4천 ~ 5천 여개의 데이터를 이용해 3천 ~ 2천 여개의 데이터를 예측하는데 분 단위의 시간이 걸림.
- c를 이용할 수 있도록 리팩토링하여, Jit을 적용시킬 경우 속도가 비약적으로 빨라짐.
  - custom metric에 Jit을 적용하자, 분 단위에서 초 단위로 빨라짐.
  - input이 함수에서 처리될 때 중간값으로 문자를 경유하면 안됨.
  - dict 자료형을 사용하면 안되고, array를 사용해야 함.
- baseline을 잡기 위해 XGBoost 등의 machine learning을 사용하는 것이 개발 속도 등의 측면에서 매우 유용할 수 있음.

## [Cucker-Smale 모델 및 그 확장에 대한 시뮬레이션](https://github.com/doeun-235/Cucker-Smale-Model)

- Cucker-Smale 모델은 비선형 ODE system으로, 운동하는 물체들이 상대속도 정보를 주고 받음으로써 같은 속도로 동기화 되어 수렴할 수 있는 모델.
- Cucker-Smale 모델 및 그 확장들의 수치적 해를 구하는 시뮬레이션을 진행.
  - NumPy를 이용해 ODE의 수치적 해를 구하는 알고리즘(Runge-Kutta 4th order) 및 SDE의 수치적 해를 구하는 알고리즘(Improved Euler-Maruyama Method)를 구현함.
  - Matplotlib을 이용해 이론과 시뮬레이션이 부합함을 시각화하고, 설계에 맞게 운동이 동기화 되는 것을 확인하기 위한 시연 영상 제작.
- **석사 학위 논문** : ["Flocking Behavior in Stochastic Cucker-Smale Model with Formation Control on Symmetric Digraphs"](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=c40c7fb1b28114ebffe0bdc3ef48d419)  (개명 전 이름으로 표기됨)
  - 운동하는 물체들이 의도된 모양의 군집을 이루도록 동기화 시킬 수 있는 상호작용의 예시가 될 수 있는 모델을 제시.
  - 상대위치 및 상대속도에 대한 함수로 표현되는 힘을 노이즈가 섞인 형태로 물체들 간에 주고받는 시스템.
  - Cucker-Smale을 확률 미분방정식으로 확장한 모델로, 에너지 관련 지표를 제시해 특정 조건에서 해의 존재성과 수렴성을 보임.
- **후속 연구 논문** : ["Controlled pattern formation of stochastic Cucker-Smale systems with network structures"](https://arxiv.org/abs/2105.07353)
  - 위 모델에서의 수렴 속도에 대한 이론적 · 수치적 추정을 진행.
  - SCIE급 저널이자 SCOPUS 등재지인 ["Communications in Nonlinear Science and Numerical Simulation"](https://www.sciencedirect.com/science/article/pii/S1007570422001265?dgcid=coauthor)에 게재.
  - **기여** : 모델 제안, 해의 존재성 및 수렴성 증명, 수치적 시뮬레이션 구현, 진행 및 이론에 부합되는지 검토 등에 기여

![이론시각화](./imgs/graph_ein-3.png)
<center><i><b>도표.</b> 변수 별 기대값 간의 부등식이 이론에 맞게 성립함을 보인 예시 </i></center>

![시뮬레이션](./imgs/scs-em.v.1.8-simu-einstein.gif)
<center><i><b>도표.</b> 이론에 맞게 설계대로 운동이 동기화 됨을 보인 예시 </i></center>

# 경력

## 주식회사 딥메트릭스

- Researcher / 22.06 - 23.05
- 서울대병원 인공호흡기 자율주행 AI 프로젝트 및 분당 서울대 병원 인공호흡기 자율주행 AI 프로젝트 참여
- 데이터 전처리 프로세스 구축, 유지보수 및 개선에 참여

## 교수 경험

- 공학수학 조교 (연세대학교)
  - 2018-2020 (4학기)
  - 수학 이론 설명 및 문제풀이
    - 미적분학, 선형대수, 상미분방정식 및 편미분방정식, 복소해석 등.

# 학력

- M.S in Mathematics, 2021 (Yonsei University, Seoul)
- B.S in Mathematics & Philosophy, 2018 (Yonsei University,Seoul)

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
