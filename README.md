# :wave: Hi there, I'm Doeun Oh.

## Education
M.S in Mathematics, 2021 (Yonsei University, Seoul) / B.S in Mathematics & Philosophy, 2018 (Yonsei University,Seoul)

## Tech Stack
![c](https://img.shields.io/badge/C-a8b9cc?style=flat-square&logo=c&logoColor=black) ![python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python&logoColor=white) ![jupyter](https://img.shields.io/badge/Jupyter-f37626?style=flat-square&logo=jupyter&logoColor=white) ![mysql](https://img.shields.io/badge/mysql-4479A1?style=flat-square&logo=mysql&logoColor=white) ![matlab](https://img.shields.io/badge/MATLAB-0076a8?style=flat-square&logo=mathworks&logoColor=white)

![numpy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![matplotlib](https://img.shields.io/badge/matplotlib-11557c?style=flat-square) ![tensorflow](https://img.shields.io/badge/TensorFlow-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)

## 주요 경험
### [DACON 재정정보 AI 검색 알고리즘 경진대회 참여](https://github.com/theNocturni/WASSUP-DACON-FinAI)
- huggingface, langchain, peft, faiss, trl, pymupdf, gmft
- 재정정보 pdf 문서의 내용을 참고하여 주어진 질문에 답변하는 검색 알고리즘 개발하는 경진대회
- faiss 기반 retriver와 multilingual-E5기반 embedding 모델로 vector DB를 구축하고, gemma2 기반 LLM 모델 학습
- pymupdf와 gmft를 이용한 표 전처리, 코드 리팩토링 등에 기여

### [알라딘 주간 베스트 셀러 및 중고 매장 도서 DB 구축](https://github.com/kdt-3-second-Project/aladin_usedbook)
- NumPy, Pandas, Matplotlib, Beautifulsoup, re, Scikit-learn, xgboost, Mecab
- 알라딘 00년 1월 1주차 ~ 24년 7월 2주차의 베스트셀러 목록을 크롤링하여 141.5만 행의 DB 구축
- 주간 베스트 셀러 DB를 바탕으로, 78만 행의 알라딘 중고 매장의 중고 도서 DB 구축
- XGBoost Regressor를 이용하여 중고가 예측을 하였고, rmse 811, r2 score 0.95의 성능을 보임
  - train set에 포함된 적 없는 종류의 도서로 평가를 제한한 경우에도, rmse 1413, r2 score 0.91의 성능을 보임 
- 크롤링 코드 개발, DB 및 model의 prototype 개발, 실험 설계 및 진행 등에 기여하고 조장으로서 프로젝트 기획 및 진행

### [쿠커-스메일 모델 및 그 확장에 대한 시뮬레이션](https://github.com/doeun-235/Cucker-Smale-Model)
- NumPy, matplotlib
- 비선형 ODE 시스템인 쿠커-스메일 모델 및 그 확장들의 수치적 해를 구하는 시뮬레이션을 진행
- [석사 학위 논문](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=c40c7fb1b28114ebffe0bdc3ef48d419)(개명 전 이름으로 표기됨)
  - 쿠커-스메일 모델을 확률 미분방정식으로 확장한 모델을 하나 제시하고 특정 상호작용 하의 동기화 현상을 기술하는 
  - 에너지 관련 지표를 제시해 특정 조건에서의 해의 존재성과 수렴성을 보임
- [후속연구 논문](https://arxiv.org/abs/2105.07353)에 참여
  - 위 모델에서의 수렴 속도에 대한 이론적 · 수치적 추정
  - SCIE급 저널이자 SCOPUS 등재지인 ["Communications in Nonlinear Science and Numerical Simulation"](https://www.sciencedirect.com/science/article/pii/S1007570422001265?dgcid=coauthor)에 게재.
  - 모델 제안, 해의 존재성 및 수렴성 증명, 수치적 시뮬레이션 진행 및 이론에 부합되는지 검토 등에 기여

### 교수 경험
공학수학 과목들의 조교로서 수학 이론 설명 및 문제풀이 등을 연세대학교에서 맡았습니다. 4학기 동안 수업 조교를 하며 미적분학, 선형대수, 상미분방정식 및 편미분방정식, 복소해석 등을 다뤘습니다.

## Experience
### [1902 NIMS report](https://github.com/neulbo-187/1902-NIMS-report)
I made 2 programs. The first one is classifing iris data with decision tree and random forest. The second one is suggesting houses and positioning the houses on the map. I became more skillful at using sklearn, Pandas, and Folium through this project. Caused to collisions happened during installing libraries, I learned setting develop environment is so important. 

### [Making DB of Playlist with Web Crawling in Python](https://github.com/neulbo-187/making-DB-with-crawling)
I crawled datum of musics in the given playlist, from a music streaming site 'genie.com', and exported DB as xlsx file. It was my first time that making a crawling project from zero, so I learned how to choose a proper site and how to analyze the parsed page.

### [Simulations for Cucker-Smale-Model and Its Extensions](https://github.com/doeun-235/Cucker-Smale-Model)
I coded and ran simulations about the Cucker-Smale model, nonlinear ODE system describing the flocking behaviors, and its extensions. Mostly, the works have been done for writing [my master's degree thesis](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=c40c7fb1b28114ebffe0bdc3ef48d419)(I changed my name from 'Tack Geun' to 'Doeun'). I extended and suggested a model, made indices about energy to prove the theory and proved that there is a condition for flocking by analytic methods. I implemented numerical methods for solving ODE and SDE with NumPy and plotted for showing that numerical results match to theoretical results with matplotlib.

Also, I participated in [a follow-up study](https://arxiv.org/abs/2105.07353) of the thesis. The main achievement is a development of an index for estimating the speed of convergence. I ran simulations and showed that theoretical achievement is also supported by numerical simulations. The article is published in ["Communications in Nonlinear Science and Numerical Simulation"](https://www.sciencedirect.com/science/article/pii/S1007570422001265?dgcid=coauthor), listed in SCIE and SCOPUS. 

### Teaching Experience
I had taught 'Engineering Math' classes as TA in Yonsei Univ. for 2 years. The classes covered calculus, linear algebra, solving ODE and PDE, and complex analysis. 

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
