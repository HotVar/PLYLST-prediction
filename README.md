# PLYLST-prediction

* songs 예측 모델 링크(.npy) : https://drive.google.com/file/d/1yn_JxHCmCnVjJv46TPORkTjEJsTC-yUu/view?usp=sharing  
* tags 예측 모델 링크(.npy) : https://drive.google.com/file/d/1cqITvEibPsfgR4XK3W3aKrmw2bFaleLO/view?usp=sharing  
<br>
  
  ## 0. 학습 준비  
  * 대회 제공 data와 학습 소스코드는 같은 경로에 위치  
  (+ song titles의 형태소 정보인 ['train_title.csv'] 파일은 ubuntu 환경에서 별도로 생성하였음)  
  <br>  
  
  ## 1. 학습 방법  
  ~~~
  python SongPrediction.py  
  ~~~
  * songs의 MatrixFactorization 모델인 test_songs_pred.npy 생성됨  
  <br>
  
  ~~~
  python TagPrediction.py
  ~~~  
  * tags의 MatrixFactorization 모델인 test_tags_pred.npy 생성됨  
  <br>  
  
  ## 2. 결과 생성방법  
  * 학습코드 실행 혹은 예측 모델을 다운받아 [test_songs_pred.npy]와 [test_tags_pred.npy]를 준비  
  ~~~
  python submission.py
  ~~~  
  <br>  
  * 최종 대회제출 결과물인 results.json파일 생성됨  
  
