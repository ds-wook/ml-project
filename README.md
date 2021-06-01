# ml-project
+ 기계학습 1학기 프로젝트
+ kaggle의 [IIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview)대회의 데이터를 활용하여 image 데이터와 tabular 데이터를 활용하여 학습을 진행하고자 합니다.

## 데이터 설명
### Files
+ train.csv - 전체 데이터 셋
+ train - 이미지 데이터셋

### Columns
+ image_name - 환자의 피부 사진 이미지 이름
+ patient_id - 환자 고유 번호
+ sex - 성별
+ age_approx - 대략적인 나이
+ anatom_site_general_challenge - 이미지의 위치
+ diagnosis - 진단명
+ benign_malignant - 악성인지 음성인지 유무
+ target - benign_malignant의 이진화(음성-0, 악성-1)

## Hyperparameter Tunning 전략
+ Bayesian TPE 방식으로 빠르게 하이퍼파라미터 튜닝 -> AutoML로 접근

## BenchMark
### Tabular-learning
|model|OOF(5-fold)|OOF(10-fold)|
|:-----|:---------|:--------|
|LightGBM(before hyper parameter tunning)|0.85585|0.85472|
|LightGBM(after hyper parameter tunning)|0.85188|**0.86138**|
|CatBoost(before hyper parameter tunning)|0.84594|0.84606|
|CatBoost(after hyper parameter tunning)|0.84360|0.84485|
|XGBoost(before hyper parameter tunning)|0.85893|0.86022|
|XGBoost(after hyper parameter tunning)|0.84347|0.84705|

### Image-learning
|model|Epoch 5|Epoch 10|
|:-----|:---------|:--------|
|Efficent Net|0.87646|**0.89759**|

### Ensemble Model
|model|ROC-AUC-Score|
|:-----|:---------|
|0.9 * Effinet + 0.1 * LGBM|0.90053|
|0.8 * Effinet + 0.2 * LGBM|0.90222|
|0.7 * Effinet + 0.3 * LGBM|0.90278|
|0.6 * Effinet + 0.4 * LGBM|0.90201|
|0.5 * Effinet + 0.5 * LGBM|0.90052|
