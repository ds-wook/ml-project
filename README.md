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
+ anatom_site_general_challenge - 흑색종의 위치
+ diagnosis - 진단명
+ benign_malignant - 악성인지 음성인지 유무
+ target - benign_malignant의 이진화(음성-0, 악성-1)

### Feature Engineering
##### meta data에 관하여 GBDT 모델을 사용하기 위해 Feature Engineering을 수행
+ sex_enc: 성별을 이진화
+ age_enc: 나이를 구간별로 나누어 label encoding함
+ age_approx_mean_enc: age_enc를 mean_encoding 함
+ anatom_enc: anatom_site_general_challenge를 label encoding함
+ n_images: image의 개수를 feature로 만듬
+ n_images_enc: n_images를 label encoding 함
+ image_size: image 크기를 feature로 만듬
+ image_size_scaled: image_size를 Min Max Scaler를 사용
+ image_size_enc: image_size를 categorize하여 label encoding을 수행
+ age_id_min: 환자의 id 중 나이가 가장 적은 값을 feature로 만듬
+ age_id_max: 환자의 id 중 나이가 가장 많은 사람을 feature로 만듬

## Hyperparameter Tunning 전략
+ 베이즈 최적화 이론
    + 베이즈 최적화란 직전까지 계산한 매개변수에서의 결과에 기반을 두고 다음 탐색해야 할 매개변수를 베이즈 확률구조로 선택하는 방식
+ TPE
    + 기대향상(EI)은 어떤 매개변수로 모델의 점수를 계산 했을때 점수 계선량의 기댓값을 지금까지의 탐색 이력에 추정한 값이다.
    + TPE(Tree-structured Parzen Estimator)는 기대 향샹의 계산에 필요한 ![CodeCogsEqn (1)](https://user-images.githubusercontent.com/46340424/120889562-c590c700-c638-11eb-9fbf-d127f4d42e71.gif)을 구하는 하나의 방법이다

    + 기대향상 수식

        ![CodeCogsEqn](https://user-images.githubusercontent.com/46340424/120889548-b01b9d00-c638-11eb-9208-ecca8f311832.gif)


## BenchMark
### Tabular-learning
|model|OOF(5-fold)|OOF(10-fold)|
|:-----|:---------|:--------|
|LightGBM(before hyper parameter tunning)|0.84585|0.84472|
|LightGBM(after hyper parameter tunning)|0.85360|**0.85864**|
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
|0.9 * Effinet + 0.1 * LGBM|0.90042|
|**0.8 * Effinet + 0.2 * LGBM**|**0.90172**|
|0.7 * Effinet + 0.3 * LGBM|0.90158|
|0.6 * Effinet + 0.4 * LGBM|0.90056|
|0.5 * Effinet + 0.5 * LGBM|0.90052|
