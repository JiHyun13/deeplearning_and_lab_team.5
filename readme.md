# BERT_feature.ipynb 

담당: 22010696 이수연

<details>
<summary> <b>기본 BERT + 조기종료</b></summary>

### 파이프라인

```text
mosi_text_metadata.csv
  ↓  결측치 제거 및 데이터 분할 (Train/Val/Test)
DataLoader (BertTokenizer, MAX_LEN=50)
  ↓
SimpleBERTClassifier (bert-base-uncased)
  ↓  전체 레이어 파인튜닝 (동결 없음)
  ↓  Dropout(0.3) → Linear (768 → 256)
256-dim 피처
  ├─ return_features=False → 분류 logit (단독 학습·평가용, Early Stopping)
  └─ return_features=True  → 256-dim 벡터 (멀티모달 fusion 입력)
  ↓
pickle 캐싱 (text_features_256(basic+earlystop).pkl)
```

### Part 요약

| Part | 내용 |
| :--- | :--- |
| 1. 전처리 | csv 로드 → 결측치 제거 → Train/Val/Test 분할 |
| 2. DataLoader | `BertTokenizer` 활용, 텍스트 토큰화 (`MAX_TEXT_LEN=50`, `BATCH_SIZE=16`) |
| 3. 모델 | `SimpleBERTClassifier` — 전체 BERT 파라미터 학습, 768차원 → 256차원 축소 투영 |
| 4. 학습 설정 | Adam 옵티마이저 (BERT 파라미터 `lr=1e-5`)|
| 5. 학습 루프 | Epoch 15, `patience=5` 조기 종료(Early Stopping) 적용, 최적 모델 가중치 복원 |
| 6. 피처 추출 | 베스트 모델로 전체 데이터 256-dim 추출 → `text_features_256(basic+earlystop).pkl` |
| 7. 시각화 | 학습 종료 후 Train/Val/Test의 Loss, Accuracy, F1-Score 추이 그래프 출력 |

### 최종 결과

| 모델 | val acc | test acc | test F1 |
| :--- | :--- | :--- | :--- |
| BERT | 81.8% | 85.0% | 0.8493 |

### 피처 불러오기


```python
import pickle
import torch

# 생성된 pkl 파일 로드
with open('text_features_256(basic+earlystop).pkl', 'rb') as f:
    data = pickle.load(f)

# 데이터 추출
text_feats  = data['features']   # shape: (N, 256) torch.Tensor
text_labels = data['labels']     # shape: (N,)
```
</details>

<details>
<summary><b> 증강(유의어교체) + BERT + 조기종료</b></summary>

### 파이프라인

```
text
mosi_text_metadata.csv
  ↓  결측치 제거 및 데이터 분할 (Train/Val/Test)
  ↓  데이터 증강 (Train 셋 한정 - NLTK/nlpaug 유의어 치환)
DataLoader (BertTokenizer, MAX_LEN=50)
  ↓
TEXTEncoderWithClassifier (bert-base-uncased)
  ↓  하위 6개 레이어 동결 (Freeze)
  ↓  Dropout(0.3) → Linear (768 → 256)
256-dim 피처
  ├─ return_features=False → 분류 logit (단독 학습·평가용, Early Stopping)
  └─ return_features=True  → 256-dim 벡터 (멀티모달 fusion 입력)
  ↓
pickle 캐싱 (text_features_256(증강+동결6+earlystop).pkl)

```

### Part 요약

| Part | 내용 |
| :--- | :--- |
| 1. 전처리 | csv 로드 → 결측치 제거 → Train/Val/Test 분할 → Train 셋 유의어 데이터 증강(aug_p=0.2) |
| 2. DataLoader | `BertTokenizer` 활용, 텍스트 토큰화 (`MAX_TEXT_LEN=50`, `BATCH_SIZE=16`) |
| 3. 모델 | `TEXTEncoderWithClassifier` — BERT 하위 6레이어 동결, 768차원 → 256차원 축소 투영 |
| 4. 학습 설정 | AdamW 옵티마이저 (BERT 파라미터 `lr=1e-5`, Classifier 파라미터 `lr=1e-3` 차등 적용) |
| 5. 학습 루프 | Epoch 15, `patience=5` 조기 종료(Early Stopping) 적용, 최적 모델 가중치 복원 |
| 6. 피처 추출 | 베스트 모델로 전체 데이터 256-dim 추출 → `text_features_256(증강+동결6+earlystop).pkl` |
| 7. 시각화 | 학습 종료 후 Train/Val/Test의 Loss, Accuracy, F1-Score 추이 그래프 출력 |


### 최종 결과

| 모델 | val acc | test acc | test F1 |
| :--- | :--- | :--- | :--- |
| BERT (동결 6 + 증강) | 82.4% | 84.5% | 0.8451 |


### 피처 불러오기


```python
import pickle
import torch

# 생성된 pkl 파일 로드
with open('text_features_256(증강+동결6+earlystop).pkl', 'rb') as f:
    data = pickle.load(f)

# 데이터 추출
text_feats  = data['features']   # shape: (N, 256) torch.Tensor
text_labels = data['labels']     # shape: (N,)
```
</details>

--- 

둘 다 사용해보려고 하는데 우선적으로 기본+조기종료 모델이 성능이 더 높으니 '기본+조기종료'로 일단 사용해주시면 될 것 같아요!
