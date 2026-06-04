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


둘 다 사용해보려고 하는데 우선적으로 기본+조기종료 모델이 성능이 더 높으니 '기본+조기종료'로 일단 사용해주시면 될 것 같아요!
--- 



<details>
<summary> <b>self-attention</b></summary>

### 파이프라인

```
Video / Audio / Text Features (.pkl)
  ↓  데이터 로드 및 결측치/형식 정리
  ↓  Train/Val/Test 분할 (비율 8:1:1, 계층적 분할 적용)
  ↓  StandardScaler 정규화 (Train 데이터 기준 피팅)
DataLoader (FeatureDataset)
  ↓  Gaussian Noise 주입 (noise_std, 과적합 방지)
  ↓  Mixup Data Augmentation (학습 시 v, a, t 무작위 혼합)
  
[모달리티 개별 인코더 (ModalEncoder)]
  ├─ Video 피처 → BatchNorm → Linear → GELU → Dropout → Linear → 128-dim
  ├─ Audio 피처 → BatchNorm → Linear → GELU → Dropout → Linear → 128-dim
  └─ Text 피처  → BatchNorm → Linear → GELU → Dropout → Linear → 128-dim
  ↓
[모달리티 융합 준비 (Regularization & Embedding)]
  ↓  Modal Dropout (학습 시 특정 모달리티 무작위 비활성화, 최소 1개 유지)
  ↓  모달리티 병합 (torch.stack) 및 Positional Embedding 추가
  
[Self-Attention 융합 (SelfAttentionFusionModel)]
  ↓  MultiheadAttention (shared_dim=128, num_heads=4)
  ↓  Add & Norm 1 (LayerNorm)
  ↓  Feed Forward Network (128 → 512 → 128)
  ↓  Add & Norm 2 (LayerNorm)
  ↓  Flatten (3개 모달 × 128 = 384-dim 융합 벡터)
  
[최종 분류기 (Classifier)]
  ↓  Linear (384 → 64) → GELU → Dropout
  ↓  Linear (64 → 2)
  
최종 분류 Logit (Class 0 / Class 1)
  ├─ 학습 (Train)
  │   ├─ CrossEntropyLoss (Label Smoothing 적용, Mixup Loss 계산)
  │   ├─ AdamW + Warmup & Cosine Decay Scheduler
  │   ├─ Gradient Clipping (기울기 폭발 방지)
  │   └─ Early Stopping (Validation Accuracy 기준 갱신)
  │
  ├─ 평가 (Evaluate / Test)
  │   ├─ Loss, Accuracy, Macro F1-Score 계산
  │   ├─ 시각화 (Loss, Acc, F1 곡선 그래프 출력)
  │   └─ Confusion Matrix 및 Classification Report 도출
  │
  ├─ 절제 연구 (Ablation Study)
  │   ├─ 의도적 모달리티 차단 (평가 시 0 텐서로 Zero Masking)
  │   ├─ 7가지 시나리오 성능 테스트 (단일 모달, 두 개 모달, 전체 조합)
  │   └─ 모달리티별 기여도 가로 막대 그래프 시각화
  │
  └─ 추론 (Single Sample Inference)
      └─ Softmax 적용 → 각 감정 클래스별 최종 확률(%) 반환
```

### Part 요약

| Part | 내용 |
| :--- | :--- |
| 1. 데이터 준비 및 전처리 | 피처(.pkl, .pt) 로드 및 결측치/형식 정리 → Train/Val/Test 분할(8:1:1, 계층적 분할) → Train 기준 StandardScaler 정규화 |
| 2. DataLoader & 증강 | `FeatureDataset` 활용, 가우시안 노이즈(`noise_std`) 주입 및 학습 시 모달리티 무작위 혼합(`Mixup`) 적용 |
| 3. 개별 모달리티 인코더 | `ModalEncoder` — 비디오/오디오/텍스트 피처 각각 BatchNorm, Linear, GELU, Dropout을 거쳐 128차원으로 투영 |
| 4. 융합 준비 | `Modal Dropout`(무작위 비활성화, 최소 1개 유지) 적용, 모달 병합(`torch.stack`) 및 포지셔널 임베딩 추가 |
| 5. Self-Attention 융합 | `SelfAttentionFusionModel` — `MultiheadAttention`(4 헤드) 적용 후 LayerNorm, FFN(128→512→128)을 거쳐 384차원 벡터로 Flatten |
| 6. 최종 분류기 (Classifier) | 384차원 융합 벡터 → 64차원으로 축소(GELU, Dropout) → 최종 2개 클래스(0 / 1) Logit 출력 |
| 7. 학습 (Train) | AdamW 옵티마이저, Warmup & Cosine Decay 스케줄러, Gradient Clipping, Label Smoothing 및 Mixup Loss 적용, 조기 종료(Early Stopping) |
| 8. 평가 (Evaluate / Test) | Loss, Accuracy, Macro F1-Score 계산, 추이 곡선 시각화, 혼동 행렬(Confusion Matrix) 및 분류 리포트 도출 |
| 9. 절제 연구 (Ablation Study) | 평가 시 Zero Masking을 통한 7가지 시나리오(단일/멀티) 성능 테스트 및 모달리티별 기여도 막대 그래프 시각화 |
| 10. 추론 (Inference) | Softmax를 적용하여 단일 샘플에 대한 각 감정 클래스별 최종 확률(%) 반환 |


### 도입 이유
#### 1. AdamW 옵티마이저 (Adam 대신 사용한 이유)
수업에서 배운 기본 Adam은 빠르고 편리하지만, 모델이 복잡해질수록 가중치가 너무 커지는 것을 막는 '정규화(Weight Decay)' 과정에서 수학적인 한계가 있습니다.

* **도입 이유:** AdamW는 이 정규화 과정을 최적화 스텝과 완벽하게 분리한 개선 버전입니다.
* **기대 효과:** 멀티모달처럼 파라미터가 많고 복잡한 모델이 학습 데이터에만 과도하게 맞춰지는 과적합(Overfitting)을 훨씬 효과적으로 방지하며, 새롭게 들어오는 테스트 데이터에 대한 일반화 성능을 높이기 위해 채택했습니다.

---

#### 2. Warmup & Cosine Decay 스케줄러 
학습의 시작과 끝을 모두 통제하기 위한 가장 진보된 스케줄링 전략입니다.

* **Warmup (예열):** 모델이 아무것도 모르는 초기 상태에서 갑자기 큰 보폭(큰 학습률)으로 학습하면 방향을 잃고 망가질 수 있습니다(기울기 폭발). 따라서 초반 몇 Epoch 동안은 학습률을 서서히 올리며 모델을 부드럽게 안정화시킵니다.
* **Cosine Decay (코사인 감쇠):** 후반부로 갈수록 학습률을 코사인 곡선처럼 아주 부드럽게(Smooth) 줄여나갑니다.
* **기대 효과:** StepLR처럼 학습률이 계단식으로 뚝뚝 떨어질 때 발생하는 충격을 없애고, 목표 지점 근처에서 미세 조정을 통해 가장 안정적으로 최적점에 안착(수렴)하게 만들기 위해 도입했습니다.

---

#### 3. Label Smoothing (라벨 스무딩)
감정 인식이라는 태스크의 본질적인 특성을 반영하기 위한 손실 함수 보정 기법입니다.

* **도입 이유:** 사람의 감정은 "100% 기쁨", "0% 슬픔"처럼 이분법으로 딱 떨어지지 않습니다. 하지만 기존 AI는 정답을 [1.0, 0.0] 처럼 극단적으로 강요받습니다.
* **기대 효과:** 정답을 [0.9, 0.1] 처럼 약간 부드럽게(Smooth) 깎아주어, 모델이 자신의 예측에 대해 너무 오만해지는 것을 방지합니다. 모호한 감정 데이터나 노이즈가 섞인 데이터에 대해서도 유연하고 강건하게 판단할 수 있도록 도와줍니다.

---

#### 4. Mixup Loss (믹스업 데이터 증강 및 손실 계산)
데이터가 부족하거나 모달리티 간의 복잡한 관계를 학습해야 할 때 사용하는 강력한 정규화 기법입니다.

* **도입 이유:** 기존에는 주어진 데이터만 그대로 학습했다면, Mixup은 두 개의 데이터를 무작위 비율(예: A데이터 70%, B데이터 30%)로 섞어 새로운 가상의 데이터를 창조하여 학습시킵니다.
* **기대 효과:** 모델이 단순히 특정 데이터의 특징을 외우는 것을 원천 차단합니다. 데이터와 데이터 사이의 빈 공간을 선형적으로 추론하는 능력을 길러주어, 노이즈가 있거나 본 적 없는 새로운 화자의 데이터가 들어와도 흔들리지 않는 강력한 성능을 이끌어내기 위해 적용했습니다.

### 최종 결과
| 모델 | val acc | test acc | test F1 |
| :--- | :--- | :--- | :--- |
| 기본256 | 87.73% | 86.82% | 86.80% |