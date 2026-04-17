# 얼굴 감정 분류 프로젝트 기획서

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 과목 | 중간고사 프로젝트 |
| 주제 | 이미지 기반 감정 이진 분류 (Sentiment: Positive / Negative) |
| 데이터셋 | CMU-MOSI — Segmented 비디오에서 중간 프레임 추출 |
| 정답 라벨 | `mosi_audio_metadata.csv` → sentiment ≥ 0: Positive(1), < 0: Negative(0) |
| 목표 | 수업 코드(defalut) 대비 **(1) 전처리 고도화** + **(2) 모델 변경**으로 성능·일반화 개선 |

---

## 수업 코드(defalut) 성능 기준선

| 파일 | 모델 | Train Acc | Test Acc | 문제점 |
|---|---|---|---|---|
| defalut_2 | SimpleCNN (MaxPool) | 84.3% | 60.0% | 과적합 (갭 24%p) |
| defalut_2 | AlexNet fine-tuning | 98.5% | **64.1%** | 과적합 (갭 34%p) |
| defalut_2 | VGG16 (scratch) | 46.5% | 46.6% | 학습 불안정 |
| defalut_3 | I3D (비디오) | — | 58.9% | 이미지 정보만 활용 못함 |

> **핵심 문제:** 과적합이 심각함. AlexNet조차 Train-Test 갭 34%p. 전처리·모델 모두 개선 필요.

---

## 차별화 전략 — 두 가지 축

```
defalut 문제점                   내 개선 방향
─────────────────────────────────────────────────────
① Haar Cascade face crop     →  MTCNN 얼굴 탐지 + landmark 정렬
② 노이즈 증강만              →  Rotation / Shifting / Flip 추가
③ 클래스 균형 처리 없음      →  WeightedRandomSampler
④ /255 정규화만              →  ImageNet mean/std 정규화
⑤ LR 고정                   →  CosineAnnealingLR
⑥ SimpleCNN / AlexNet / VGG →  DAN (EfficientNet-B0 backbone)
                                 → 얼굴 부위별 Multi-head Attention
                                 → Affinity Loss로 feature 분리
```

---

## 전체 파이프라인

```
raw_data/Video/Segmented/*.mp4   (2199개 세그먼트)
    │
    ▼ [STEP 0 — 프레임 추출, 1회 저장]
raw_data/Video/Images/*.jpg      (중간 프레임 1장/세그먼트)
    │
    ▼ [STEP 1 — 전처리 고도화]
  ├─ MTCNN 얼굴 탐지 + landmark 기반 alignment
  ├─ Pad → Resize 224×224
  ├─ Augmentation: RandomHorizontalFlip / Rotation(±15°) / Shift / GaussianNoise / SaltPepper
  ├─ ImageNet Normalize (mean/std)
  └─ WeightedRandomSampler (배치 내 Pos:Neg = 1:1)
    │
    ▼ [STEP 2 — 모델 학습]
  ├─ Backbone: EfficientNet-B0 (ImageNet pretrained, fine-tune)
  ├─ Head: DAN-style Multi-head Spatial Attention (4 heads)
  ├─ Loss: CrossEntropy + Affinity Loss (λ=0.1)
  └─ Optimizer: AdamW + CosineAnnealingLR
    │
    ▼ [STEP 3 — 평가 및 비교]
  ├─ Accuracy / F1 / Confusion Matrix
  ├─ 전처리 Ablation (MTCNN / Augmentation / Sampler 각각의 기여도)
  └─ defalut 모델들과 Test Accuracy 비교 바 차트
```

---

## 모델 선정 근거 — DAN (EfficientNet-B0 backbone)

### 왜 DAN 구조인가?

> Wen et al., *"Distract Your Attention: Multi-Head Cross Attention Network for Facial Expression Recognition with Face Landmarks"*,  
> **Biomimetics 2023** / arXiv:2109.16235

- 수업에서 다룬 모델 (SimpleCNN, AlexNet, VGG, I3D)은 **얼굴 전체를 하나의 벡터**로 처리
- 감정 표현은 눈썹·눈·코·입이 **독립적으로** 변화 → 부위별 주의가 필요
- DAN은 **Multi-head Attention**으로 각 head가 서로 다른 얼굴 부위를 자동 학습
- **Affinity Loss**: 같은 감정 클래스의 feature를 가깝게 → 과적합 억제에도 기여
- **Partition Loss**: head들이 중복 없이 다른 부위를 담당하도록 유도
- 성능: AffectNet-7 **65.69%**, AffectNet-8 **62.09%** (SOTA 근처)

### 왜 EfficientNet-B0 backbone인가?

- DAN 원 논문은 ResNet-18을 backbone으로 사용
- **ResNet 계열은 과제에서 가장 흔히 선택됨** → 차별화 필요
- EfficientNet-B0:
  - 2019 ICML, Google Brain — Compound Scaling으로 depth/width/resolution 동시 최적화
  - ResNet-18 대비 파라미터 적고 정확도 높음
  - ImageNet Top-1: ResNet-18 69.8% vs EfficientNet-B0 77.1%
  - FER 과제에서 EfficientNet + DAN attention 조합은 거의 사용되지 않음

### 구조 요약

```
입력 224×224 RGB
    │
    ▼
EfficientNet-B0 Feature Extractor (pretrained, fine-tune)
  → feature map: (B, 1280, 7, 7)
    │
    ▼
Multi-head Spatial Attention (4 heads)
  ├─ Head 1: 상단 (이마·눈썹 영역)
  ├─ Head 2: 중단 (눈·코 영역)
  ├─ Head 3: 하단 (입·턱 영역)
  └─ Head 4: 전체 맥락
    │
    ▼
Attention Fusion → feature vector (B, 1280)
    │
    ├─ → Affinity Loss (학습 시)
    │
    ▼
FC → Dropout(0.5) → 2-class output
```

---

## 파일 구성

```
mid_test_image/
├── raw_data/
│   ├── mosi_audio_metadata.csv      ← 정답 라벨
│   ├── Video/
│   │   ├── Segmented/*.mp4          ← 원본 비디오 (2199개)
│   │   └── Images/*.jpg             ← 추출 이미지 (defalut_1에서 생성, 재사용)
│   └── Audio/                       ← 사용 안 함
├── defalut_1.ipynb                  ← 1차시: 전처리 기준선
├── defalut_2.ipynb                  ← 2차시: CNN 모델 기준선
├── defalut_3.ipynb                  ← 3차시: 비디오 모델 기준선
├── my_preprocessing.ipynb           ← (신규) 개선된 전처리
├── my_model.ipynb                   ← (신규) DAN + EfficientNet 구현
├── 01_project_overview.md           ← 이 파일
├── 02_data_and_training.md          ← 전처리·학습 전략 상세
└── 03_related_works.md              ← 선행 연구 분석
```

---

## 일정 계획

| 단계 | 작업 | 핵심 산출물 |
|---|---|---|
| Phase 1 | 개선 전처리 파이프라인 구축 | `my_preprocessing.ipynb` — MTCNN + 증강 + Sampler |
| Phase 2 | DAN 모델 구현 + 기본 학습 | `my_model.ipynb` — EfficientNet-B0 + 4-head Attention |
| Phase 3 | Ablation Study | 전처리 요소별 기여도 측정 |
| Phase 4 | defalut 비교 + 결과 정리 | 최종 비교 차트, Confusion Matrix |
