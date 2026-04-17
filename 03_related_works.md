# 선행 연구 분석

---

## 벤치마크 성능 요약

### AffectNet 기준 (FER 표준 벤치마크)

| 모델 | 연도 | AffectNet-7 | AffectNet-8 | 핵심 기법 |
|---|---|---|---|---|
| POSTER++ | 2023 | **67.49%** | **63.77%** | Landmark + Image cross-attention |
| TriCAFFNet | 2024 | 67.40% | 63.49% | LBP+HOG+Landmark+CNN multi-feature |
| **DAN** | **2023** | **65.69%** | **62.09%** | **Multi-head attention + Affinity/Partition Loss** |
| SupCon+Relabeling | 2023 | — | 61.00% | Contrastive + uncertainty relabeling |
| EfficientNet-B0 (fine-tune) | — | ~62% | ~59% | ImageNet pretrain + CE |
| ResNet-50 (baseline) | — | ~60% | ~58% | ImageNet pretrain + CE |
| AlexNet (defalut_2 결과) | — | — | — | Test Acc **64.1%** (CMU-MOSI 이진) |

> 이 프로젝트는 AffectNet이 아닌 **CMU-MOSI 이진 분류**이므로 수치 직접 비교는 불가.  
> 그러나 구조적 우수성은 동일하게 적용됨.

---

## 채택 모델: DAN (Distract Your Attention)

### 논문 정보

> Wen et al., *"Distract Your Attention: Multi-Head Cross Attention Network for  
> Facial Expression Recognition with Face Landmarks"*  
> **Biomimetics, MDPI, 2023** / arXiv:2109.16235  
> GitHub: https://github.com/yaoing/DAN

### 문제 인식

- 기존 CNN 모델(VGG, ResNet, AlexNet)은 얼굴을 **하나의 전역 벡터**로 처리
- 그러나 감정 표현은 **눈·눈썹·입·코가 독립적으로** 변화
  - 놀람: 눈썹 올라감 + 입 벌어짐
  - 혐오: 코 주름 + 입술 내밀기
  - 분노: 눈썹 모임 + 입술 조임
- 하나의 attention으로는 이 다양한 부위별 변화를 포착하기 어려움

### 핵심 구조

```
입력 이미지
    │
    ▼
Backbone Feature Extractor
  └─ 원 논문: ResNet-18
  └─ 이 프로젝트: EfficientNet-B0 (대체, 성능·효율 개선)
  → feature map: (B, C, H, W)
    │
    ▼
Multi-head Attention Network (MAN)
  ├─ Head 1 → attention weight map 1 (상단 영역 집중)
  ├─ Head 2 → attention weight map 2 (중단 영역 집중)
  ├─ Head 3 → attention weight map 3 (하단 영역 집중)
  └─ Head 4 → attention weight map 4 (전체 맥락)
  각 head: feature map × attention weight → 가중 feature
    │
    ▼
Attention Fusion (head 평균)
  → fused feature vector
    │
    ├─ → [Affinity Loss] (학습 시)
    │     같은 클래스 feature를 클래스 중심으로 모음
    │
    ├─ → [Partition Loss] (학습 시)
    │     head 간 attention 중복 억제
    │
    ▼
FC → 분류 출력
```

### 손실 함수

```
L_total = L_CE + λ1 × L_affinity + λ2 × L_partition

L_CE        : 분류 손실 (CrossEntropy + Label Smoothing 0.1)
L_affinity  : Center Loss 변형 — 같은 클래스 feature가 클래스 중심에 가까워지도록
L_partition : head 간 코사인 유사도 최소화 — 각 head가 다른 부위 담당하도록
```

### 왜 이 프로젝트에 적합한가

| 조건 | 부합 여부 | 이유 |
|---|---|---|
| 남들이 안 쓸 모델 | ✓ | 수업에서 다루지 않음, ResNet 아님 |
| 감정 분류에 효과적 | ✓ | 얼굴 부위별 attention이 핵심 |
| 선행 연구 근거 | ✓ | MDPI Biomimetics 2023, arXiv |
| 구현 가능 | ✓ | GitHub 공개 + attention 구조 단순 |
| defalut와 비교 가능 | ✓ | 동일 데이터·입력 형식 |

### EfficientNet-B0으로 Backbone 교체하는 이유

원 논문 DAN은 ResNet-18을 backbone으로 사용.  
이 프로젝트에서는 **EfficientNet-B0**으로 교체한다.

| 항목 | ResNet-18 (원 논문) | EfficientNet-B0 (이 프로젝트) |
|---|---|---|
| 파라미터 수 | 11.7M | 5.3M |
| ImageNet Top-1 | 69.8% | 77.1% |
| FER 과제 사용 빈도 | 높음 | 낮음 |
| 특징 맵 크기 | (512, 7, 7) | (1280, 7, 7) |
| 연산량 | 1.8 GFLOPs | 0.39 GFLOPs |

> EfficientNet-B0은 ResNet-18보다 파라미터는 절반이지만 ImageNet 성능은 7%p 높음.  
> 적은 데이터(2199장)에서 과적합 억제에도 유리함.  
> 논문: Tan & Le, *"EfficientNet: Rethinking Model Scaling for CNNs"*, ICML 2019.

---

## 참고 논문 — 전처리에 활용

### MTCNN (Face Detection + Alignment)

> Zhang et al., *"Joint Face Detection and Alignment using Multi-task Cascaded CNNs"*  
> **IEEE Signal Processing Letters, 2016**

- 3단계 cascade: P-Net → R-Net → O-Net
- 5-point landmark (양쪽 눈, 코, 양쪽 입꼬리) 자동 탐지
- landmark 기반 affine transform으로 얼굴을 정면 기준 정렬
- `facenet-pytorch` 라이브러리로 1줄 사용 가능:
  ```python
  from facenet_pytorch import MTCNN
  mtcnn = MTCNN(image_size=224, margin=20)
  aligned = mtcnn(pil_img)   # 탐지 + 정렬 + crop + resize 한번에
  ```

### Haar Cascade (defalut) vs MTCNN 비교

| 항목 | Haar Cascade (defalut) | MTCNN (내 프로젝트) |
|---|---|---|
| 방법론 | 전통 ML (Viola-Jones, 2001) | 딥러닝 Cascade CNN |
| 탐지 정확도 | 낮음 (정면만 강함) | 높음 (다양한 각도) |
| 탐지 실패 시 | 원본 반환 (배경 포함) | Fallback 중앙 crop |
| Alignment | 없음 | 5-point landmark 정렬 |
| 속도 | 빠름 | 느림 (GPU 가속 가능) |

---

## 참고 논문 — 아이디어 차용

### POSTER++ (2023) — Landmark Feature 아이디어

> Mao et al., *"POSTER V2: A Simpler and Stronger Facial Expression Recognition Network"*  
> Pattern Recognition, 2024 / arXiv:2301.12149

- CNN feature와 landmark feature를 cross-attention으로 융합
- 전체 구조 구현은 복잡 → **DAN의 spatial attention**으로 유사 효과 달성

### TriCAFFNet (2024) — Multi-feature 아이디어

> PMC (Sensors, MDPI) 2024

- HOG + LBP + CNN + Landmark 4가지 feature 결합
- 아이디어 차용: augmentation에서 다양한 변환으로 학습 데이터 다양화

### Supervised Contrastive Learning (2023) — Label Smoothing 대안

> Khosla et al., *"Supervised Contrastive Learning"*, NeurIPS 2020  
> FER 적용: *"Contrastive Learning for FER"*, IJNS 2023

- 같은 클래스 샘플끼리 feature space에서 가깝게, 다른 클래스는 멀게
- DAN의 **Affinity Loss**가 유사한 역할을 수행 → 별도 구현 불필요

---

## 이 프로젝트의 적용 계획

### 직접 구현

| 요소 | 출처 | 구현 위치 |
|---|---|---|
| MTCNN face alignment | Zhang et al. 2016 | `my_preprocessing.ipynb` |
| EfficientNet-B0 backbone | Tan & Le, ICML 2019 | `my_model.ipynb` |
| Multi-head Spatial Attention | DAN, 2023 | `my_model.ipynb` |
| Affinity Loss | DAN, 2023 | `my_model.ipynb` |
| Partition Loss | DAN, 2023 | `my_model.ipynb` |
| WeightedRandomSampler | 표준 PyTorch | `my_model.ipynb` |
| CosineAnnealingLR | 표준 PyTorch | `my_model.ipynb` |
| RandomRotation / RandomAffine | 표준 torchvision | `my_preprocessing.ipynb` |

### 참고만

| 요소 | 출처 | 이유 |
|---|---|---|
| Window-based cross-attention | POSTER++ | 구현 복잡도 높음 |
| Tri-cross-attention | TriCAFFNet | 구현 복잡도 높음 |
| Uncertainty relabeling | SupCon+FER | 데이터 양이 적어 효과 미미 |

---

## 참고 링크

| 자료 | URL |
|---|---|
| DAN GitHub | https://github.com/yaoing/DAN |
| DAN arXiv | https://arxiv.org/abs/2109.16235 |
| EfficientNet 논문 | https://arxiv.org/abs/1905.11946 |
| POSTER++ arXiv | https://arxiv.org/abs/2301.12149 |
| facenet-pytorch (MTCNN) | https://github.com/timesler/facenet-pytorch |
| SupCon Loss | https://github.com/HobbitLong/SupContrast |
| pytorch-grad-cam | https://github.com/jacobgil/pytorch-grad-cam |
