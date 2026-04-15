# 선행 연구 분석

## 벤치마크 성능 요약 (AffectNet 기준)

| 모델 | 연도 | AffectNet-7 | AffectNet-8 | 핵심 기법 |
|---|---|---|---|---|
| POSTER++ | 2023 | **67.49%** | **63.77%** | Landmark + Image cross-attention |
| TriCAFFNet | 2024 | 67.40% | 63.49% | LBP+HOG+Landmark+CNN multi-feature |
| DAN | 2022 | 65.69% | 62.09% | Multi-head attention (얼굴 부위별) |
| SupCon+Relabeling | 2023 | - | 61.00% | Contrastive + uncertainty relabeling |
| ResNet-50 (baseline) | - | ~60% | ~58% | ImageNet pretrain + CE Loss |

> AffectNet-8 기준 현재 SOTA는 약 63~67% 수준. 인간 수준 정확도는 약 65% 내외로 알려져 있음.

---

## 논문별 상세 분석

---

### 1. DAN — *Distract Your Attention* (2022)

**출처:** PMC (Biomimetics) / GitHub: `yaoing/DAN`

**문제 인식:**
- 기존 모델은 얼굴 전체를 하나의 feature로 보는 경향
- 감정 표현은 눈, 입, 눈썹 등 부위별로 다르게 나타남

**핵심 구조:**

```
입력 이미지
    │
    ▼
Feature Clustering Network (FCN)
  └─ affinity loss로 클래스 간 feature 분리
    │
    ▼
Multi-head Attention Network (MAN)
  └─ 각 head가 얼굴의 다른 부위에 집중
    │
    ▼
Attention Fusion Network (AFN)
  └─ head들의 attention을 분산 후 fusion
    │
    ▼
분류 출력
```

**손실 함수:**

```
L_total = L_CE + λ1 * L_affinity + λ2 * L_partition
```

- `L_affinity`: Center Loss 계열. 같은 클래스 feature를 중심으로 모음
- `L_partition`: attention head 간 중복을 줄여 각자 다른 부위 담당하도록 유도

**성능:** AffectNet-7: 65.69%, AffectNet-8: 62.09%

**참고할 것:**
- multi-head attention 구조 자체보다 **affinity loss 구현**이 핵심
- GitHub 코드의 `convert_affectnet.py` — AffectNet 데이터 전처리 스크립트로 바로 사용 가능
- affinity loss = 이 프로젝트의 Center Loss와 동일한 역할

---

### 2. POSTER++ (2023)

**출처:** arXiv 2301.12149 → Pattern Recognition 2024 / GitHub: `zczcwh/POSTER`

**문제 인식:**
- CNN은 local feature에 강하지만 global context 약함
- ViT는 global에 강하지만 얼굴 landmark 정보를 활용 못 함
- 두 가지를 합쳐야 함

**핵심 구조:**

```
입력 이미지
    ├─ CNN Backbone → Image feature (multi-scale)
    └─ Landmark Detector → Landmark feature
              │
              ▼
    Window-based Cross-Attention
    (image feature ↔ landmark feature 상호 참조)
              │
              ▼
    분류 출력
```

POSTER 대비 POSTER++의 개선점:
- vanilla cross-attention → **window-based cross-attention** (연산량 절감)
- image-to-landmark 브랜치 제거 (단순화)
- pyramid 구조 → multi-scale feature 직접 fusion

**성능:** AffectNet-7: 67.49%, AffectNet-8: 63.77% (43.7M params, 8.4G FLOPs)

**참고할 것:**
- 전체 구조를 그대로 구현하기엔 복잡함
- **landmark feature를 auxiliary input으로 추가하는 아이디어** 자체를 차용
- 경량화 버전: landmark 좌표를 FC layer로 인코딩해서 image feature에 concat하는 방식으로 단순화 가능

---

### 3. TriCAFFNet (2024)

**출처:** PMC (Sensors, MDPI) — Open Access

**핵심 구조:**

```
입력 이미지
    ├─ CNN → Deep feature
    ├─ LBP → Texture feature
    ├─ HOG → Gradient feature
    └─ Landmark Detector → Landmark feature
              │
              ▼
    Tri-Cross-Attention Blocks
    (4가지 feature 쌍방향 상호 참조)
              │
              ▼
    분류 출력
```

**성능:** AffectNet-7: 67.40%, AffectNet-8: 63.49%

**참고할 것:**
- HOG, LBP는 OpenCV로 쉽게 추출 가능 → CNN feature에 **추가 채널**로 넣는 방식
- 구현 난이도 낮음에 비해 성능 기여도가 있음
- tri-cross-attention 자체보다 **multi-feature를 concat하는 아이디어**가 핵심

```python
# HOG feature 추출 예시
from skimage.feature import hog

hog_feature = hog(gray_img, orientations=9,
                  pixels_per_cell=(8,8),
                  cells_per_block=(2,2))
```

---

### 4. Contrastive Learning + Uncertainty-guided Relabeling (2023)

**출처:** PubMed (IJNS 2023) / GitHub: `xiaohu-run/fer_supCon`

**문제 인식:**
- AffectNet 레이블은 노이즈가 많음 (어노테이터 주관 차이)
- 단순 CE Loss는 노이즈 레이블에 그대로 fit됨

**핵심 아이디어:**

```
1단계: 노이즈 레이블 중 uncertainty 높은 샘플 식별
2단계: 식별된 샘플에 대해 KNN 기반 neighbor label로 relabeling
3단계: Supervised Contrastive Loss로 같은 클래스 feature를 가깝게
```

**SupCon Loss 개념:**

```
같은 클래스 샘플들 → feature space에서 가깝게
다른 클래스 샘플들 → feature space에서 멀게
CE Loss와 jointly 학습
```

```python
# SupCon Loss 구조 (개념)
# github.com/HobbitLong/SupContrast 참고
loss = supcon_loss(features, labels) + ce_loss(logits, labels)
```

**성능:** AffectNet-8: 61.00%

**참고할 것:**
- label smoothing보다 강력한 노이즈 처리
- SupCon은 단독으로도 쓸 수 있고 CE Loss에 더할 수도 있음
- GitHub 코드가 공개되어 있어 loss function만 가져다 쓰기 쉬움

---

## 이 프로젝트에 적용할 것 vs 참고만 할 것

### 직접 구현 (구현 난이도 낮음)

| 요소 | 출처 | 구현 방법 |
|---|---|---|
| Focal Loss | EfficientFace | 직접 작성 (~20줄) |
| Center Loss (affinity loss) | DAN | 직접 작성 (~15줄) |
| Label Smoothing | 표준 | `nn.CrossEntropyLoss(label_smoothing=0.1)` |
| Grad-CAM | pytorch-grad-cam | 라이브러리 사용 |
| AffectNet 전처리 | DAN | `convert_affectnet.py` 참고 |

### 아이디어 차용 (직접 구현 X)

| 요소 | 출처 | 차용 방식 |
|---|---|---|
| Landmark feature | POSTER++ | MediaPipe landmark 좌표 → FC 인코딩 후 concat |
| HOG feature | TriCAFFNet | skimage HOG → auxiliary feature |
| SupCon Loss | SupCon+Relabeling | `xiaohu-run/fer_supCon` 코드 가져다 사용 |

### 참고만 할 것 (구현 불필요)

- POSTER++ 전체 architecture (복잡도 높음)
- Tri-cross-attention 전체 구조
- Uncertainty-guided relabeling 전체 파이프라인

---

## 참고 링크

| 자료 | URL |
|---|---|
| DAN GitHub | https://github.com/yaoing/DAN |
| POSTER++ arXiv | https://arxiv.org/abs/2301.12149 |
| SupCon Loss | https://github.com/HobbitLong/SupContrast |
| SupCon FER | https://github.com/xiaohu-run/fer_supCon |
| pytorch-grad-cam | https://github.com/jacobgil/pytorch-grad-cam |
| facenet-pytorch (MTCNN) | https://github.com/timesler/facenet-pytorch |
