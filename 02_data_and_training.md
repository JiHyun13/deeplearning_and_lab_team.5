# 데이터 전처리 및 학습 전략

---

## 1. CMU-MOSI 데이터셋 특성

### 클래스 구성 (이진 분류)

| 클래스 | 샘플 수 | 비율 |
|---|---|---|
| Positive (sentiment ≥ 0) | 1,176 | 53.5% |
| Negative (sentiment < 0) | 1,023 | 46.5% |
| **합계** | **2,199** | 100% |

> 극단적 불균형은 아니지만, WeightedRandomSampler로 배치마다 정확히 1:1 비율을 보장한다.  
> 이유: defalut_2에서 클래스 편향이 baseline 55% 수준에 갇히는 원인 중 하나로 추정.

### defalut vs 내 프로젝트 데이터 흐름 비교

| 항목 | defalut_2 | 내 프로젝트 |
|---|---|---|
| 입력 | 중간 프레임 1장 (jpg) | 중간 프레임 1장 (동일) |
| Face Crop | Haar Cascade (구식, 오탐 多) | MTCNN (딥러닝 기반, landmark 포함) |
| Alignment | 없음 | 눈 좌표 기반 affine 정렬 |
| 정규화 | /255 만 | ImageNet mean/std |
| 증강 | Gaussian noise + Salt&Pepper | + Rotation / Shift / H-Flip |
| 클래스 균형 | 없음 | WeightedRandomSampler |

---

## 2. 전처리 전략 상세

### 2-1. Face Crop 고도화 — Haar Cascade → MTCNN

**defalut의 문제:**
```python
# defalut_1.ipynb — Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
if len(faces) > 0:
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]
return img   # ← 탐지 실패 시 원본 반환 (배경 포함)
```
- Haar Cascade는 정면 얼굴만 탐지, 각도 변화에 취약
- 탐지 실패 시 원본 이미지를 그대로 사용 → 배경 노이즈 포함
- 얼굴 정렬(alignment) 없음 → 같은 표정도 위치·각도가 다름

**내 전처리:**
```python
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image

mtcnn = MTCNN(
    image_size=224,
    margin=20,          # 얼굴 주변 여유 공간
    keep_all=False,     # 가장 큰 얼굴만
    device=device
)

def face_crop_mtcnn(img_pil):
    """
    MTCNN: 탐지 + 5-point landmark 기반 affine 정렬 + crop + resize 한번에
    반환: 224×224 tensor (정렬된 얼굴) or None
    """
    face_tensor = mtcnn(img_pil)  # (3, 224, 224) float32, [-1, 1]
    return face_tensor

def face_crop_with_fallback(img_pil, fallback_size=(224, 224)):
    """
    MTCNN 탐지 실패 시 → 중앙 crop fallback (원본 반환 X)
    """
    face = mtcnn(img_pil)
    if face is not None:
        return face, True   # (tensor, is_detected)
    # fallback: 이미지 중앙을 얼굴로 가정
    w, h = img_pil.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    cropped = img_pil.crop((left, top, left + min_dim, top + min_dim))
    cropped = cropped.resize(fallback_size)
    return cropped, False
```

**MTCNN 장점:**
- DeepLearning 기반 → 각도·조명 변화에 강함
- 5-point landmark (눈 2, 코 1, 입꼬리 2) → 자동 affine 정렬
- 탐지률: Haar Cascade 대비 현저히 높음
- 논문: Zhang et al., *"Joint Face Detection and Alignment using Multi-task Cascaded CNNs"*, IEEE SPL 2016

---

### 2-2. Augmentation 강화

**defalut의 문제:**
```python
# defalut_2.ipynb — 노이즈만
def apply_noise_augmentation(img):
    if np.random.random() > 0.8:
        img = aug_gaussian_noise(img, std=0.03)
    if np.random.random() > 0.8:
        img = aug_salt_pepper_noise(img, amount=0.01)
    return img
```
- Gaussian noise + Salt&Pepper만 적용
- 공간적 변환(회전, 이동) 전혀 없음 → 위치 과적합 발생

**내 전처리 — torchvision transforms 기반:**
```python
from torchvision import transforms

train_transform = transforms.Compose([
    # [1] 공간 변환 (새로 추가)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),                          # ±15° 회전
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),   # 상하좌우 10% 이동 (Shifting)
        scale=(0.9, 1.1)        # ±10% 스케일
    ),
    # [2] 색상 변환 (새로 추가)
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=0.2, hue=0.05
    ),
    # [3] 노이즈 (defalut 유지 + 개선)
    transforms.ToTensor(),      # [0,1] 변환
    AddGaussianNoise(std=0.02),  # 커스텀 transform
    # [4] ImageNet 정규화 (defalut에서 누락되었던 항목)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    # [5] Random Erasing (부분 가림 → occluded face 강건성)
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 커스텀 Gaussian Noise Transform
class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

# Salt & Pepper도 커스텀 Transform으로 구현
class AddSaltPepperNoise:
    def __init__(self, amount=0.01):
        self.amount = amount
    def __call__(self, tensor):
        noisy = tensor.clone()
        num_pixels = tensor.numel() // 3
        # Salt
        salt_idx = torch.randint(0, num_pixels, (int(num_pixels * self.amount),))
        noisy.view(3, -1)[:, salt_idx] = 1.0
        # Pepper
        pepper_idx = torch.randint(0, num_pixels, (int(num_pixels * self.amount),))
        noisy.view(3, -1)[:, pepper_idx] = 0.0
        return noisy
```

**증강 전략 비교:**

| 증강 방법 | defalut | 내 프로젝트 | 목적 |
|---|---|---|---|
| Horizontal Flip | ✗ | ✓ (p=0.5) | 좌우 대칭 robustness |
| Rotation (±15°) | ✗ | ✓ | 각도 변화 robustness |
| Shift (10%) | ✗ | ✓ | 위치 변화 robustness |
| Scale (±10%) | ✗ | ✓ | 거리 변화 robustness |
| Color Jitter | ✗ | ✓ | 조명·색상 robustness |
| Gaussian Noise | ✓ | ✓ | 센서 노이즈 |
| Salt & Pepper | ✓ | ✓ | 결측 픽셀 |
| Random Erasing | ✗ | ✓ | 부분 가림 상황 |
| ImageNet Normalize | △ (AlexNet만) | ✓ (전체) | pretrained 모델과 입력 분포 일치 |

---

### 2-3. 배치 내 클래스 균형화 — WeightedRandomSampler

```python
from torch.utils.data import WeightedRandomSampler

# 클래스별 가중치 계산
class_counts = [neg_count, pos_count]   # [1023, 1176]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

# 각 샘플에 해당 클래스 weight 부여
sample_weights = class_weights[y_train]  # shape: (N_train,)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# DataLoader — shuffle=False (sampler와 함께 쓸 때)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,         # shuffle 대신 sampler 사용
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED)
)
```

**효과:**
- 매 배치마다 Positive : Negative ≈ 1:1 보장
- 소수 클래스가 과소 학습되는 문제 방지
- defalut_2에서 test acc가 55~60%에 갇혔던 원인 중 하나 해소

---

## 3. 모델 학습 전략

### 3-1. EfficientNet-B0 + DAN Multi-head Attention

```python
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class AffinityLoss(nn.Module):
    """
    DAN의 Affinity Loss: 같은 클래스 feature를 클래스 중심(center)으로 모음
    Center Loss의 변형 — 논문: Wen et al., Biomimetics 2023
    """
    def __init__(self, num_classes=2, feat_dim=1280):
        super().__init__()
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim)
        )
    def forward(self, features, labels):
        batch_centers = self.centers[labels]           # (B, feat_dim)
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean()
        return loss

class PartitionLoss(nn.Module):
    """
    DAN의 Partition Loss: head 간 attention 중복 억제
    각 head가 서로 다른 얼굴 부위를 담당하도록 유도
    """
    def forward(self, attention_maps):
        # attention_maps: (B, num_heads, H*W)
        # head 간 코사인 유사도를 최소화
        B, num_heads, HW = attention_maps.shape
        loss = 0
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                sim = (attention_maps[:, i] * attention_maps[:, j]).sum(dim=1)
                loss += sim.mean()
        return loss / (num_heads * (num_heads - 1) / 2)

class MultiHeadSpatialAttention(nn.Module):
    """DAN-style 4-head Spatial Attention"""
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        # 각 head: feature map → attention weight map
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (B, C, H, W)
        attention_maps = []
        weighted_features = []

        for head in self.heads:
            attn = head(x)                        # (B, 1, H, W)
            attended = x * attn                   # (B, C, H, W)
            pooled = self.gap(attended).flatten(1) # (B, C)
            attention_maps.append(attn.flatten(2))  # (B, 1, H*W)
            weighted_features.append(pooled)

        # head 결과 평균 fusion
        fused = torch.stack(weighted_features, dim=1).mean(dim=1)  # (B, C)
        attn_stack = torch.cat(attention_maps, dim=1)              # (B, num_heads, H*W)
        return fused, attn_stack

class DANEfficientNet(nn.Module):
    """
    EfficientNet-B0 Backbone + DAN Multi-head Spatial Attention
    """
    def __init__(self, num_classes=2, num_heads=4, dropout=0.5):
        super().__init__()

        # EfficientNet-B0 feature extractor (pretrained)
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # features: Conv stem + MBConv blocks (output: B, 1280, 7, 7)
        self.features = backbone.features

        # DAN Multi-head Attention
        self.attention = MultiHeadSpatialAttention(in_channels=1280, num_heads=num_heads)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        feat_map = self.features(x)             # (B, 1280, 7, 7)
        features, attn_maps = self.attention(feat_map)  # (B, 1280), (B, 4, 49)
        logits = self.fc(self.dropout(features))
        return logits, features, attn_maps
```

### 3-2. 손실 함수 조합

```python
ce_loss      = nn.CrossEntropyLoss(label_smoothing=0.1)
affinity_loss = AffinityLoss(num_classes=2, feat_dim=1280).to(device)
partition_loss = PartitionLoss()

# 학습 루프 내
logits, features, attn_maps = model(imgs)
loss = (ce_loss(logits, labels)
      + 0.1 * affinity_loss(features, labels)   # λ1 = 0.1
      + 0.01 * partition_loss(attn_maps))        # λ2 = 0.01
```

**각 loss의 역할:**

| Loss | 역할 | defalut 대비 |
|---|---|---|
| CE + Label Smoothing | 기본 분류 + 과적합 억제 | defalut는 CE만 |
| Affinity Loss | 같은 클래스 feature 뭉치게 → 일반화 ↑ | 없음 |
| Partition Loss | head들이 중복 없이 다른 부위 커버 | 없음 |

### 3-3. Optimizer & Scheduler

```python
# AdamW: weight decay 포함 (defalut는 Adam)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# CosineAnnealingLR: 학습률 자연 감소 (defalut는 고정 lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,
    eta_min=1e-6
)

# Gradient Clipping: 학습 안정화
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3-4. Fine-tuning 전략 (2단계)

```
[1단계] Feature Extractor 동결 — FC + Attention만 학습 (5 epochs)
    → pretrained 특징 보존, 분류기 안정화

[2단계] 전체 Unfreeze — 전체 파인튜닝 (25 epochs)
    → 낮은 lr(1e-4)로 세밀하게 조정
```

```python
# 1단계: backbone freeze
for param in model.features.parameters():
    param.requires_grad = False

# 2단계: unfreeze
for param in model.features.parameters():
    param.requires_grad = True
```

---

## 4. Ablation Study 계획

각 개선 요소의 **개별 기여도**를 측정한다.

| 실험 | MTCNN | Rotation/Shift | WeightedSampler | LR Scheduler | 예상 효과 |
|---|---|---|---|---|---|
| Baseline (defalut_2 AlexNet) | ✗ | ✗ | ✗ | ✗ | 64.1% (기준) |
| + MTCNN | ✓ | ✗ | ✗ | ✗ | 전처리 기여도 측정 |
| + Augmentation | ✓ | ✓ | ✗ | ✗ | 공간 증강 기여도 |
| + Sampler | ✓ | ✓ | ✓ | ✗ | 균형화 기여도 |
| **Full (내 모델)** | ✓ | ✓ | ✓ | ✓ | 목표 |

---

## 5. 평가 지표

이진 분류이지만 클래스 불균형을 고려하여 다각도로 평가한다.

| 지표 | 이유 |
|---|---|
| Accuracy | 주 지표 (defalut와 동일 기준 비교) |
| F1-score (Macro) | Positive/Negative 균형 평가 |
| Confusion Matrix | 오분류 패턴 파악 |
| Train-Test 갭 | 과적합 여부 확인 (defalut: AlexNet 34%p 갭) |
| Attention Map 시각화 | 모델이 실제로 얼굴 부위에 집중하는지 확인 |
