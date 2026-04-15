# 데이터 전처리 및 학습 전략

## 1. AffectNet 데이터셋 특성

### 클래스 구성

| 클래스 | 학습 샘플 수 (8 cls 기준) | 비율 |
|---|---|---|
| Neutral | ~74,874 | 26% |
| Happy | ~134,415 | 47% |
| Sad | ~25,459 | 9% |
| Surprise | ~14,090 | 5% |
| Fear | ~6,378 | 2% |
| Disgust | ~3,803 | 1% |
| Anger | ~24,882 | 9% |
| Contempt | ~3,750 | 1% |

> **핵심 문제:** Happy와 Neutral이 전체의 73%를 차지. Fear, Disgust, Contempt는 극소수.

### 주요 난제
- **클래스 불균형** — 다수 클래스 과적합, 소수 클래스 성능 급락
- **노이즈 레이블** — 감정은 주관적이라 어노테이터 간 불일치가 많음
- **intra-class variation** — 같은 감정도 표현 방식이 사람마다 다름
- **inter-class similarity** — fear/surprise, disgust/anger는 시각적으로 유사

---

## 2. 전처리 전략

### 2.1 얼굴 정렬 (Face Alignment)

단순 crop 대신 **눈 위치 기준으로 얼굴을 정렬**한다.
표정 feature가 위치 변동 없이 일관되게 추출된다.

```python
# MTCNN 또는 MediaPipe 사용
# 눈 랜드마크를 기준으로 affine transform 적용
from facenet_pytorch import MTCNN

mtcnn = MTCNN(image_size=224, margin=20)
aligned = mtcnn(img)  # 정렬 + crop + resize 한번에
```

적용 여부에 따른 성능 차이를 ablation study에서 확인할 것.

### 2.2 Augmentation

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.RandomErasing(p=0.3),   # 부분 가림 robustness
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

### 2.3 클래스 불균형 처리

**두 가지 방법을 동시에 적용한다.**

**① Weighted Random Sampler** — 배치 내 클래스 비율 균형화

```python
from torch.utils.data import WeightedRandomSampler

class_counts = [74874, 134415, 25459, 14090, 6378, 3803, 24882, 3750]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[labels]  # 각 샘플의 weight

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
```

**② Focal Loss** — 쉬운 샘플의 gradient를 억제, 어려운 소수 클래스에 집중

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 클래스별 weight tensor

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()
```

### 2.4 노이즈 레이블 처리

최소한 **Label Smoothing**은 적용한다.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

여유가 있으면 SupCon Loss와 조합 (03_related_works.md 참고).

---

## 3. 모델 학습 전략

### 3.1 Backbone 선택

| 옵션 | 파라미터 | AffectNet-8 참고 성능 | 비고 |
|---|---|---|---|
| ResNet-50 | 25M | ~60% | 베이스라인 |
| EfficientNet-B3 | 12M | ~62% | 효율적 |
| Swin-T | 28M | ~63% | Transformer 계열 |

**권장: EfficientNet-B3**
- 파라미터 대비 성능이 높음
- RTX 4060 환경에서 batch 64 이상 가능
- `torchvision.models.efficientnet_b3(pretrained=True)` 사용

### 3.2 사전학습 모델 전략

ImageNet pretrain만 쓰는 것보다, **얼굴 도메인 사전학습**을 거친 모델이 훨씬 유리하다.

```
ImageNet pretrain
    → VGGFace2 pretrain (선택)
    → AffectNet fine-tune
```

VGGFace2 사전학습 가중치:
- `timm` 라이브러리에서 일부 제공
- 또는 MS-Celeb pretrained ResNet-50 사용 가능

### 3.3 손실 함수 조합

**베이스라인:** CE Loss + Label Smoothing

**개선안:** CE Loss + Center Loss

```python
class CenterLoss(nn.Module):
    """같은 클래스 feature를 클래스 중심으로 모아주는 loss"""
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean()
        return loss

# 학습 시
loss = ce_loss(logits, labels) + 0.003 * center_loss(features, labels)
```

### 3.4 Learning Rate 스케줄러

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-6
)
```

### 3.5 학습 루프 핵심

```python
for epoch in range(num_epochs):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        optimizer.zero_grad()
        features, logits = model(imgs)  # feature와 logit 동시 출력

        loss = ce_loss(logits, labels) + 0.003 * center_loss(features, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()
    evaluate(model, val_loader)
```

---

## 4. 평가 지표

AffectNet은 클래스 불균형이 심하므로 **Accuracy 단독으로 평가하면 안 된다.**

| 지표 | 이유 |
|---|---|
| Weighted F1 | 클래스 불균형 고려한 주 지표 |
| Per-class Accuracy | 소수 클래스 성능 확인 |
| Confusion Matrix | 혼동 패턴 파악 (fear↔surprise 등) |
| Accuracy | 참고용 |

---

## 5. Ablation Study 계획

아래 조합을 순서대로 실험해서 각 요소의 기여도를 측정한다.

| 실험 | Face Align | Focal Loss | Center Loss | 예상 효과 |
|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | 기준선 |
| + Focal | ✗ | ✓ | ✗ | 소수 클래스 성능 ↑ |
| + Align | ✓ | ✓ | ✗ | 전체 성능 ↑ |
| + Center | ✓ | ✓ | ✓ | feature 분리 ↑ |
