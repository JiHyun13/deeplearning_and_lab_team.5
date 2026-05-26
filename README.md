# 딥러닝실습 5조 — Multimodal Sentiment Analysis

> CMU-MOSI 데이터셋 감성 분석 (긍정 / 부정)  
> **Video + Audio + Text** 각자 256-dim 피처 추출 → fusion


---

## image_fixed.ipynb (영상 모달리티)

담당: 24011917 이지현

### 파이프라인

```
mp4 영상
  ↓  중간 프레임 1장 추출 (cv2)
  ↓  640×480 리사이즈 → MTCNN 얼굴 탐지 → 224×224 크롭
     (탐지 실패 시 전체 이미지 리사이즈로 fallback)
  ↓  pickle 캐싱 (video_preprocessed.pkl)
  ↓  증강 (flip / rotate / colorjitter / GaussianNoise 추가) — train에 증강 적용
DataLoader
  ↓
VideoEncoder (EfficientNet-B0 or ResNet-18)
  ↓  Linear → BatchNorm → ReLU → Dropout(0.3)
256-dim 피처
  ├─ feature_only=False  → 분류 logit  (단독 학습·평가용)
  └─ feature_only=True   → 256-dim 벡터 (멀티모달 fusion 입력)
```

---

### Part 요약

| Part | 내용 |
|------|------|
| 1. 셋업 | 라이브러리, 시드 고정, 경로·상수 (`FEATURE_DIM=256`) |
| 2. 전처리 | 중간 프레임 추출 → MTCNN(배치) → pickle 저장. 이미 pkl이 있으면 바로 로드 |
| 3. DataLoader | 64% train / 16% val / 20% test stratified split |
| 4. 모델 | `VideoEncoder` — EfficientNet-B0(1280→256) / ResNet-18(512→256) |
| 5. 랜덤 서치 | lr × epoch × optimizer × weight_decay 랜덤 5샘플 탐색 (val 기준) |
| 6. 최종 학습 | 최적 config로 재학습 + CosineAnnealingLR + 학습 곡선 저장 |
| 7. 피처 추출 | 베스트 모델로 전체 데이터 256-dim 추출 → `video_features_256.pkl` |

### 최종 결과

| 모델 | val acc | test acc |
|------|---------|----------|
| EfficientNet-B0 | 0.5426 | **0.5227** |
| ResNet-18 | 0.5966 | 0.4341 |

→ 실습 때 Video만 53%정도 나왔었음. 비슷한 수치여서 랜덤 서치를 더 돌려봐야할거같아욥..

---

## 팀원 참고! — Fusion 통합 방법입니다
비디오가 엄청 오래걸려서 ㅠㅠ 실습실에서 같이 하는거 아니면
피클로 다 만들어뒀으니 가져다 쓰는 방식으로 퓨전해야할거같아요

### 1. 피처 불러오기

Part 7 실행 후 `video_features_256.pkl` 생성됨.

```python
import pickle, torch

with open('/path/to/video_features_256.pkl', 'rb') as f:
    data = pickle.load(f)

video_feats  = data['features']   # shape: (2199, 256)  torch.Tensor
video_labels = data['labels']     # shape: (2199,)
```

audio, text도 필요하다면 동일한 방식으로 256-dim 피처를 저장해두면 됩니다.

### 2. Fusion 예시

```python
import torch.nn as nn

# 세 모달리티 concat → MLP
fused = torch.cat([video_feats, audio_feats, text_feats], dim=1)  # (N, 768)

mlp = nn.Sequential(
    nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 2)
)
```

### 3. 주요 파라미터 수정 위치

| 수정하고 싶은 것 | 위치 |
|---|---|
| 피처 차원 변경 | **Part 1** `FEATURE_DIM = 256` |
| backbone 변경 | **Part 4** `VideoEncoder(backbone='resnet18')` |
| 탐색 범위·횟수 | **Part 5** `PARAM_SPACE`, `N_SEARCH` |

---

## Figma

Figma 선행연구조사: https://www.figma.com/design/EN7aqVx7fGlpnr40zESsVR/%EB%94%A5%EC%8B%A4-%EC%84%A0%ED%96%89%EC%97%B0%EA%B5%AC%EC%A1%B0%EC%82%AC%EA%B8%B0%EB%B0%98-%EA%B8%B0%ED%9A%8D?node-id=0-1&t=zNJREpARBSwNf59X-1
