# 얼굴 감정 분류 프로젝트 기획서

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 과목 | 중간고사 프로젝트 |
| 주제 | 이미지 기반 얼굴 감정 분류 (Facial Expression Recognition) |
| 데이터셋 | AffectNet |
| 목표 | 단순 분류 정확도 향상을 넘어, 해석가능성과 데이터 품질 처리까지 포함한 완결된 파이프라인 구성 |

---

## 차별화 전략 요약

모든 팀이 동일한 데이터로 감정 분류를 수행하는 조건에서, 다음 세 축으로 차별점을 만든다.

```
1. 데이터 전처리  →  얼굴 정렬 + 클래스 불균형 처리 + 노이즈 레이블 대응
2. 학습 전략     →  Face-pretrained backbone + 다중 손실 함수 + Grad-CAM
3. 알파 (추후)  →  Valence-Arousal 회귀 확장 or 공정성(Fairness) 분석
```

현재 문서는 **1번과 2번**에 집중한다.

---

## 전체 파이프라인

```
원본 이미지
    │
    ▼
[전처리]
  ├─ MTCNN/MediaPipe 얼굴 정렬 (Face Alignment)
  ├─ Resize → 224×224
  ├─ Normalize (ImageNet 기준)
  └─ Augmentation (flip, color jitter, random erasing)
    │
    ▼
[클래스 불균형 처리]
  ├─ Weighted Random Sampler
  └─ Focal Loss / Class-balanced CE
    │
    ▼
[모델 학습]
  ├─ Backbone: EfficientNet-B3 또는 ResNet-50 (Face-pretrained)
  ├─ Loss: CE + Center Loss (or SupCon Loss)
  └─ Scheduler: CosineAnnealingLR
    │
    ▼
[평가 및 해석]
  ├─ Accuracy / Weighted F1 / Confusion Matrix
  └─ Grad-CAM 시각화
```

---

## 파일 구성

```
project/
├── data/
│   ├── affectnet/          # 원본 데이터
│   └── processed/          # 전처리 완료 데이터
├── src/
│   ├── dataset.py          # Dataset 클래스, 전처리 로직
│   ├── model.py            # 모델 정의
│   ├── loss.py             # 손실 함수 (Focal, Center, SupCon)
│   ├── train.py            # 학습 루프
│   ├── evaluate.py         # 평가 및 Grad-CAM
│   └── utils.py            # 공통 유틸
├── configs/
│   └── config.yaml         # 하이퍼파라미터
├── notebooks/
│   └── eda.ipynb           # 데이터 탐색
└── README.md
```

---

## 일정 계획 (예시)

| 단계 | 작업 | 비고 |
|---|---|---|
| 1주차 | EDA, 데이터 전처리 파이프라인 구축 | 클래스 분포 확인 필수 |
| 2주차 | 베이스라인 모델 학습 (ResNet-50 + CE Loss) | 성능 기준선 확보 |
| 3주차 | 개선 실험 (Focal Loss, Center Loss, Face Alignment) | ablation study |
| 4주차 | Grad-CAM 시각화, 결과 정리, 발표 준비 | |
