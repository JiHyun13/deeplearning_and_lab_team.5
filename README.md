# 딥러닝실습 5조 — 실험 결과 시각화 모음

CMU-MOSI 멀티모달 감성 분석 프로젝트의 각 팀원 실험 결과 이미지를 모아둔 브랜치입니다.

---

## 폴더 구조

```
jihyun/          이지현 (24011917) — 영상 인코더 + Cross-Attention Fusion
hyunseo/         최현서 (24011834) — 텍스트+오디오 Middle Fusion (MLP)
jeein/           유지인 (22012137) — 오디오 Graph Fusion
sooyeon/	    이수연(22010696) — 텍스트 + Self-Attention Fusion
```

---

## jihyun/ — 영상 모달리티 & Cross-Attention Fusion

### video_encoding/ — 영상 인코더 모델 비교

| 파일 | 설명 |
|------|------|
| `exp1_model_comparison.png` | EfficientNet-B0 / ResNet-18 / ConvNeXt-Tiny 세 모델의 Stage1·Stage2 학습 곡선 비교 |
| `exp2_aug_comparison.png` | 최고 모델의 데이터 증강 있음 vs 없음 Stage2 학습 곡선 |
| `exp2_aug_bar.png` | 증강 유/무 Test Accuracy 막대 그래프 |
| `video_feature_dist.png` | 추출된 768-dim · 256-dim 피처 값 분포 히스토그램 |

### fusion/ — Cross-Attention Fusion 실험 결과 (6가지 조건)

각 폴더는 `fusion_training_curves.png` · `fusion_confusion_matrix.png` · `fusion_ablation.png` 포함.  
5_768dim_basic에는 Attention 분석 시각화 추가 수록.

| 폴더 | 조건 설명 |
|------|-----------|
| `1_256dim_basic/` | 256-dim 피처, 기본 학습 (증강 없음) |
| `2_256dim_freeze/` | 256-dim 피처, 백본 동결 학습 |
| `3_256dim_basic_aug/` | 256-dim 피처, 기본 증강 |
| `4_256dim_full_aug/` | 256-dim 피처, 풀 증강 |
| `5_768dim_basic/` | 768-dim 피처, 기본 학습 + **Attention 분석 시각화** |
| `6_768dim_freeze/` | 768-dim 피처, 백본 동결 학습 |

#### 5_768dim_basic/ 추가 시각화

| 파일 | 설명 |
|------|------|
| `attn_headwise_heatmap.png` | 헤드별 Attention 가중치 히트맵 |
| `attn_inspector_correct.png` | 정답 샘플의 Attention 분포 |
| `attn_inspector_wrong.png` | 오답 샘플의 Attention 분포 |
| `attn_posneg_distribution.png` | 긍정/부정 클래스별 Attention 분포 비교 |
| `attn_radar_comparison.png` | 헤드별 Attention 기여도 레이더 차트 |
| `error_faces.png` | 오분류된 얼굴 샘플 시각화 |
| `fusion_attn_correct_wrong.png` | 정답/오답 샘플 Attention 비교 |
| `fusion_attn_mean.png` | 평균 Attention 가중치 |
| `fusion_attn_posneg.png` | 긍정/부정 샘플 Attention 비교 |

---

## hyunseo/ — 텍스트+오디오 Middle Fusion (MLP)

최현서 팀원의 MLP 기반 Middle Fusion 실험. 각 폴더에 `Loss.png` · `ablation.png` · `confusion_matrix.png` 포함.

| 폴더 | 조건 설명 |
|------|-----------|
| `1_MLP_256_text_basic/` | 256-dim, 텍스트 기본(동결 없음) |
| `2_MLP_256_aug_freeze6/` | 256-dim, 텍스트 증강 + 6레이어 동결 + EarlyStopping |
| `3_MLP_768_text_aug_freeze6/` | 768-dim, 텍스트 증강 + 6레이어 동결 + EarlyStopping |
| `4_MLP_768_text_basic/` | 768-dim, 텍스트 기본 |
| `5_MLP_256_text_freeze_audio_aug/` | 256-dim, 텍스트 동결 + 오디오 증강 |
| `6_MLP_256_text_basic_audio_aug/` | 256-dim, 텍스트 기본 + 오디오 증강 |

---

## jeein/ — 오디오 Graph Fusion

유지인 팀원의 Graph Neural Network 기반 오디오 Fusion 실험.

### graph_fusion/
| 파일 | 설명 |
|------|------|
| `graph_fusion.png` | Graph Fusion 모델 학습 결과 (1차) |
| `graph_fusion2.png` | Graph Fusion 모델 학습 결과 (2차) |

### graph_result/ — 조건별 Graph Fusion 실험 (6가지 + 재실험)

각 폴더에 `*-1_training_curves.png` · `*-2_confusion_matrix.png` · `*-3_ablation.png` 포함.

| 폴더 | 조건 설명 |
|------|-----------|
| `1_256_text_basic_audio_basic/` | 256-dim 텍스트 기본 + 오디오 기본 |
| `2_256_text_freeze_audio_basic/` | 256-dim 텍스트 동결 + 오디오 기본 |
| `3_768_text_basic_audio_basic/` | 768-dim 텍스트 기본 + 오디오 기본 |
| `4_768_text_freeze_audio_basic/` | 768-dim 텍스트 동결 + 오디오 기본 |
| `5_256_text_freeze_audio_aug/` | 256-dim 텍스트 동결 + 오디오 증강 |
| `6_256_text_basic_audio_aug/` | 256-dim 텍스트 기본 + 오디오 증강 |
| `re_result/` | 최종 재실험 결과 (1~5.png) |
---

## sooyeon/ — 텍스트 + Self-Attention Fusion

###BERT/ — BERT 모델 조건별 학습 결과 비교

| 파일 | 설명 |
|------|------|
| `기본모델+조기종료(256차원).png` | 256-dim 피처, 기본 BERT 모델의 Loss, Accuracy, F1-Score 학습 곡선 (조기 종료 적용) |
| `기본모델 + 조기종료 (768차원).png` | 768-dim 피처, 기본 BERT 모델의 Loss, Accuracy, F1-Score 학습 곡선 (조기 종료 적용) |
| `증강 + 동결6 + 조기종료(256).png` | 256-dim 피처, 데이터 증강(Augmentation) 및 하위 6개 레이어 동결(Freeze) 적용 시의 학습 곡선 |
| `증강 + 동결6 + 조기종료(768).png` | 768-dim 피처, 데이터 증강(Augmentation) 및 하위 6개 레이어 동결(Freeze) 적용 시의 학습 곡선 |

### self_fusion/ — Self-Attention 기반 모달리티 융합 실험 결과

각 폴더에는 모델 학습 및 평가 결과를 분석하기 위한 4개의 시각화 파일이 공통으로 포함되어 있습니다.

### 폴더명 및 조건 설명

| 폴더 | 조건 설명 |
|------|-----------|
| `기본+256/` | 256-dim 피처, 기본 학습 (증강 없음) |
| `기본+756/` | 756-dim 피처, 기본 학습 (증강 없음, ※ 768의 오타로 추정됨) |
| `256 텍스트만 증강/` | 256-dim 피처, 텍스트 데이터에만 증강 기법 적용 |
| `256 오디오만 증강/` | 256-dim 피처, 오디오 데이터에만 증강 기법 적용 |
| `256 오디오, 텍스트 둘다 증강/` | 256-dim 피처, 오디오 및 텍스트 데이터 모두 증강 기법 적용 |
| `768 증강/` | 768-dim 피처, 데이터 증강 적용 |

### 공통 포함 시각화 파일

| 파일 | 설명 |
|------|------|
| `loss.png` | Epoch에 따른 Loss, Accuracy, F1-Score 학습 곡선 및 조기 종료(Early Stopping) 지점 시각화 |
| `Classification Report.png` | 모델의 긍정/부정(Positive/Negative) 예측 성능을 보여주는 Confusion Matrix |
| `Attention.png` | 텍스트, 오디오, 비디오 모달리티 간의 Fusion Self-Attention 가중치 히트맵 |
| `Ablation Study.png` | 단일 모달리티 및 결합 조건에 따른 Test Accuracy 비교 막대 그래프 (Modality Contribution) |
