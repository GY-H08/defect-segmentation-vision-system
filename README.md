# defect-segmentation-vision-system
Industrial defect segmentation system using custom UNet3+ASPP architecture. Real-world vision inspection project in manufacturing environment# Real-World Vision Inspection System Improvement
## Custom UNet3+ Architecture Design & Defect Segmentation

> **더그리트(The Greet)** 플랫폼팀 인턴십 실무 프로젝트  
> 재사용 컵 외관 검사 비전 시스템의 세그멘테이션 모델 개선

---

## 📌 프로젝트 개요

재사용 컵 세척 후 고객에게 제공하기 전 최종 외관 검사를 수행하는 **실운영 산업용 비전 시스템**의 성능 개선 프로젝트.

현장(공장) 방문을 통해 전체 시스템 아키텍처와 운영 환경을 직접 파악하고, 데이터 특성 분석부터 기존 모델 구조 분석, 커스텀 모델 아키텍처 설계, 학습, 결과 분석까지 전 과정을 수행하였다.

---

## 🏭 현장 파악 및 시스템 아키텍처

### 현장 방문 내용
- 컵 반환 기계 실제 운영 환경 직접 확인
- Basler 산업용 카메라 6대 구성 및 FastAPI 백엔드 구조 파악
- 라벨링 데이터 구조 및 클래스 정의 확인
- 현행 추론 파이프라인(model_object.py) 코드 분석

### 카메라 구성

```
컵 반환 기계
  ├── 상단 카메라 (1대)  →  컵 내부 촬영        →  본 프로젝트 대상
  ├── 측면 카메라 (4대)  →  컵 외부 360도 촬영  →  본 프로젝트 대상 (별도 모델)
  └── 하단 카메라 (1대)  →  QR코드 / 레터링 확인 (QR 인식)
```

### 기존 운영 모델 문제점

| 문제 | 내용 |
|------|------|
| 운영 모델 | SMP UNet++ + EfficientNet-B4 |
| Hard_Object 오검률 | 약 50% |
| 원인 | 단일 스케일 특징 추출로 크기가 불규칙한 이물질 경계 구분 한계 |

---

## 🗂️ 데이터셋 분석

| 항목 | 내용 |
|------|------|
| 이미지 타입 | 흑백 (1채널), 1200×1200 |
| 검출 대상 클래스 | Water_Stain, Inner_Defect, Hard_Object, Background |
| 결함 픽셀 비율 | 전체 픽셀의 약 0.4%(평균) |
| 클래스 불균형 | Hard_Object : Background = 약 1 : 250 |
| 실험용 데이터 | 191쌍 (train 153 / val 38) |

> 세척 후 최종 검사 특성상 결함 픽셀 비율이 극히 낮은 고난이도 클래스 불균형 문제

---

## 🔬 모델 탐색 및 실험

### 모델 선정 과정
기존 UNet++ 구조 분석 → UNet3+(풀스케일 스킵 연결)로의 교체 근거 도출 → 단계적 실험 진행

### Experiment A — UNet3+ (v1)

| 항목 | 내용 |
|------|------|
| 모델 | UNet3+ + EfficientNet-B4 |
| 파라미터 | 23,702,833 |
| 인코더 초기화 | ImageNet 사전학습 |
| Loss | HybridLoss (Focal γ=2, w=0.5 + Dice w=0.5) + CLASS_WEIGHTS |
| Optimizer | AdamW + CosineAnnealingLR |
| Epoch | 300 |

**결과 (Best checkpoint — epoch 63)**

| 클래스 | IoU | Recall | Precision |
|--------|-----|--------|-----------|
| Background | 0.9883 | 0.9945 | 0.9937 |
| Water_Stain | 0.4622 | 0.6350 | 0.6294 |
| Inner_Defect | 0.4221 | 0.5717 | 0.6173 |
| Hard_Object | 0.2945 | 0.4384 | 0.4728 |
| **mIoU_defect** | **0.3929** | - | - |

> epoch 63 이후 과적합으로 mIoU 지속 하락 (epoch 300: 0.3578)

---

### Experiment B — UNet3+ + MSCA + SWA (v2)

| 항목 | 내용 |
|------|------|
| 모델 | UNet3+ + MSCA + EfficientNet-B4 |
| 파라미터 | 32,513,393 (+37%) |
| 인코더 초기화 | 실험 A best checkpoint (epoch 63) warm start |
| Loss / Optimizer | 실험 A와 동일 |
| SWA | epoch 225~300 적용 |

**결과 (SWA_FINAL)**

| 클래스 | IoU | Recall |
|--------|-----|--------|
| Background | 0.9886 | 0.9951 |
| Water_Stain | 0.4438 | 0.5783 |
| Inner_Defect | 0.4239 | 0.5666 |
| Hard_Object | 0.1876 | 0.2608 |
| **mIoU_defect** | **0.3518** | - |
| **defect_image_FNR** | **0.000** | ✅ |

---

### 실험 A vs B 비교 분석

| 항목 | 실험 A | 실험 B | 비고 |
|------|--------|--------|------|
| mIoU_defect | **0.3929** | 0.3518 | A 우위 |
| Hard_Object Recall | **0.4384** | 0.2608 | A 우위 |
| defect_image_FNR | 미집계 | **0.000** | B 우위 (안전성) |
| MSCA 효과 | - | 기대 이하 | 파라미터 대비 효과 없음 |

**핵심 인사이트:**
- MSCA는 파라미터 37% 증가 대비 성능 향상 없음 → 소량 데이터 환경에서 과파라미터 문제
- SWA는 defect_image_FNR 0 달성 → 운영 안전성 확보에 기여
- best가 epoch 63으로 이름 → 이후 과적합 발생, early stopping 필요성 확인

---

## 🏗️ 커스텀 모델 아키텍처 설계 — UNet3+ + ASPP + SWA (v3)

### 설계 배경 및 근거

| 문제 | 분석 | 해결 방향 |
|------|------|-----------|
| Hard_Object 크기 불규칙 | 단일 receptive field로 대응 불가 | ASPP로 다중 receptive field 확보 |
| MSCA 효과 없음 | 파라미터 과다, 삽입 위치 분산 | 경량 ASPP로 대체, bottleneck 단일 삽입 |
| 오검 50% 문제 | 반사광이 이물질과 픽셀 패턴 유사 | threshold 후처리로 FPR 최소화 |

### ASPP (Atrous Spatial Pyramid Pooling)

> Chen et al., "DeepLabV3+", ECCV 2018 — 불규칙 크기 객체 검출에 효과 입증

**삽입 위치: e5 bottleneck 직후**
- e5는 가장 고수준 특징이 압축된 지점
- 여기서 receptive field를 넓혀야 디코더 전체에 효과 전파
- 출력 채널 448ch 유지 → 기존 UNet3+ 디코더와 완전 호환

```
e5 bottleneck (448ch)
  ├── Conv 1×1 (dilation=1)   → 256ch
  ├── Conv 3×3 (dilation=6)   → 256ch
  ├── Conv 3×3 (dilation=12)  → 256ch
  ├── Conv 3×3 (dilation=18)  → 256ch
  └── Global Average Pooling  → 256ch
       ↓ concat (1280ch)
  Conv 1×1 → 448ch / BN / ReLU
```

### 전체 아키텍처

```
EfficientNet-B4 Encoder (ImageNet pretrained, 1ch 입력)
  e1(48ch) → e2(32ch) → e3(56ch) → e4(160ch) → e5(448ch)
                                                      ↓
                                              ASPP (448→448ch)
                                                      ↓
                         UNet3+ Full-Scale Decoder (d4→d3→d2→d1)
                                                      ↓
                                        Segmentation Output (num_classes)
```

### 파라미터 비교

| 모델 | 파라미터 수 | 비고 |
|------|------------|------|
| UNet++ (기존 운영) | 20,813,409 | - |
| UNet3+ (v1, 실험 A) | 23,702,833 | - |
| UNet3+MSCA (v2, 실험 B) | 32,513,393 | 과파라미터 |
| **UNet3+ASPP (v3)** | **~24,900,000** | 경량 설계 |

### 후처리: Hard_Object Threshold 자동 탐색

학습 완료 후 val 데이터로 threshold 0.1~0.9 자동 탐색.
`defect_image_FNR = 0` (미검 없음) 조건 하에서 Hard_Object FPR 최소 threshold 자동 선택.

```bash
python threshold_search.py \
    --model-file jib_model_v3.py \
    --checkpoint best_swa_unet3plus_aspp.pth \
    --img-dir /path/to/images \
    --mask-dir /path/to/masks
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-orange)
![SMP](https://img.shields.io/badge/segmentation--models--pytorch-latest-green)
![WandB](https://img.shields.io/badge/WandB-Tracking-yellow)

- **Framework**: PyTorch
- **Model Library**: segmentation-models-pytorch (SMP)
- **Backbone**: EfficientNet-B4
- **Experiment Tracking**: Weights & Biases (wandb)
- **Camera**: Basler 산업용 카메라
- **Backend**: FastAPI
- **GPU**: NVIDIA RTX 5060

---

## 📁 파일 구조

```
├── jib_model_v3.py                    # 커스텀 모델 (UNet3+ASPP)
├── threshold_search.py                # Hard_Object threshold 자동 탐색
└── UNet3Plus_ASPP_SWA_설계문서_v3.md  # 모델 설계 문서
```

---

## 📚 참고 문헌

| 논문 | 링크 |
|------|------|
| Huang et al., "UNet 3+", ICASSP 2020 | [arxiv](https://arxiv.org/abs/2004.08790) |
| Chen et al., "DeepLabV3+", ECCV 2018 | [arxiv](https://arxiv.org/abs/1802.02611) |
| Izmailov et al., "SWA", UAI 2018 | [arxiv](https://arxiv.org/abs/1803.05407) |
| Lin et al., "Focal Loss", ICCV 2017 | [arxiv](https://arxiv.org/abs/1708.02002) |
| Zhan et al., "MSCA_UNet3+", ICCR 2024 | [ACM](https://dl.acm.org/doi/10.1145/3687488.3687532) |
