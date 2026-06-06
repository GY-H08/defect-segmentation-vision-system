# Experiment C — UNet3+ + ASPP + SWA (v3)
**날짜**: 2026-06-02  
**모델**: UNet3+ + ASPP + EfficientNet-B4 + SWA  
**파일**: jib_model_v3, threshold_search, 설계문서_v3  

---

## 설계 문서

# UNet3+ + ASPP + SWA 모델 설계 문서

**버전**: v3.0  
**기반**: UNet3Plus_0527 / UNet3Plus_MSCA_SWA_0601 결과 분석  
**파일**: jib_model_v3.py, threshold_search.py

---

## 1. 이전 실험 결과 요약 및 개선 방향

### 실험 A (UNet3+, v1) / 실험 B (UNet3+MSCA+SWA, v2) 문제점

| 문제 | 내용 |
|------|------|
| Hard_Object recall 낮음 | 실험 A best 0.4384, 실험 B SWA_FINAL 0.2608 — 운영 기준 미달 |
| MSCA 효과 미미 | 파라미터 37% 증가(23.7M→32.5M)에 비해 성능 향상 없음 |
| MSCA best가 epoch 5 | warm start 시 MSCA 레이어 랜덤 초기화로 초기 학습 불안정 |
| 데이터 191장 한계 | 소량 데이터에서 복잡한 구조가 과파라미터로 작용 |

### 개선 방향

```
UNet3+ (v1)
    - MSCA 제거 (파라미터 대비 효과 없음)
    + ASPP (e5 bottleneck 삽입) → Hard_Object 크기 불규칙성 대응
    + SWA (학습 후처리)         → 가중치 안정화, defect_image_FNR 0 유지
    + 데이터 800장              → 충분한 학습 신호 확보
= UNet3+ + ASPP + SWA (v3)
```

---

## 2. ASPP (Atrous Spatial Pyramid Pooling)

### 선택 근거

Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+)", ECCV 2018

- 다양한 dilation rate의 병렬 atrous convolution으로 다중 receptive field 확보
- 크기가 불규칙한 객체 검출에 효과적임이 검증됨
- ResUNet++, COPLE-Net 등 다수 의료 영상 세그멘테이션에서 bottleneck 삽입 방식이 표준으로 사용됨

### MSCA 대신 ASPP를 선택한 이유

| 항목 | MSCA (v2) | ASPP (v3) |
|------|-----------|-----------|
| 방식 | 커널 크기(1×1~7×7)로 다중 스케일 | dilation rate(1,6,12,18)로 다중 receptive field |
| 삽입 위치 | 디코더 각 노드(4곳) | e5 bottleneck(1곳) |
| 파라미터 증가 | ~37% (약 8.8M) | ~5% (약 1.2M) |
| 실험 결과 | 800장 미만에서 과파라미터 문제 | 경량 구조로 800장 환경에 적합 |

### 삽입 위치: e5 bottleneck 직후

```
EfficientNet-B4 인코더
  e1(48ch) → e2(32ch) → e3(56ch) → e4(160ch) → e5(448ch)
                                                      ↓
                                              ASPP 삽입
                                      (dilation 1, 6, 12, 18 + GAP)
                                              448ch → 448ch
                                                      ↓
                     UNet3+ 풀스케일 디코더 (d4 → d3 → d2 → d1)
                                                      ↓
                                              최종 세그멘테이션
```

e5는 인코더에서 가장 고수준 특징이 압축된 지점으로, 여기서 receptive field를 넓혀야 디코더 전체에 효과가 전파된다. 출력 채널을 448ch로 유지하여 기존 UNet3+ 디코더와 완전 호환.

### ASPP 구조

```
e5 (448ch)
  ├── Conv 1×1 (dilation=1)  → 256ch
  ├── Conv 3×3 (dilation=6)  → 256ch
  ├── Conv 3×3 (dilation=12) → 256ch
  ├── Conv 3×3 (dilation=18) → 256ch
  └── Global Average Pooling → 256ch
       ↓ concat (1280ch)
  Conv 1×1 → 448ch
  BN → ReLU
```

---

## 3. SWA (Stochastic Weight Averaging)

### 선택 근거

실험 B에서 SWA 적용 결과 **defect_image_FNR 0.0** 달성 확인.  
결함 포함 이미지를 완전히 놓친 케이스 없음 → 운영 안전성 확보.

Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018

### 적용 방법

v2와 동일. 학습 스크립트 변경 불필요.

```python
from jib_model_v3 import SWATrainer

swa_start = int(total_epochs * 0.75)
for epoch in range(swa_start):
    train(model, ...)

swa_trainer = SWATrainer(model, swa_lr=0.001)
for epoch in range(swa_start, total_epochs):
    train(model, ...)
    swa_trainer.update()

swa_trainer.finalize(train_loader)
swa_trainer.save("best_swa_model.pt")
```

### 주의사항

- `finalize(train_loader)` 반드시 호출 (BatchNorm statistics 재계산)
- 저장 포맷이 기존 `model_state_dict`와 동일하므로 추론 코드 수정 불필요

---

## 4. 모델 교체 방법

### model_object.py 수정

```python
# import 변경
from .jib_model import ResNetUNet, crate_smp_UnetPlustPlus, create_UNet3PlusASPP

# SegmentModel.__init__ 내부
# 기존 (v2)
# self.model = create_UNet3PlusMSCA(num_classes=len(class_list)+1)

# 변경 (v3)
self.model = create_UNet3PlusASPP(num_classes=len(class_list)+1)
```

---

## 5. 파라미터 수 비교 (예상)

| 모델 | 파라미터 수 | 비고 |
|------|------------|------|
| UNet++ (기존 운영) | 20,813,409 | - |
| UNet3+ (v1) | 23,702,833 | 실험 A |
| UNet3+ + MSCA (v2) | 32,513,393 | 실험 B, 파라미터 과다 |
| UNet3+ + ASPP (v3) | ~24,900,000 | 예상치, 실측 필요 |

---

## 6. threshold_search.py 사용 방법

학습 완료 후 val 데이터로 Hard_Object threshold를 자동 탐색한다.  
`defect_image_FNR = 0` (미검 없음) 조건을 유지하면서  
Hard_Object FPR(오검률)을 최소화하는 최적 threshold를 자동 선택한다.

```bash
python threshold_search.py \
    --model-file jib_model_v3.py \
    --checkpoint best_swa_unet3plus_aspp.pth \
    --img-dir /path/to/images \
    --mask-dir /path/to/masks \
    --output-dir ./threshold_result
```

**출력:**
- 콘솔: 최적 threshold 및 해당 Hard_Object Recall/Precision/FPR
- `threshold_search_result.csv`: threshold별 전체 지표

**선택 기준:**
1. (필수) `defect_image_FNR = 0` — 결함 이미지 미검 없음
2. (최적) `Hard_Object FPR 최소` — 오검 최소화

---

## 7. 학습 조건 (v3 권고)

| 항목 | 설정값 |
|------|--------|
| 데이터 | 800장 (train 640 / val 160) |
| Epoch | 300 |
| SWA start | epoch 225 (75%) |
| Loss | HybridLoss: Focal(γ=2, w=0.5) + Dice(w=0.5) + CLASS_WEIGHTS |
| CLASS_WEIGHTS | Hard_Object 가중치 현재값 확인 후 3~5배 상향 권고 |
| Optimizer | AdamW + CosineAnnealingLR |
| Warm start | 실험 A best checkpoint (epoch 63) 사용 가능 |

---

## 8. 실험 가설

**H₀ (귀무)**: UNet3+에 ASPP를 추가해도 Hard_Object recall 및 전체 mIoU가 UNet3+ (v1) 대비 유의미하게 개선되지 않는다.

**H₁ (대립)**: ASPP의 다중 dilation rate가 크기가 불규칙한 Hard_Object의 receptive field를 확장하여, recall 및 mIoU가 v1 대비 유의미하게 향상된다.

**평가 지표**: mIoU_defect, Hard_Object recall/precision/FPR, defect_image_FNR  
**필수 조건**: defect_image_FNR = 0 유지

---

## 9. 참고 문헌

| 번호 | 논문 |
|------|------|
| [1] | Huang et al., "UNet 3+: A Full-Scale Connected UNet", ICASSP 2020 — arxiv.org/abs/2004.08790 |
| [2] | Chen et al., "Encoder-Decoder with Atrous Separable Convolution (DeepLabV3+)", ECCV 2018 — arxiv.org/abs/1802.02611 |
| [3] | Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018 — arxiv.org/abs/1803.05407 |
| [4] | Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017 — arxiv.org/abs/1708.02002 |
| [5] | Tan & Le, "EfficientNet", ICML 2019 — arxiv.org/abs/1905.11946 |


---

##  모델 코드

# jib model v3 unet3plus aspp 20260602 ivory cup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import segmentation_models_pytorch as smp
from torchvision import models



# 공통 기본 블록


class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)



# ASPP (Atrous Spatial Pyramid Pooling)
#
# 근거: Chen et al., "Encoder-Decoder with Atrous Separable
#       Convolution for Semantic Image Segmentation (DeepLabV3+)",
#       ECCV 2018. arxiv.org/abs/1802.02611
#
# 삽입 위치: 인코더 e5(bottleneck) 직후, 디코더 진입 전
#
# 선택 근거:
# - Hard_Object는 이물질 크기가 불규칙 → 단일 receptive field로는 한계
# - dilation rate 1, 6, 12, 18 병렬 처리로 다양한 크기 동시 포착
# - MSCA(실험 B) 대비 파라미터 증가 최소화 (~5%)
# - e5 bottleneck은 가장 고수준 특징이 압축된 지점으로
#   여기서 receptive field를 넓혀야 디코더 전체에 효과 전파
#
# 구조:
#   e5 (448ch)
#     ├── 1×1 conv         → 256ch
#     ├── 3×3 dilated(r=6) → 256ch
#     ├── 3×3 dilated(r=12)→ 256ch
#     ├── 3×3 dilated(r=18)→ 256ch
#     └── Global Avg Pool  → 256ch
#          ↓ concat (1280ch)
#     1×1 conv → 448ch (원래 채널 수 복원)
#     BN → ReLU


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling

    e5 bottleneck 직후에 삽입.
    입력/출력 채널을 동일하게 유지하여 UNet3+ 디코더와 호환.
    """
    def __init__(self, in_channels, out_channels=None, dilation_rates=(1, 6, 12, 18)):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = 256  # 각 브랜치 채널 수

        # 병렬 브랜치
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels,
                    kernel_size=1 if r == 1 else 3,
                    padding=0 if r == 1 else r,
                    dilation=r,
                    bias=False
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            for r in dilation_rates
        ])

        # Global Average Pooling 브랜치 (이미지 수준 컨텍스트)
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # concat 후 채널 수 복원
        total_mid = mid_channels * (len(dilation_rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_mid, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # 각 브랜치 특징 추출
        branch_outs = [branch(x) for branch in self.branches]

        # GAP 브랜치: 전역 컨텍스트 → 원래 크기로 복원
        gap_out = self.gap_branch(x)
        gap_out = F.interpolate(gap_out, size=(h, w), mode='bilinear', align_corners=False)

        # 모든 브랜치 concat → 채널 복원
        out = torch.cat(branch_outs + [gap_out], dim=1)
        return self.project(out)



# UNet3+ + ASPP (v3)
#
# 변경점 (v2 대비):
# - MSCA(디코더 각 노드 어텐션) 제거
#   → 파라미터 37% 증가에 비해 효과 없음 (실험 B 결과)
# - ASPP(bottleneck 다중 스케일 pooling) 추가
#   → 파라미터 ~5% 증가, Hard_Object 크기 불규칙성 대응
#   → DeepLabV3+ 검증된 방식 (Chen et al., ECCV 2018)
#
# 시스템 조건:
# - in_channels=1 (흑백 이미지)
# - num_classes: 결함 클래스 수 + 1 (배경 포함)
# - encoder: efficientnet-b4
# - 데이터: 800장 (train 640 / val 160)


class UNet3PlusASPP(nn.Module):
    """
    UNet3+ with ASPP at bottleneck

    구조:
      EfficientNet-B4 인코더
        e1(48ch) → e2(32ch) → e3(56ch) → e4(160ch) → e5(448ch)
                                                           ↓
                                                   ASPP (448ch → 448ch)
                                                           ↓
                              UNet3+ 풀스케일 디코더 (d4 → d3 → d2 → d1)
                                                           ↓
                                                    최종 세그멘테이션
    """

    def __init__(self, num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
        super().__init__()

        # ── 인코더 ──
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=1,
            depth=5,
            weights=encoder_weights
        )
        encoder_channels = self.encoder.out_channels
        # efficientnet-b4 기준: [1, 48, 32, 56, 160, 448]

        # ── ASPP (bottleneck: e5 직후) ──
        self.aspp = ASPP(
            in_channels=encoder_channels[5],   # 448
            out_channels=encoder_channels[5],  # 448 유지 (디코더 호환)
            dilation_rates=(1, 6, 12, 18)
        )

        # ── UNet3+ 디코더 설정 ──
        self.cat_channels = 64
        self.cat_blocks = 5
        self.filters = self.cat_channels * self.cat_blocks  # 320

        # ── 디코더 d4 (stride 16 해상도) ──
        # 입력: e1(↓8), e2(↓4), e3(↓2), e4(same), aspp(e5)(↑2)
        self.d4_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d4_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d4_e3 = ConvBnRelu(encoder_channels[3], self.cat_channels)
        self.d4_e4 = ConvBnRelu(encoder_channels[4], self.cat_channels)
        self.d4_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)  # ASPP 출력 받음
        self.d4_conv = ConvBnRelu(self.filters, self.filters)

        # ── 디코더 d3 (stride 8 해상도) ──
        self.d3_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d3_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d3_e3 = ConvBnRelu(encoder_channels[3], self.cat_channels)
        self.d3_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d3_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d3_conv = ConvBnRelu(self.filters, self.filters)

        # ── 디코더 d2 (stride 4 해상도) ──
        self.d2_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d2_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d2_d3 = ConvBnRelu(self.filters, self.cat_channels)
        self.d2_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d2_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d2_conv = ConvBnRelu(self.filters, self.filters)

        # ── 디코더 d1 (stride 2 해상도) ──
        self.d1_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d1_d2 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_d3 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d1_conv = ConvBnRelu(self.filters, self.filters)

        # ── 최종 출력 ──
        # d1이 stride 2 → 2배 업샘플링으로 원본 해상도 복원
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(self.filters, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # ── 인코더 ──
        features = self.encoder(x)
        e1 = features[1]  # stride 2,  ch: 48
        e2 = features[2]  # stride 4,  ch: 32
        e3 = features[3]  # stride 8,  ch: 56
        e4 = features[4]  # stride 16, ch: 160
        e5 = features[5]  # stride 32, ch: 448

        # ── ASPP: e5 bottleneck에서 다중 스케일 컨텍스트 추출 ──
        e5 = self.aspp(e5)  # 448ch → 448ch (채널 유지, 채널별 receptive field 확장)

        # ── 디코더 d4 ──
        d4 = self.d4_conv(torch.cat([
            self.d4_e1(F.max_pool2d(e1, kernel_size=8, stride=8)),
            self.d4_e2(F.max_pool2d(e2, kernel_size=4, stride=4)),
            self.d4_e3(F.max_pool2d(e3, kernel_size=2, stride=2)),
            self.d4_e4(e4),
            self.d4_e5(F.interpolate(e5, size=e4.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))

        # ── 디코더 d3 ──
        d3 = self.d3_conv(torch.cat([
            self.d3_e1(F.max_pool2d(e1, kernel_size=4, stride=4)),
            self.d3_e2(F.max_pool2d(e2, kernel_size=2, stride=2)),
            self.d3_e3(e3),
            self.d3_d4(F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)),
            self.d3_e5(F.interpolate(e5, size=e3.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))

        # ── 디코더 d2 ──
        d2 = self.d2_conv(torch.cat([
            self.d2_e1(F.max_pool2d(e1, kernel_size=2, stride=2)),
            self.d2_e2(e2),
            self.d2_d3(F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)),
            self.d2_d4(F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True)),
            self.d2_e5(F.interpolate(e5, size=e2.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))

        # ── 디코더 d1 ──
        d1 = self.d1_conv(torch.cat([
            self.d1_e1(e1),
            self.d1_d2(F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_d3(F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_d4(F.interpolate(d4, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_e5(F.interpolate(e5, size=e1.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))

        return self.final_conv(d1)


def create_UNet3PlusASPP(num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
    """
    UNet3+ + ASPP 생성 함수.
    기존 create_UNet3PlusMSCA()와 동일한 인터페이스.

    model_object.py 교체 방법:
        # 기존
        self.model = create_UNet3PlusMSCA(num_classes=len(class_list)+1)
        # 변경
        self.model = create_UNet3PlusASPP(num_classes=len(class_list)+1)
    """
    return UNet3PlusASPP(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )



# SWATrainer
# (v2와 동일, 하위 호환 유지)


class SWATrainer:
    """
    Stochastic Weight Averaging 래퍼.

    근거: Izmailov et al., "Averaging Weights Leads to Wider Optima
          and Better Generalization", UAI 2018.
          arxiv.org/abs/1803.05407

    사용법:
        swa_start = int(total_epochs * 0.75)
        for epoch in range(swa_start):
            train(model, ...)

        swa_trainer = SWATrainer(model, swa_lr=0.001)
        for epoch in range(swa_start, total_epochs):
            train(model, ...)
            swa_trainer.update()

        swa_trainer.finalize(train_loader)
        swa_trainer.save("best_swa_model.pt")
    """

    def __init__(self, model, swa_lr=0.001, anneal_epochs=10):
        self.model = model
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            torch.optim.SGD(model.parameters(), lr=swa_lr),
            swa_lr=swa_lr,
            anneal_epochs=anneal_epochs,
            anneal_strategy='cos'
        )

    def update(self):
        """매 epoch 끝에 호출 — 가중치 평균 누적"""
        self.swa_model.update_parameters(self.model)
        self.swa_scheduler.step()

    def finalize(self, train_loader, device='cuda'):
        """
        학습 완료 후 호출.
        SWA 모델의 BatchNorm statistics 업데이트.
        반드시 호출해야 추론 시 정확한 결과가 나옴.
        """
        print("SWA: BatchNorm statistics 업데이트 중...")
        update_bn(train_loader, self.swa_model, device=device)
        print("SWA: 완료")

    def save(self, path):
        """SWA 가중치 저장 (기존 model_state_dict 포맷 유지)"""
        torch.save(
            {'model_state_dict': self.swa_model.module.state_dict()},
            path
        )
        print(f"SWA 모델 저장 완료: {path}")

    def get_model(self):
        """SWA 모델 반환 (추론용)"""
        return self.swa_model



# 레거시 (기존 코드 유지, 하위 호환)


def crate_smp_UnetPlustPlus(num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=num_classes,
    )


class UNet3Plus(nn.Module):
    """기존 UNet3+ (ASPP 없는 버전, 비교 실험용)"""

    def __init__(self, num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=1, depth=5, weights=encoder_weights
        )
        encoder_channels = self.encoder.out_channels
        self.cat_channels = 64
        self.cat_blocks = 5
        self.filters = self.cat_channels * self.cat_blocks

        self.d4_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d4_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d4_e3 = ConvBnRelu(encoder_channels[3], self.cat_channels)
        self.d4_e4 = ConvBnRelu(encoder_channels[4], self.cat_channels)
        self.d4_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d4_conv = ConvBnRelu(self.filters, self.filters)

        self.d3_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d3_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d3_e3 = ConvBnRelu(encoder_channels[3], self.cat_channels)
        self.d3_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d3_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d3_conv = ConvBnRelu(self.filters, self.filters)

        self.d2_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d2_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d2_d3 = ConvBnRelu(self.filters, self.cat_channels)
        self.d2_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d2_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d2_conv = ConvBnRelu(self.filters, self.filters)

        self.d1_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d1_d2 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_d3 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d1_conv = ConvBnRelu(self.filters, self.filters)

        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(self.filters, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        e1, e2, e3, e4, e5 = features[1], features[2], features[3], features[4], features[5]

        d4 = self.d4_conv(torch.cat([
            self.d4_e1(F.max_pool2d(e1, kernel_size=8, stride=8)),
            self.d4_e2(F.max_pool2d(e2, kernel_size=4, stride=4)),
            self.d4_e3(F.max_pool2d(e3, kernel_size=2, stride=2)),
            self.d4_e4(e4),
            self.d4_e5(F.interpolate(e5, size=e4.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))

        d3 = self.d3_conv(torch.cat([
            self.d3_e1(F.max_pool2d(e1, kernel_size=4, stride=4)),
            self.d3_e2(F.max_pool2d(e2, kernel_size=2, stride=2)),
            self.d3_e3(e3),
            self.d3_d4(F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)),
            self.d3_e5(F.interpolate(e5, size=e3.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))

        d2 = self.d2_conv(torch.cat([
            self.d2_e1(F.max_pool2d(e1, kernel_size=2, stride=2)),
            self.d2_e2(e2),
            self.d2_d3(F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)),
            self.d2_d4(F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True)),
            self.d2_e5(F.interpolate(e5, size=e2.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))

        d1 = self.d1_conv(torch.cat([
            self.d1_e1(e1),
            self.d1_d2(F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_d3(F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_d4(F.interpolate(d4, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_e5(F.interpolate(e5, size=e1.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))

        return self.final_conv(d1)


def create_UNet3Plus(num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
    return UNet3Plus(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )


# ── UNet3PlusMSCA 레거시 (v2 하위 호환) ──

class MSCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(channels, channels // 4, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(channels, channels // 4, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 4)
        self.fc2 = nn.Linear(channels // 4, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ms = torch.cat([self.conv1x1(x), self.conv3x3(x), self.conv5x5(x), self.conv7x7(x)], dim=1)
        ms = self.relu(self.bn(ms))
        b, c, _, _ = ms.shape
        gap = self.gap(ms).view(b, c)
        attn = self.sigmoid(self.fc2(F.relu(self.fc1(gap)))).view(b, c, 1, 1)
        return ms * attn + x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

```


---

##  Threshold 탐색 스크립트

# threshold search unet3plus aspp swa 20260602 ivory cup

```python
"""
threshold_search.py
────────────────────────────────────────────────────────────────
Hard_Object 최적 threshold 자동 탐색 스크립트

목적:
  학습 완료 후 val 데이터로 Hard_Object threshold를 자동으로 탐색.
  "defect_image_FNR = 0 (미검 없음)" 조건을 유지하면서
  Hard_Object FPR(오검률)을 최소화하는 threshold를 찾는다.

사용법:
  python threshold_search.py \
      --model-file jib_model_v3.py \
      --checkpoint best_unet3plus_aspp.pth \
      --img-dir /path/to/images \
      --mask-dir /path/to/masks

출력:
  - 콘솔: 최적 threshold 및 해당 지표
  - threshold_search_result.csv: 전체 탐색 결과
────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

# ── 프로젝트 경로 (필요시 수정) ──
PROJECT_ROOT = Path(r"C:\workspace\Vision_Platform")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import configs.UNet3Plus_ivr_top_compile as cfg
from src.data.UNET_dataset_paired import PairedDefectDataset



# 유틸 함수


def parse_args():
    parser = argparse.ArgumentParser(description="Hard_Object threshold 자동 탐색")
    parser.add_argument("--model-file", type=Path, required=True,
                        help="jib_model_v3.py 경로")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="학습 완료된 모델 checkpoint (.pth)")
    parser.add_argument("--img-dir", type=Path, default=cfg.IMG_DIR)
    parser.add_argument("--mask-dir", type=Path, default=cfg.MASK_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="결과 CSV 저장 경로")
    parser.add_argument("--threshold-min", type=float, default=0.1,
                        help="탐색 시작 threshold (default: 0.1)")
    parser.add_argument("--threshold-max", type=float, default=0.9,
                        help="탐색 종료 threshold (default: 0.9)")
    parser.add_argument("--threshold-step", type=float, default=0.01,
                        help="탐색 간격 (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def load_model(model_file, checkpoint_path, device):
    """모델 로드"""
    spec = importlib.util.spec_from_file_location("jib_model_v3", str(model_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.create_UNet3PlusASPP(
        num_classes=cfg.NUM_CLASS,
        encoder_name=cfg.ENCODER_NAME,
        encoder_weights=None,
    )

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # module. prefix 제거 (SWA 모델 호환)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def make_val_loader(img_dir, mask_dir, batch_size):
    """val 데이터 로더 생성 (학습 스크립트와 동일한 split 사용)"""
    import random

    color_to_class = {
        tuple(color): class_idx
        for class_idx, color in cfg.COLOR_MAP_RGB.items()
    }

    dataset = PairedDefectDataset(
        image_dir=str(img_dir),
        mask_dir=str(mask_dir),
        color_class=color_to_class,
        is_train=False,
        h=cfg.H_TENSOR,
        w=cfg.W_TENSOR,
    )

    # 학습 스크립트와 동일한 seed/split
    indices = list(range(len(dataset)))
    rng = random.Random(cfg.SEED)
    rng.shuffle(indices)
    val_count = max(1, int(len(dataset) * cfg.VAL_RATIO))
    val_indices = sorted(indices[:val_count])

    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Val 데이터: {len(val_indices)}장")
    return loader



# 핵심: threshold 적용 추론


HARD_OBJECT_IDX = cfg.CLASS_NAMES.index("Hard_Object")
DEFECT_CLASSES = cfg.DEFECT_CLASSES


@torch.no_grad()
def run_inference_with_threshold(model, loader, hard_object_threshold, device):
    """
    주어진 threshold로 val 전체 추론 후 지표 계산.

    Hard_Object 판정 방식:
      - softmax 확률맵에서 Hard_Object 채널 값 >= threshold 이면 Hard_Object로 판정
      - 나머지 클래스는 기존 argmax 방식 유지
      - 즉 Hard_Object가 argmax가 아니어도 threshold 이상이면 잡힘 → FN 감소 효과
      - threshold를 높이면 확실한 경우만 잡음 → FP 감소 효과

    반환:
      dict: {
        hard_object_tp, hard_object_fp, hard_object_fn,
        hard_object_pixel_recall, hard_object_pixel_precision,
        hard_object_pixel_fpr,
        defect_image_total, defect_image_missed, defect_image_fnr,
        hard_object_image_total, hard_object_image_missed, hard_object_image_fnr,
      }
    """
    use_amp = cfg.USE_AMP and device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

    hard_object_tp = 0
    hard_object_fp = 0
    hard_object_fn = 0
    hard_object_tn = 0

    defect_image_total = 0
    defect_image_missed = 0
    hard_object_image_total = 0
    hard_object_image_missed = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        if cfg.USE_CHANNELS_LAST and device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        with amp_ctx:
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )

        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # ── Hard_Object threshold 적용 ──
        # 1) 기본 argmax 예측
        pred = torch.argmax(probs, dim=1)  # [B, H, W]

        # 2) Hard_Object 확률이 threshold 이상인 픽셀은 Hard_Object로 override
        hard_prob = probs[:, HARD_OBJECT_IDX, :, :]  # [B, H, W]
        pred[hard_prob >= hard_object_threshold] = HARD_OBJECT_IDX

        # ── 픽셀 단위 Hard_Object TP/FP/FN/TN ──
        gt_hard = (masks == HARD_OBJECT_IDX)
        pred_hard = (pred == HARD_OBJECT_IDX)

        hard_object_tp += int((gt_hard & pred_hard).sum().item())
        hard_object_fp += int((~gt_hard & pred_hard).sum().item())
        hard_object_fn += int((gt_hard & ~pred_hard).sum().item())
        hard_object_tn += int((~gt_hard & ~pred_hard).sum().item())

        # ── 이미지 단위 결함 미검 (defect_image_fnr) ──
        gt_defect = torch.zeros_like(masks, dtype=torch.bool)
        pred_defect = torch.zeros_like(pred, dtype=torch.bool)
        for class_idx in DEFECT_CLASSES:
            gt_defect |= (masks == class_idx)
            pred_defect |= (pred == class_idx)

        gt_defect_images = gt_defect.flatten(1).any(dim=1)
        pred_defect_images = pred_defect.flatten(1).any(dim=1)
        defect_image_total += int(gt_defect_images.sum().item())
        defect_image_missed += int((gt_defect_images & ~pred_defect_images).sum().item())

        # ── 이미지 단위 Hard_Object 미검 ──
        gt_hard_images = gt_hard.flatten(1).any(dim=1)
        pred_hard_images = pred_hard.flatten(1).any(dim=1)
        hard_object_image_total += int(gt_hard_images.sum().item())
        hard_object_image_missed += int((gt_hard_images & ~pred_hard_images).sum().item())

    # ── 지표 계산 ──
    eps = 1e-8
    pixel_recall    = hard_object_tp / (hard_object_tp + hard_object_fn + eps)
    pixel_precision = hard_object_tp / (hard_object_tp + hard_object_fp + eps)
    # FPR: 실제 배경 픽셀 중 Hard_Object로 잘못 잡은 비율 (오검률)
    pixel_fpr       = hard_object_fp / (hard_object_fp + hard_object_tn + eps)

    defect_image_fnr      = defect_image_missed / max(defect_image_total, 1)
    hard_object_image_fnr = hard_object_image_missed / max(hard_object_image_total, 1)

    return {
        "hard_object_tp": hard_object_tp,
        "hard_object_fp": hard_object_fp,
        "hard_object_fn": hard_object_fn,
        "hard_object_pixel_recall": pixel_recall,
        "hard_object_pixel_precision": pixel_precision,
        "hard_object_pixel_fpr": pixel_fpr,
        "defect_image_total": defect_image_total,
        "defect_image_missed": defect_image_missed,
        "defect_image_fnr": defect_image_fnr,
        "hard_object_image_total": hard_object_image_total,
        "hard_object_image_missed": hard_object_image_missed,
        "hard_object_image_fnr": hard_object_image_fnr,
    }



# 메인


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model = load_model(args.model_file, args.checkpoint, device)
    val_loader = make_val_loader(args.img_dir, args.mask_dir, args.batch_size)

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step
    )

    print(f"\n탐색 범위: {args.threshold_min:.2f} ~ {args.threshold_max:.2f} "
          f"(간격 {args.threshold_step:.2f}, 총 {len(thresholds)}개)")
    print("=" * 80)

    rows = []
    for thr in tqdm(thresholds, desc="threshold 탐색"):
        thr = round(float(thr), 4)
        result = run_inference_with_threshold(model, val_loader, thr, device)
        row = {"threshold": thr, **result}
        rows.append(row)

    # ── 결과 CSV 저장 ──
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "threshold_search_result.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n전체 결과 저장: {csv_path}")

    # ── 최적 threshold 선택 ──
    # 조건 1 (필수): defect_image_fnr == 0.0  (미검 없음, 절대 조건)
    # 조건 2 (최적): hard_object_pixel_fpr 최소 (오검 최소)
    print("\n" + "=" * 80)
    print("최적 threshold 탐색 결과")
    print("조건: defect_image_FNR = 0 유지 하에서 Hard_Object FPR 최소")
    print("=" * 80)

    # 조건 1 충족 행만 필터
    valid_rows = [r for r in rows if r["defect_image_fnr"] == 0.0]

    if not valid_rows:
        print("\n[경고] defect_image_FNR = 0을 만족하는 threshold가 없습니다.")
        print("       threshold_min을 낮추거나 모델 재학습을 권고합니다.")
        # 그래도 FNR이 가장 낮은 것 출력
        best = min(rows, key=lambda r: (r["defect_image_fnr"], r["hard_object_pixel_fpr"]))
        print(f"\n참고 (FNR 최소 기준):")
        _print_result(best)
    else:
        # 조건 2: FPR 최소
        best = min(valid_rows, key=lambda r: r["hard_object_pixel_fpr"])
        print(f"\n✅ 최적 threshold: {best['threshold']:.2f}")
        _print_result(best)

        # 추가: FNR=0 구간 전체 출력
        print(f"\ndefect_image_FNR = 0 구간: "
              f"threshold {valid_rows[0]['threshold']:.2f} ~ {valid_rows[-1]['threshold']:.2f} "
              f"({len(valid_rows)}개)")

    print("\n이 threshold 값을 추론 파이프라인(model_object.py)에 적용하세요.")


def _print_result(row):
    print(f"  threshold              : {row['threshold']:.2f}")
    print(f"  Hard_Object Recall     : {row['hard_object_pixel_recall']:.4f}")
    print(f"  Hard_Object Precision  : {row['hard_object_pixel_precision']:.4f}")
    print(f"  Hard_Object FPR (오검) : {row['hard_object_pixel_fpr']:.4f}")
    print(f"  defect_image_FNR       : {row['defect_image_fnr']:.4f}")
    print(f"  Hard_Object image FNR  : {row['hard_object_image_fnr']:.4f}")
    print(f"  Hard_Object TP/FP/FN   : {row['hard_object_tp']} / {row['hard_object_fp']} / {row['hard_object_fn']}")


if __name__ == "__main__":
    main()

```

