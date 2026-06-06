# Experiment B — UNet3+ + MSCA + SWA (v2)
**날짜**: 2026-06-02  
**모델**: UNet3+ + MSCA + EfficientNet-B4 + SWA  
**파일**: jib_model_v2, 설계문서_v2  

---

##  설계 문서

# UNet3+ + MSCA + SWA 모델 설계 문서

**버전**: v2.0  
**기반**: UNet3Plus_0527_1558 학습 결과 분석  
**파일**: jib_model_v2.py

---

## 1. 이전 실험 결과 요약 및 개선 방향

### 이전 실험 (UNet3+, v1) 문제점

| 문제 | 내용 |
|------|------|
| Hard_Object recall ~0.35 | 오검 주범 클래스가 불안정하고 하락 추세 |
| Hard_Object 경계 불안정 | 단일 스케일 Conv로는 다양한 크기의 OBJECT 경계 구분 한계 |
| 학습 후반 가중치 불안정 | val_loss 수렴 구간에서 진동 발생 |

### 개선 방향

```
UNet3+ (v1)
    + MSCA 어텐션 (디코더 각 노드에 추가)  → Hard_Object 경계 개선
    + SWA (학습 후처리)                    → 가중치 안정화
= UNet3+ + MSCA + SWA (v2)
```

---

## 2. MSCA (Multi-Scale Convolutional Attention)

### 선택 근거

MSCA_UNet3+, ICCR 2024 — UNet3+ 디코더에 다중 스케일 어텐션 추가 시:
- Dice +1.268%
- Recall +0.708%
- 파라미터 소폭 증가만으로 실험 증명

### 구조

```
입력 feature map (channels=320)
    ├── Conv 1×1 → 80ch
    ├── Conv 3×3 → 80ch
    ├── Conv 5×5 → 80ch
    └── Conv 7×7 → 80ch
         ↓ concat
    [320ch] → BN → ReLU
         ↓
    채널 어텐션 (GAP → FC → Sigmoid)
         ↓
    출력 + residual connection
```

### 이 시스템에서 기대 효과

Hard_Object는 이물질 크기가 다양함. 1×1~7×7 다중 스케일 커널이 작은 이물질부터 큰 이물질까지 동시에 포착하고, 채널 어텐션이 엔젤링(반사광)과 실제 이물질을 채널 단위로 구분하는 데 기여할 것으로 기대.

---

## 3. SWA (Stochastic Weight Averaging)

### 선택 근거

Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018

학습 후반 N epoch의 가중치를 평균내어 더 평평한 loss landscape로 수렴.  
이전 실험에서 val_loss 수렴 구간의 진동을 SWA로 안정화.

### 적용 방법

```python
# 학습 코드에 추가하는 방법

from jib_model_v2 import SWATrainer

# 일반 학습 (전체 epoch의 75% 진행)
swa_start = int(total_epochs * 0.75)
for epoch in range(swa_start):
    train(model, ...)

# SWA 시작 (나머지 25% epoch)
swa_trainer = SWATrainer(model, swa_lr=0.001)
for epoch in range(swa_start, total_epochs):
    train(model, ...)
    swa_trainer.update()  # 매 epoch 끝에 호출

# 완료 후 BN 업데이트 및 저장
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
from .jib_model import ResNetUNet, crate_smp_UnetPlustPlus, create_UNet3PlusMSCA

# SegmentModel.__init__ 내부
# 기존 UNet3+
# self.model = create_UNet3Plus(num_classes=len(class_list)+1)

# 변경: UNet3+ + MSCA
self.model = create_UNet3PlusMSCA(num_classes=len(class_list)+1)
```

---

## 5. 파라미터 수 비교 (CPU 실측)

| 모델 | 파라미터 수 | 비고 |
|------|------------|------|
| UNet++ (기존) | 20,813,409 | 현재 운영 중 |
| UNet3+ (v1) | 23,702,833 | 이전 실험 |
| UNet3+ + MSCA (v2) | 32,513,393 | 이번 실험 |

파라미터 증가는 MSCA 어텐션 모듈 추가에 의한 것. RTX 5060에서 1초 이내 추론 가능 여부는 실측 필요.

---

## 6. 실험 가설

**H₀ (귀무)**: UNet3+에 MSCA 어텐션을 추가해도 Hard_Object recall 및 전체 mIoU가 UNet3+ (v1) 대비 유의미하게 개선되지 않는다.

**H₁ (대립)**: MSCA의 다중 스케일 어텐션이 Hard_Object 경계 검출을 개선하고 채널 어텐션이 엔젤링 반사광을 억제해, recall 및 mIoU가 v1 대비 유의미하게 향상된다.

**평가 지표**: mIoU, Hard_Object recall/precision, 추론 시간(ms)  
**필수 조건**: 미검률(FNR) 0% 유지

---

## 7. 참고 문헌

| 번호 | 논문 |
|------|------|
| [1] | Huang et al., "UNet 3+: A Full-Scale Connected UNet", ICASSP 2020 — arxiv.org/abs/2004.08790 |
| [2] | Zhan et al., "MSCA_UNet3+: UNet3+ with multi-scale convolutional attention", ICCR 2024 — dl.acm.org/doi/10.1145/3687488.3687532 |
| [3] | Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018 — arxiv.org/abs/1803.05407 |
| [4] | Zhou et al., "UNet++", IEEE TMI 2020 — arxiv.org/abs/1912.05074 |
| [5] | Tan & Le, "EfficientNet", ICML 2019 — arxiv.org/abs/1905.11946 |



---

##  모델 코드

# jib model v2 unet3plus msca swa 20260602 ivory cup

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



# MSCA (Multi-Scale Convolutional Attention) 모듈
#
# 근거: MSCA_UNet3+, ICCR 2024
# "UNet3+ 디코더에 다중 스케일 어텐션 추가 시
#  Dice +1.268%, Recall +0.708% 향상 실험으로 확인"
#
# 구조:
# - 3가지 커널(1x1, 3x3, 5x5)로 다중 스케일 특징 추출
# - 채널 어텐션으로 중요 특징 강조
# - Hard_Object처럼 다양한 크기의 결함 경계 개선 목적


class MSCA(nn.Module):
    """
    Multi-Scale Convolutional Attention
    UNet3+ 각 디코더 노드 출력에 적용
    """
    def __init__(self, channels):
        super().__init__()

        # 다중 스케일 특징 추출 (3가지 커널 크기)
        self.conv1x1 = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(channels, channels // 4, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(channels, channels // 4, kernel_size=7, padding=3, bias=False)

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # 채널 어텐션 (Squeeze-and-Excitation 방식)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 4)
        self.fc2 = nn.Linear(channels // 4, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 다중 스케일 특징 concat
        ms = torch.cat([
            self.conv1x1(x),
            self.conv3x3(x),
            self.conv5x5(x),
            self.conv7x7(x),
        ], dim=1)
        ms = self.relu(self.bn(ms))

        # 채널 어텐션
        b, c, _, _ = ms.shape
        gap = self.gap(ms).view(b, c)
        attn = self.sigmoid(self.fc2(F.relu(self.fc1(gap)))).view(b, c, 1, 1)

        return ms * attn + x  # residual connection



# UNet3+ + MSCA
#
# 기존 UNet3+ 구조에 MSCA를 각 디코더 노드 출력에 추가
# UNet3+ 논문: Huang et al., ICASSP 2020
# MSCA 논문: MSCA_UNet3+, ICCR 2024


class UNet3PlusMSCA(nn.Module):
    """
    UNet3+ with Multi-Scale Convolutional Attention

    현재 시스템 조건:
      - in_channels=1 (흑백 1200x1200)
      - num_classes: 결함 클래스 수 + 1 (배경 포함)
      - encoder: efficientnet-b4

    UNet3+ 대비 변경점:
      - 각 디코더 노드(d4, d3, d2, d1) 출력에 MSCA 추가
      - Hard_Object처럼 다양한 크기의 결함 경계 검출 개선
    """

    def __init__(self, num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=1,
            depth=5,
            weights=encoder_weights
        )
        encoder_channels = self.encoder.out_channels
        # efficientnet-b4: [1, 48, 32, 56, 160, 448]

        self.cat_channels = 64
        self.cat_blocks = 5
        self.filters = self.cat_channels * self.cat_blocks  # 320

        # ── 디코더 d4 ──
        self.d4_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d4_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d4_e3 = ConvBnRelu(encoder_channels[3], self.cat_channels)
        self.d4_e4 = ConvBnRelu(encoder_channels[4], self.cat_channels)
        self.d4_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d4_conv = ConvBnRelu(self.filters, self.filters)
        self.d4_msca = MSCA(self.filters)  # ← MSCA 추가

        # ── 디코더 d3 ──
        self.d3_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d3_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d3_e3 = ConvBnRelu(encoder_channels[3], self.cat_channels)
        self.d3_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d3_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d3_conv = ConvBnRelu(self.filters, self.filters)
        self.d3_msca = MSCA(self.filters)  # ← MSCA 추가

        # ── 디코더 d2 ──
        self.d2_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d2_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d2_d3 = ConvBnRelu(self.filters, self.cat_channels)
        self.d2_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d2_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d2_conv = ConvBnRelu(self.filters, self.filters)
        self.d2_msca = MSCA(self.filters)  # ← MSCA 추가

        # ── 디코더 d1 ──
        self.d1_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d1_d2 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_d3 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d1_conv = ConvBnRelu(self.filters, self.filters)
        self.d1_msca = MSCA(self.filters)  # ← MSCA 추가

        # ── 최종 출력 ──
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(self.filters, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        e1 = features[1]  # stride 2,  ch: 48
        e2 = features[2]  # stride 4,  ch: 32
        e3 = features[3]  # stride 8,  ch: 56
        e4 = features[4]  # stride 16, ch: 160
        e5 = features[5]  # stride 32, ch: 448

        # d4
        d4 = self.d4_conv(torch.cat([
            self.d4_e1(F.max_pool2d(e1, kernel_size=8, stride=8)),
            self.d4_e2(F.max_pool2d(e2, kernel_size=4, stride=4)),
            self.d4_e3(F.max_pool2d(e3, kernel_size=2, stride=2)),
            self.d4_e4(e4),
            self.d4_e5(F.interpolate(e5, size=e4.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))
        d4 = self.d4_msca(d4)  # ← MSCA

        # d3
        d3 = self.d3_conv(torch.cat([
            self.d3_e1(F.max_pool2d(e1, kernel_size=4, stride=4)),
            self.d3_e2(F.max_pool2d(e2, kernel_size=2, stride=2)),
            self.d3_e3(e3),
            self.d3_d4(F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)),
            self.d3_e5(F.interpolate(e5, size=e3.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))
        d3 = self.d3_msca(d3)  # ← MSCA

        # d2
        d2 = self.d2_conv(torch.cat([
            self.d2_e1(F.max_pool2d(e1, kernel_size=2, stride=2)),
            self.d2_e2(e2),
            self.d2_d3(F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)),
            self.d2_d4(F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True)),
            self.d2_e5(F.interpolate(e5, size=e2.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))
        d2 = self.d2_msca(d2)  # ← MSCA

        # d1
        d1 = self.d1_conv(torch.cat([
            self.d1_e1(e1),
            self.d1_d2(F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_d3(F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_d4(F.interpolate(d4, size=e1.shape[2:], mode='bilinear', align_corners=True)),
            self.d1_e5(F.interpolate(e5, size=e1.shape[2:], mode='bilinear', align_corners=True)),
        ], dim=1))
        d1 = self.d1_msca(d1)  # ← MSCA

        return self.final_conv(d1)


def create_UNet3PlusMSCA(num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
    """
    UNet3+ + MSCA 생성 함수.

    model_object.py 교체 방법:
        # 변경
        from .jib_model import create_UNet3PlusMSCA
        self.model = create_UNet3PlusMSCA(num_classes=len(class_list)+1)
    """
    return UNet3PlusMSCA(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )



# SWA (Stochastic Weight Averaging) 학습 유틸리티
#
# 근거: Izmailov et al., "Averaging Weights Leads to Wider Optima
#        and Better Generalization", UAI 2018
#
# 학습 후반부 N epoch의 가중치를 평균내어
# 더 평평한 loss landscape로 수렴 → 일반화 성능 향상
# 소규모 데이터에서 가중치 불안정 문제 완화 목적


class SWATrainer:
    """
    SWA 학습 헬퍼 클래스.

    사용 예시:
        # 학습 초반: 일반 학습
        for epoch in range(swa_start):
            train(model, ...)

        # SWA 시작
        swa_trainer = SWATrainer(model, swa_lr=0.001)
        for epoch in range(swa_start, total_epochs):
            train(model, ...)
            swa_trainer.update()

        # 학습 완료 후 BN 업데이트 및 저장
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
        반드시 호출해야 추론 시 정확한 결과 나옴.
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



# 기존 모델 유지 (하위 호환)


def crate_smp_UnetPlustPlus(num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=num_classes,
    )


class UNet3Plus(nn.Module):
    """기존 UNet3+ (MSCA 없는 버전, 비교 실험용)"""

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
    return UNet3Plus(num_classes=num_classes, encoder_name=encoder_name, encoder_weights=encoder_weights)



# 레거시 (기존 코드 유지)


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

class ResNetUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNetUNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        original_weights = self.backbone.conv1.weight.data
        mean_weights = torch.mean(original_weights, dim=1, keepdim=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = mean_weights
        self.encoder0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.encoder1 = self.backbone.layer1
        self.encoder2 = self.backbone.layer2
        self.encoder3 = self.backbone.layer3
        self.encoder4 = self.backbone.layer4
        self.up_conv4 = ConvBlock(2048+1024, 1024)
        self.up_conv3 = ConvBlock(1024+512, 512)
        self.up_conv2 = ConvBlock(512+256, 256)
        self.up_conv1 = ConvBlock(256+64, 64)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.encoder0(x); x1 = self.encoder1(x0); x2 = self.encoder2(x1)
        x3 = self.encoder3(x2); x4 = self.encoder4(x3)
        d4 = self.up_conv4(torch.cat([F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True), x3], dim=1))
        d3 = self.up_conv3(torch.cat([F.interpolate(d4, size=x2.shape[2:], mode='bilinear', align_corners=True), x2], dim=1))
        d2 = self.up_conv2(torch.cat([F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=True), x1], dim=1))
        d1 = self.up_conv1(torch.cat([F.interpolate(d2, size=x0.shape[2:], mode='bilinear', align_corners=True), x0], dim=1))
        return self.final_conv(self.final_up(d1))

```

