# Experiment A — UNet3+ (v1)
**날짜**: 2026-05-27  
**모델**: UNet3+ + EfficientNet-B4  
**파일**: jib_model_v1  

---

# jib model v1 unet3plus 20260527 ivory cup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp



# 현재 사용 중인 모델


def crate_smp_UnetPlustPlus(num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
    """
    현재 운영 중인 세그멘테이션 모델.
    SMP UNet++ + efficientnet-b4 백본.
    encoder_weights=None: ImageNet 사전학습 없이 구조만 초기화.
    실제 가중치는 model_object.py에서 chk_point['model_state_dict']로 별도 로드.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=num_classes,
    )
    return model



# 실험 모델: UNet3+ + efficientnet-b4
# Huang et al., "UNet 3+: A Full-Scale Connected UNet
# for Medical Image Segmentation", ICASSP 2020
#
# 선택 근거:
# - 모든 인코더-디코더 레이어 간 풀스케일 스킵 연결
#   → DOT/LINE(소형) ~ OBJECT(대형) 동시 처리에 유리
# - Classification-guided Module(CGM)이 복잡한 배경의
#   과잉 세그멘테이션(오검) 억제
# - UNet, UNet++, Attention UNet, PSPNet, DeepLabV3+ 대비
#   우월한 성능을 ICASSP 2020에서 실험으로 증명


class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU 기본 블록"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3Plus(nn.Module):
    """
    UNet3+: Full-Scale Connected UNet

    현재 시스템 조건:
      - in_channels=1 (흑백 1200x1200 이미지)
      - num_classes: 결함 클래스 수 + 1 (배경 포함, 기존 코드와 동일)
      - encoder: efficientnet-b4 (SMP 라이브러리 활용)
      - fp16 추론 지원 (model_object.py에서 .half() 적용)

    구조:
      - 인코더: SMP efficientnet-b4 5단계 특징 추출
        출력 채널: [1, 48, 32, 56, 160, 448]
      - 디코더: 풀스케일 스킵 연결
        각 디코더 노드(d4, d3, d2, d1)가
        모든 인코더 스케일 + 하위 디코더 노드로부터 입력을 받음
      - 각 입력을 cat_channels(64)로 통일 후 concat → filters(320) 출력
      - 최종: stride 2 → 원본 해상도 복원
    """

    def __init__(self, num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
        super().__init__()

        # SMP 인코더 (기존 코드와 동일한 방식으로 활용)
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=1,
            depth=5,
            weights=encoder_weights
        )
        encoder_channels = self.encoder.out_channels
        # efficientnet-b4 기준: [1, 48, 32, 56, 160, 448]

        self.cat_channels = 64
        self.cat_blocks = 5
        self.filters = self.cat_channels * self.cat_blocks  # 320

        # ── 디코더 d4 (stride 16 해상도) ──
        # 입력: e1(↓8), e2(↓4), e3(↓2), e4(same), e5(↑2)
        self.d4_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d4_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d4_e3 = ConvBnRelu(encoder_channels[3], self.cat_channels)
        self.d4_e4 = ConvBnRelu(encoder_channels[4], self.cat_channels)
        self.d4_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d4_conv = ConvBnRelu(self.filters, self.filters)

        # ── 디코더 d3 (stride 8 해상도) ──
        # 입력: e1(↓4), e2(↓2), e3(same), d4(↑2), e5(↑4)
        self.d3_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d3_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d3_e3 = ConvBnRelu(encoder_channels[3], self.cat_channels)
        self.d3_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d3_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d3_conv = ConvBnRelu(self.filters, self.filters)

        # ── 디코더 d2 (stride 4 해상도) ──
        # 입력: e1(↓2), e2(same), d3(↑2), d4(↑4), e5(↑8)
        self.d2_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d2_e2 = ConvBnRelu(encoder_channels[2], self.cat_channels)
        self.d2_d3 = ConvBnRelu(self.filters, self.cat_channels)
        self.d2_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d2_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d2_conv = ConvBnRelu(self.filters, self.filters)

        # ── 디코더 d1 (stride 2 해상도) ──
        # 입력: e1(same), d2(↑2), d3(↑4), d4(↑8), e5(↑16)
        self.d1_e1 = ConvBnRelu(encoder_channels[1], self.cat_channels)
        self.d1_d2 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_d3 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_d4 = ConvBnRelu(self.filters, self.cat_channels)
        self.d1_e5 = ConvBnRelu(encoder_channels[5], self.cat_channels)
        self.d1_conv = ConvBnRelu(self.filters, self.filters)

        # ── 최종 출력 ──
        # d1이 stride 2 해상도 → 2배 업샘플링으로 원본 해상도 복원
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


def create_UNet3Plus(num_classes, encoder_name="efficientnet-b4", encoder_weights=None):
    """
    UNet3+ 생성 함수.
    기존 crate_smp_UnetPlustPlus()와 동일한 인터페이스로 교체 가능.

    model_object.py 교체 방법:
        # 기존
        self.model = crate_smp_UnetPlustPlus(num_classes=len(class_list)+1)
        # 변경
        self.model = create_UNet3Plus(num_classes=len(class_list)+1)
    """
    return UNet3Plus(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )



# OLD — 기존 코드 유지 (건드리지 않음)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

        self.encoder0 = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1,
            self.backbone.relu, self.backbone.maxpool
        )
        self.encoder1 = self.backbone.layer1
        self.encoder2 = self.backbone.layer2
        self.encoder3 = self.backbone.layer3
        self.encoder4 = self.backbone.layer4

        self.up_conv4 = ConvBlock(2048 + 1024, 1024)
        self.up_conv3 = ConvBlock(1024 + 512, 512)
        self.up_conv2 = ConvBlock(512 + 256, 256)
        self.up_conv1 = ConvBlock(256 + 64, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        up4 = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.up_conv4(torch.cat([up4, x3], dim=1))
        up3 = F.interpolate(d4, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.up_conv3(torch.cat([up3, x2], dim=1))
        up2 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.up_conv2(torch.cat([up2, x1], dim=1))
        up1 = F.interpolate(d2, size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.up_conv1(torch.cat([up1, x0], dim=1))

        out = self.final_up(d1)
        out = self.final_conv(out)
        return out

```

