import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two Conv3d → BN → ReLU layers. The building block of every U-Net level."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class Small3DUNet(nn.Module):
    """
    Lightweight 3D U-Net for binary brain lesion segmentation.

    Architecture rationale:
    - 3D convolutions instead of 2D: brain lesions are volumetric; processing
      slice-by-slice loses inter-slice context critical for small lesions.
    - Skip connections: preserve fine spatial detail that is lost during
      downsampling — essential for accurate mask boundaries.
    - Global-aware bottleneck: the 8×16×16 bottleneck still sees the full
      spatial context at 1/8 resolution before upsampling.
    - ~3M parameters: small enough to train on CPU in <4h for 50 epochs.

    Input:  (B, 1, 64, 128, 128)  — normalised CT volume
    Output: (B, 1, 64, 128, 128)  — sigmoid segmentation probability map
    """

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels  # 16 by default

        # Encoder
        self.enc1 = _conv_block(1, c)        # → (B, 16, 64, 128, 128)
        self.pool1 = nn.MaxPool3d(2)          # → (B, 16, 32,  64,  64)

        self.enc2 = _conv_block(c, c * 2)    # → (B, 32, 32,  64,  64)
        self.pool2 = nn.MaxPool3d(2)          # → (B, 32, 16,  32,  32)

        self.enc3 = _conv_block(c * 2, c * 4) # → (B, 64, 16, 32,  32)
        self.pool3 = nn.MaxPool3d(2)           # → (B, 64,  8, 16,  16)

        # Bottleneck
        self.bottleneck = _conv_block(c * 4, c * 8)  # → (B, 128, 8, 16, 16)

        # Decoder (upsample + concat skip + conv block)
        self.up3  = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(c * 8, c * 4)  # cat(up3, enc3) = 128ch input

        self.up2  = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(c * 4, c * 2)

        self.up1  = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _conv_block(c * 2, c)

        # 1×1×1 conv to single output channel
        self.output_conv = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder path with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.output_conv(d1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
