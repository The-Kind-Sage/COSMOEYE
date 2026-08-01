import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention: recalibrates channel
    importance (helps the network weigh spectral/temporal channels).

    Cost: negligible params/flops; benefit: consistent segmentation gains
    on multi-spectral imagery.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention for skip connections.

    Landslides are tiny (median ~4x4 px). Spatial attention lets the decoder
    amplify encoder features at object locations and suppress uniform
    background/cloud regions before the skip concatenation.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attn


class DoubleConvBlock(nn.Module):
    """Two Conv -> BN -> ReLU blocks with a residual shortcut and
    channel attention (SE). The 1x1 shortcut lets the block learn a pure
    residual update at every depth level. Reflect padding avoids the
    border artifacts of zero padding on small 128px tiles.
    """

    def __init__(self, in_channels, out_channels, use_se=True, use_dropout=False, p=0.1):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.dropout = nn.Dropout2d(p) if use_dropout else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)
        out = self.dropout(out)
        return self.relu(out + residual)


class CustomUNet(nn.Module):
    """Custom U-Net built from scratch for segmenting
    multi-spectral satellite imagery.

    3 downsampling levels (bottleneck at 16x16 for 128x128 input)
    instead of 4: most landslide blobs are only ~15 pixels (median
    4x4 px) and a 8x8 bottleneck destroys them completely. Keeping the
    spatial resolution higher lets the decoder reconstruct small blobs.

    Upgrades over the previous build:
      - wider filters (96 base instead of 64)  -> more capacity
      - SE channel attention in every block
      - residual shortcuts (ResUNet)           -> deeper gradient flow
      - dropout at the bottleneck              -> noisy-label robustness
      - deep supervision (aux losses on the two coarse decoder levels)
    """

    def __init__(self, in_channels=42, out_channels=1, base_filters=96,
                 deep_supervision=True):
        super(CustomUNet, self).__init__()
        self.deep_supervision = deep_supervision
        b = base_filters

        # --- ENCODER (Downsampling) ---
        self.down1 = DoubleConvBlock(in_channels, b)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConvBlock(b, b * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConvBlock(b * 2, b * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK (with dropout against label noise) ---
        self.bottleneck = DoubleConvBlock(b * 4, b * 8, use_dropout=True, p=0.1)

        # --- DECODER (Upsampling) ---
        self.upconv3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.up3 = DoubleConvBlock(b * 4 + b * 4, b * 4)

        self.upconv2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.up2 = DoubleConvBlock(b * 2 + b * 2, b * 2)

        self.upconv1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.up1 = DoubleConvBlock(b + b, b)

        # --- OUTPUT LAYER ---
        self.final_conv = nn.Conv2d(b, out_channels, kernel_size=1)

        # --- DEEP SUPERVISION HEADS (auxiliary losses on coarse levels) ---
        self.aux_conv2 = nn.Conv2d(b * 2, out_channels, kernel_size=1)
        self.aux_conv3 = nn.Conv2d(b * 4, out_channels, kernel_size=1)

        # Spatial attention on the skip connections
        self.skip_att = nn.ModuleList([SpatialAttention() for _ in range(3)])

        # Start predicting background: zero-initialized heads stop the noisy
        # labels from flooding the first epochs with garbage gradient signal.
        for head in (self.final_conv, self.aux_conv2, self.aux_conv3):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, return_aux=False):
        # 1. Encoder pass
        e1_features = self.down1(x)
        p1 = self.pool1(e1_features)

        e2_features = self.down2(p1)
        p2 = self.pool2(e2_features)

        e3_features = self.down3(p2)
        p3 = self.pool3(e3_features)

        # 2. Bottleneck
        b_features = self.bottleneck(p3)

        # 3. Decoder pass with spatial-attended skip connections
        u3 = self.upconv3(b_features)
        merged3 = torch.cat((u3, self.skip_att[0](e3_features)), dim=1)
        d3 = self.up3(merged3)

        u2 = self.upconv2(d3)
        merged2 = torch.cat((u2, self.skip_att[1](e2_features)), dim=1)
        d2 = self.up2(merged2)

        u1 = self.upconv1(d2)
        merged1 = torch.cat((u1, self.skip_att[2](e1_features)), dim=1)
        d1 = self.up1(merged1)

        # 4. Final output is raw logits. The loss applies sigmoid internally;
        # callers must apply sigmoid for probability maps.
        logits = self.final_conv(d1)

        if return_aux and self.deep_supervision:
            aux2 = F.interpolate(self.aux_conv2(d2), scale_factor=2,
                                 mode="bilinear", align_corners=False)
            aux3 = F.interpolate(self.aux_conv3(d3), scale_factor=4,
                                 mode="bilinear", align_corners=False)
            return logits, [aux2, aux3]
        return logits


# --- Quick architecture test ---
if __name__ == "__main__":
    print("Running U-Net architecture check...")

    model = CustomUNet(in_channels=42, out_channels=1)

    fake_satellite_batch = torch.randn(2, 42, 128, 128)
    print(f"Input tensor shape  : {fake_satellite_batch.shape}")

    predicted = model(fake_satellite_batch)
    print(f"Output mask shape   : {predicted.shape}")

    logits, auxs = model(fake_satellite_batch, return_aux=True)
    print(f"Aux shapes          : {[tuple(a.shape) for a in auxs]}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameter count     : {n_params:.1f}M")

    if predicted.shape[2] == 128 and predicted.shape[3] == 128:
        print("\n------ Model Verification ------")
        print("U-Net compiled and tested successfully.")
        print("Skip connections + SE + residual + deep supervision active.")
        print("Output is raw logits (apply sigmoid for probabilities).")
        print("--------------------------------\n")
    else:
        print("Shape mismatch detected. Check tensor dimensions.")
