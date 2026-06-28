import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    """A helper block that runs two consecutive Conv -> BatchNorm -> ReLU
    operations. This stabilizes feature extraction at each depth level of the U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            # First conv layer
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv layer
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CustomUNet(nn.Module):
    """Custom U-Net built from scratch for segmenting
    multi-spectral satellite imagery.
    """

    def __init__(self, in_channels=14, out_channels=1):
        super(CustomUNet, self).__init__()

        # --- ENCODER (Downsampling) ---
        self.down1 = DoubleConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK ---
        self.bottleneck = DoubleConvBlock(256, 512)

        # --- DECODER (Upsampling) ---
        # ConvTranspose2d doubles the spatial resolution at each step
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # After concatenating the skip connection: 256 + 256 = 512 input channels
        self.up3 = DoubleConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConvBlock(128, 64)

        # --- OUTPUT LAYER ---
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. Encoder pass
        e1_features = self.down1(x)  # saved for skip connection 1
        p1 = self.pool1(e1_features)

        e2_features = self.down2(p1)  # saved for skip connection 2
        p2 = self.pool2(e2_features)

        e3_features = self.down3(p2)  # saved for skip connection 3
        p3 = self.pool3(e3_features)

        # 2. Bottleneck
        b_features = self.bottleneck(p3)

        # 3. Decoder pass with skip connection concatenation
        u3 = self.upconv3(b_features)
        # Concatenate encoder features with upsampled features along channel dim
        merged3 = torch.cat((u3, e3_features), dim=1)
        d3 = self.up3(merged3)

        u2 = self.upconv2(d3)
        merged2 = torch.cat((u2, e2_features), dim=1)
        d2 = self.up2(merged2)

        u1 = self.upconv1(d2)
        merged1 = torch.cat((u1, e1_features), dim=1)
        d1 = self.up1(merged1)

        # 4. Final output passed through sigmoid for binary segmentation
        logits = self.final_conv(d1)
        return torch.sigmoid(logits)


# --- Quick architecture test ---
if __name__ == "__main__":
    print("Running U-Net architecture check...")

    # in_channels=14 because Landslide4Sense has 14 spectral bands
    model = CustomUNet(in_channels=14, out_channels=1)

    # Dummy batch to test forward pass: (batch_size, channels, height, width)
    fake_satellite_batch = torch.randn(2, 14, 128, 128)

    print(f"Input tensor shape  : {fake_satellite_batch.shape}")

    # Run forward pass
    predicted_output_mask = model(fake_satellite_batch)

    print(f"Output mask shape   : {predicted_output_mask.shape}")

    # Check that output spatial dimensions match the input
    if (
        predicted_output_mask.shape[2] == 128
        and predicted_output_mask.shape[3] == 128
    ):
        print("\n------ Model Verification ------")
        print("U-Net compiled and tested successfully.")
        print("Skip connections working as expected.")
        print("Model is ready for training.")
        print("--------------------------------\n")
    else:
        print("Shape mismatch detected. Check tensor dimensions.")