"""GAN model definitions and utilities for image-to-image translation and super-resolution.

This module provides:
- CycleGAN generator architecture for season transfer (summer ↔ winter)
- ESRGAN utilities for 4× super-resolution using ONNX Runtime
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image


class HFConvBlock(nn.Module):
    """Convolutional block with optional transposed convolution, instance norm, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: bool = True,
        transpose: bool = False,
        output_padding: int = 0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if transpose:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=True,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True,
                )
            )
        layers.append(nn.InstanceNorm2d(out_channels, affine=False))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HFResidualBlock(nn.Module):
    """Residual block with two convolutional blocks and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            HFConvBlock(channels, channels, kernel_size=3, padding=1),
            HFConvBlock(channels, channels, kernel_size=3, padding=1, activation=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HuggingFaceGenerator(nn.Module):
    """CycleGAN generator architecture compatible with masquerlin/cycleGAN_winter_summer weights.

    Architecture:
    - Initial conv layer (7×7)
    - 2 downsampling blocks (stride 2)
    - 9 residual blocks
    - 2 upsampling blocks (transposed conv)
    - Output conv layer (7×7) with tanh
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=True),
            nn.InstanceNorm2d(ngf, affine=False),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                HFConvBlock(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                HFConvBlock(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_blocks = nn.ModuleList([HFResidualBlock(ngf * 4) for _ in range(n_blocks)])

        self.up_blocks = nn.ModuleList(
            [
                HFConvBlock(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, transpose=True, output_padding=1),
                HFConvBlock(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, transpose=True, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        for block in self.down_blocks:
            x = block(x)
        for block in self.residual_blocks:
            x = block(x)
        for block in self.up_blocks:
            x = block(x)
        return torch.tanh(self.last(x))


# CycleGAN pretrained model configuration
CYCLEGAN_REPO = "masquerlin/cycleGAN_winter_summer"
CYCLEGAN_FILES = {
    "summer2winter": "gen_a_new.pth",
    "winter2summer": "gen_b_new.pth",
}
_CYCLEGAN_CACHE: dict[str, HuggingFaceGenerator] = {}


def load_cyclegan_generator(direction: Literal["summer2winter", "winter2summer"]) -> HuggingFaceGenerator:
    """Load a pretrained CycleGAN generator from Hugging Face Hub.

    Args:
        direction: Either "summer2winter" or "winter2summer"

    Returns:
        Generator model in eval mode
    """
    if direction in _CYCLEGAN_CACHE:
        return _CYCLEGAN_CACHE[direction]

    weight_file = hf_hub_download(repo_id=CYCLEGAN_REPO, filename=CYCLEGAN_FILES[direction])
    state = torch.load(weight_file, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = HuggingFaceGenerator()
    model.load_state_dict(state)
    model.eval()
    _CYCLEGAN_CACHE[direction] = model
    return model


# ESRGAN super-resolution configuration
ESRGAN_REPO = "qualcomm/ESRGAN"
ESRGAN_ZIP = "ESRGAN_float.onnx.zip"
ESRGAN_SCALE = 4


def get_esrgan_onnx_path() -> Path:
    """Download and extract ESRGAN ONNX model from Hugging Face Hub.

    Returns:
        Path to the extracted .onnx file
    """
    zip_path = Path(hf_hub_download(repo_id=ESRGAN_REPO, filename=ESRGAN_ZIP))
    expected_file = zip_path.with_suffix("").with_suffix(".onnx")
    if expected_file.exists():
        return expected_file

    with zipfile.ZipFile(zip_path, "r") as archive:
        onnx_files = [name for name in archive.namelist() if name.endswith(".onnx")]
        if not onnx_files:
            raise FileNotFoundError("ESRGAN zip does not contain an .onnx file")
        # Extract all files from the archive to maintain directory structure
        archive.extractall(path=zip_path.parent)
        extracted_path = zip_path.parent / onnx_files[0]
    return extracted_path


def get_esrgan_session():
    """Create an ONNX Runtime inference session for ESRGAN.

    Returns:
        onnxruntime.InferenceSession configured for CPU execution
    """
    import onnxruntime as ort

    onnx_path = get_esrgan_onnx_path()
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def prepare_esrgan_input(image: Image.Image) -> np.ndarray:
    """Convert PIL image to ESRGAN input format.

    Args:
        image: RGB PIL image

    Returns:
        Numpy array of shape (1, 3, H, W) with values in [0, 1]
    """
    array = np.array(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return array


def run_esrgan(sr_session, image: Image.Image, target_size: int = 128) -> Image.Image:
    """Run ESRGAN super-resolution on a PIL image.

    Args:
        sr_session: ONNX Runtime inference session
        image: Low-resolution RGB PIL image
        target_size: Expected input size for the model (default: 128)

    Returns:
        Super-resolved PIL image (4× resolution)
    """
    # Resize image to expected input size if needed
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.BICUBIC)
    
    lr = prepare_esrgan_input(image)
    inputs = {sr_session.get_inputs()[0].name: lr}
    output = sr_session.run(None, inputs)[0][0]
    output = np.clip(output, 0.0, 1.0)
    output = np.transpose(output, (1, 2, 0))
    return Image.fromarray((output * 255).astype(np.uint8))
