# sd-webui-flash-attention-zluda-win
A flash attention extension for stable diffusion webui with AMD ZLUDA (gfx11xx, Windows) environments.

一个为 AMD GPU Windows ZLUDA 环境提供Flash attention优化方案的stable diffusion webui扩展插件


# 目前仅支持 RX7000 系显卡和 Python 3.10 + PyTorch 2.2.1 + CUDA 11.8 环境
# Currently only supports RX7000, Python 3.10 + Pytorch 2.2.1 + CUDA 11.8

Flash attention 编译自： Flash attention Compiled From:
[https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal](https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal)


## 使用方法 (How to use)

1.从本仓库安装插件。 Install extensions from this repository.

2.选择 Flash Attention 算法，然后点击SET激活。 Select the Flash Attention algorithm and click SET to activate it.

![image](https://github.com/Repeerc/sd-webui-flash-attention-zluda-win/assets/7540581/4bcdbfb4-be61-45c8-96ba-764d5dcc7fbc)

## 测试 (Testing)

GPU: AMD RX7900 XTX 24GB

CPU: Intel 13900T

SD1.5 Model:cuteyukimixAdorable_neochapter[6ee4f31532], VAE:vae-ft-mse-840000-ema-pruned

SDXL Model:animagineXLV3_v30[1449e5b0b9], VAE:None

Clip skip:2

### SDXL

DPM++, 2M, 1024x1024, 50 Steps, BatchSize = 1
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 2.75 it/s  |  3.04 it/s |  |
| VRAM (shown in Task manager)  |  9.1 GB  | 8.6 GB |  |


DPM++, 2M, 1024x1536, 50 Steps, BatchSize = 1
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 1.70 it/s  |  1.93 it/s |  |
| VRAM (shown in Task manager) |  10.95 GB  | 9.1 GB |  |

### SD1.5

Tile VAE: Encoder Tile Size = 1536, Decoder Tile Size = 64

DPM++, 2M, 512x512, 50 Steps, BatchSize = 4
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 4.53 it/s  |  4.64 it/s |  |
| VRAM (shown in Task manager) |  9.08 GB  | 3.5 GB |  |


768x768, BatchSize = 2
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 2.92 it/s  |  3.36 it/s | |
| VRAM (shown in Task manager) |  19.0 GB  | 3.69 GB |  |


960x960, BatchSize = 1
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 2.91 it/s  |  3.36 it/s | |
| VRAM (shown in Task manager) |  15.9 GB  | 3.61 GB |  |





