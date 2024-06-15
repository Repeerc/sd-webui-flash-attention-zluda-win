# sd-webui-flash-attention-zluda-win
A flash attention extension for stable diffusion webui with AMD ZLUDA (gfx11xx, Windows) environments.

一个为 AMD GPU Windows ZLUDA 环境提供Flash attention优化方案的stable diffusion webui扩展插件


# 目前仅支持 RX7000 系显卡和 Python 3.10 + PyTorch 2.2.1 + CUDA 11.8 环境
# Currently only supports RX7000, Python 3.10 + Pytorch 2.2.1 + CUDA 11.8

Flash attention 编译自： Flash attention Compiled From:
[https://github.com/ROCm/flash-attention/blob/howiejay/navi_support](https://github.com/ROCm/flash-attention/tree/howiejay/navi_support)

### 其它限制 (Limitations)
无法计算大于128 dimsize的张量，对应SD1.5模型则回落至Sub-quad算法处理（约占总计算量的37%），SDXL未发现该问题

Unable to compute tensors larger than 128 dimsize, the SD1.5 model would falls back to the Sub-quad (about 37% of the total computation)

## 使用方法 (How to use)

1.从本仓库安装插件。 Install extensions from this repository.

2.选择 Flash Attention 算法，然后点击SET激活。 Select the Flash Attention algorithm and click SET to activate it.

![image](https://github.com/Repeerc/sd-webui-flash-attention-zluda-win/assets/7540581/4bcdbfb4-be61-45c8-96ba-764d5dcc7fbc)

## 测试 (Testing)

GPU: AMD RX7900 XTX 24GB
CPU: Intel 13900T

Launching arguments with: --medvram-sdxl 

SD1.5 Model:cuteyukimixAdorable_neochapter[6ee4f31532], VAE:vae-ft-mse-840000-ema-pruned

SDXL Model:animagineXLV3_v30[1449e5b0b9], VAE:None

Clip skip:2

Tile VAE: Encoder Tile Size = 1536, Decoder Tile Size = 128

### SDXL
DPM++, 2M, 1024x1024, 50 Steps, BatchSize = 1
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 2.71 it/s  |  3.10 it/s |  |
| VRAM (shown in Task manager)  |  7.22 GB  | 6.68 GB |  |


DPM++, 2M, 1024x1536, 50 Steps, BatchSize = 2
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 1.70 it/s  |  1.99 it/s |  |
| VRAM (shown in Task manager) |  15.12 GB  | 8.37 GB |  |

### SD1.5

Tile VAE: Encoder Tile Size = 1536, Decoder Tile Size = 64

DPM++, 2M, 512x512, 50 Steps, BatchSize = 4
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 4.36 it/s  |  4.16 it/s | flash_attn fall back to sub_quad when dimsize > 128 |
| VRAM (shown in Task manager) |  9.08 GB  | 3.23 GB |  |


768x768, BatchSize = 2
|   | Pytorch SDP  | Flash Attention |       |
|:---------:|--------:|--------:|--------:|
| Speed | 2.85 it/s  |  2.76 it/s | flash_attn fall back to sub_quad when dimsize > 128 |
| VRAM (shown in Task manager) |  18.78 GB  | 3.33 GB |  |






