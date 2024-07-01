import contextlib

from einops import rearrange
import gradio as gr
from ldm.util import default

from modules import scripts, sd_hijack
from modules import shared, errors, devices, sub_quadratic_attention
from modules import sd_hijack_unet
from modules.hypernetworks import hypernetwork
from modules.sd_hijack import undo_optimizations
from modules.sd_hijack_optimizations import (
    SdOptimization,
    list_optimizers,
    sub_quad_attnblock_forward,
)

from torch.nn.functional import silu

import ldm.modules.attention
import ldm.modules.diffusionmodules.model

import sgm.modules.attention
import sgm.modules.diffusionmodules.model

import torch
import os

os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
from scripts.fattn_kernel import flash_attn_wmma

flash_attn_name = "Flash Attention v2 (wmma)"

def flash_attention_forward(self, x, context=None, mask=None, **kwargs):

    h = self.heads
    q_in = self.to_q(x)
   
    context = default(context, x)
    context_k, context_v = hypernetwork.apply_hypernetworks(
        shared.loaded_hypernetworks, context
    )
    
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)
    
    q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype

    q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)

    sc = q.shape[-1] ** -0.5

    Br = 64
    Bc = None

    d_qkv = q.shape[-1]
    d_pad_to = 16
    d_pad_sz = ((d_qkv + d_pad_to - 1) // d_pad_to) * d_pad_to - d_qkv

    def prev_power_of_2(n: int):
        i = 1
        while 2**i < n:
            i += 1
        return 2 ** (i - 1)

    n_kv = k.shape[2]
    if n_kv >= 512:
        n_pad_to = 512
        n_pad_sz = ((n_kv + n_pad_to - 1) // n_pad_to) * n_pad_to  - n_kv 
        Bc = 512
        Br = 64
    else:
        n_pad_to = 16
        n_pad_sz = ((n_kv + n_pad_to - 1) // n_pad_to) * n_pad_to  - n_kv 
        Bc = n_pad_sz + n_kv
        Br = min(prev_power_of_2(32768 // Bc), q.shape[2])
    
    k = torch.nn.functional.pad(k, (0, d_pad_sz, n_pad_sz , 0 ,0, 0), mode='constant', value=0) 
    v = torch.nn.functional.pad(v, (0, d_pad_sz, n_pad_sz , 0 ,0, 0), mode='constant', value=0)     
    q = torch.nn.functional.pad(q, (0, d_pad_sz, 0 , 0 ,0, 0), mode='constant', value=0) 
    
    out = flash_attn_wmma.forward(q, k, v, Br, Bc)[:, :, :, :d_qkv]

    out = out.to(dtype)

    out = rearrange(out, "b h n d -> b n (h d)", h=h)
    return self.to_out(out)


def flash_attnblock_forward(self, x):

    try:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)

        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(
            lambda t: t.view(b, 1, c, -1).transpose(2, 3).contiguous(),
            (q, k, v),
        )

        dtype = q.dtype

        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
 
        sc = q.shape[-1] ** -0.5
        out = flash_attn_wmma.forward(q,k,v,64,128)

        out = out.to(dtype)

        out = out.transpose(2, 3).reshape(b, c, h, w)
        out = self.proj_out(out)
        return x + out
    except:
        return sub_quad_attnblock_forward(self, x)

force_set_optimizer = False
from modules.sd_hijack import optimizers, current_optimizer
def apply_optimizations_hijack(option=None):
    global current_optimizer, optimizers 
    if force_set_optimizer:
        print("force applying attention optimization to:", current_optimizer.name)
        current_optimizer.apply()
        return current_optimizer.name

    undo_optimizations()

    if len(optimizers) == 0:
        # a script can access the model very early, and optimizations would not be filled by then
        current_optimizer = None
        return ''

    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    sgm.modules.diffusionmodules.model.nonlinearity = silu
    sgm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    if current_optimizer is not None:
        current_optimizer.undo()
        current_optimizer = None

    selection = option or shared.opts.cross_attention_optimization
    if selection == "Automatic" and len(optimizers) > 0:
        matching_optimizer = next(iter([x for x in optimizers if x.cmd_opt and getattr(shared.cmd_opts, x.cmd_opt, False)]), optimizers[0])
    else:
        matching_optimizer = next(iter([x for x in optimizers if x.title() == selection]), None)

    if selection == "None":
        matching_optimizer = None
    elif selection == "Automatic" and shared.cmd_opts.disable_opt_split_attention:
        matching_optimizer = None
    elif matching_optimizer is None:
        matching_optimizer = optimizers[0]

    if matching_optimizer is not None:
        print(f"Applying attention optimization: {matching_optimizer.name}... ", end='')
        matching_optimizer.apply()
        print("done.")
        current_optimizer = matching_optimizer
        return current_optimizer.name
    else:
        print("Disabling attention optimization")
        return ''
    
sd_hijack.apply_optimizations = apply_optimizations_hijack
setattr(sd_hijack,"apply_optimizations", apply_optimizations_hijack)

class SdOptimizationFattn(SdOptimization):
    name = "flash_attn_v2_rocWMMA"
    label = "flash attention v2 rocWMMA"
    cmd_opt = "opt_flash_attention_rocwmma"
    priority = 80

    def is_available(self):
        return True

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = flash_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = flash_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = flash_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = flash_attnblock_forward


class AttentionSelectorPlugin(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.availableSDOptimizations = []
        self.availableSDOptimizations_name = []
        list_optimizers(self.availableSDOptimizations)

        self.availableSDOptimizations_name.append(flash_attn_name)

        for n in self.availableSDOptimizations:
            if hasattr(n, "is_available"):
                if n.is_available():
                    self.availableSDOptimizations_name.append(n.name)
            else:
                self.availableSDOptimizations_name.append(n.name)

    def set_optimizer(self, select_optim):
        global current_optimizer, force_set_optimizer
        force_set_optimizer = True
        # print(select_optim)
        if select_optim == flash_attn_name:
            flashAttnOptim = SdOptimizationFattn()
            setattr(shared.cmd_opts, flashAttnOptim.cmd_opt, True)
            flashAttnOptim.apply()
            current_optimizer = flashAttnOptim
            gr.Info(f"Applied attention optimization: {flash_attn_name}")
            return select_optim

        for n in self.availableSDOptimizations:
            if n.name == select_optim:
                print(f"Applying attention optimization: {n.name}... ", end="")
                n.apply()
                current_optimizer = n
                print("done.")
                gr.Info(f"Applied attention optimization: {n.name}")
                return select_optim
        return select_optim

    def title(self):
        return "Attention-Selector-Plugin"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Cross-Attention Algorithm Selector", open=False):
                self.types_to_sent = gr.Dropdown(
                    self.availableSDOptimizations_name, label="Optimization Algorithm"
                )
                self.send_text_button = gr.Button(value="SET", variant="primary")

        with contextlib.suppress(AttributeError):
            self.send_text_button.click(
                fn=self.set_optimizer, inputs=[self.types_to_sent]
            )
        return [self.send_text_button, self.types_to_sent]
