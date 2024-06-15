import contextlib

from einops import rearrange
import gradio as gr
from ldm.util import default
from modules import scripts
from modules import shared, errors, devices, sub_quadratic_attention
from modules.hypernetworks import hypernetwork
from modules.sd_hijack_optimizations import (
    list_optimizers,
    sub_quad_attention_forward,
    sub_quad_attnblock_forward,
)

import ldm.modules.attention
import ldm.modules.diffusionmodules.model

import sgm.modules.attention
import sgm.modules.diffusionmodules.model

import torch
import os

os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
from scripts.ck_mha import minimal_attn

large_d = 0
small_d = 0

ck_tile_attn_name = "Flash Attention (ck_tile) [ZLUDA]"

def ck_tile_attention_forward(self, x, context=None, mask=None, **kwargs):
    global large_d, small_d

    h = self.heads
    q_in = self.to_q(x)


    if (q_in.shape[-1] // h) > 128:
        large_d += 1
        if large_d % 32 == 0:
            print(
                f"Tensor dimsize {q_in.shape[-1]//h} over 128, fallback Sub-Quad...",
                large_d / (large_d + small_d),
            )
        del q_in
        return sub_quad_attention_forward(self, x, context, mask)
    
    context = default(context, x)
    context_k, context_v = hypernetwork.apply_hypernetworks(
        shared.loaded_hypernetworks, context
    )
    
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)
    small_d += 1

    q, k, v = (rearrange(t, "b n (h d) -> b n h d", h=h) for t in (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype

    q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

    sc = q.shape[-1] ** -0.5
    out = minimal_attn.fwd(q, k, v, None, 0, sc, False, False, None)[0]

    out = out.to(dtype)

    out = rearrange(out, "b n h d -> b n (h d)", h=h)
    return self.to_out(out)


def ck_tile_attnblock_forward(self, x):
    global large_d, small_d

    try:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)

        if q.shape[1] > 128:
            large_d += 1
            if large_d % 32 == 0:
                print(
                    f"Tensor dimsize {q.shape[1]} over 128, fallback Sub-Quad... [attnblock]",
                    large_d / (large_d + small_d),
                )
            return sub_quad_attnblock_forward(self, x)
        k = self.k(h_)
        v = self.v(h_)
        small_d += 1

        b, c, h, w = q.shape
        q, k, v = map(
            lambda t: t.view(b, 1, c, -1).transpose(2, 3),
            (q, k, v),
        )

        dtype = q.dtype

        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        sc = q.shape[-1] ** -0.5
        out = minimal_attn.fwd(q, k, v, None, 0, sc, False, False, None)[0]

        out = out.to(dtype)

        out = out.transpose(2, 3).reshape(b, c, h, w)
        out = self.proj_out(out)
        return x + out
    except:
        return sub_quad_attnblock_forward(self, x)


class AttentionSelectorPlugin(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.availableSDOptimizations = []
        self.availableSDOptimizations_name = []
        list_optimizers(self.availableSDOptimizations)

        self.availableSDOptimizations_name.append(ck_tile_attn_name)

        for n in self.availableSDOptimizations:
            if hasattr(n, "is_available"):
                if n.is_available():
                    self.availableSDOptimizations_name.append(n.name)
            else:
                self.availableSDOptimizations_name.append(n.name)

    def send_text_to_prompt(self, select_optim):

        # print(select_optim)
        if select_optim == ck_tile_attn_name:
            ldm.modules.attention.CrossAttention.forward = ck_tile_attention_forward
            ldm.modules.diffusionmodules.model.AttnBlock.forward = (
                ck_tile_attnblock_forward
            )
            sgm.modules.attention.CrossAttention.forward = ck_tile_attention_forward
            sgm.modules.diffusionmodules.model.AttnBlock.forward = (
                ck_tile_attnblock_forward
            )
            gr.Info(f"Applied attention optimization: {ck_tile_attn_name}")
            return select_optim

        for n in self.availableSDOptimizations:
            if n.name == select_optim:
                print(f"Applying attention optimization: {n.name}... ", end="")
                n.apply()
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
                fn=self.send_text_to_prompt, inputs=[self.types_to_sent]
            )
        return [self.send_text_button, self.types_to_sent]
