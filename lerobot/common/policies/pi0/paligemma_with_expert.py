# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import torch
import torch.version
from pytest import Cache
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from lerobot.common.policies.pi0.flex_attention import flex_attention_forward
from timm.models.vision_transformer import Block
import numpy as np
from slot_attention import SlotAttention

def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        attention_implementation: str = "eager",
        n_action_steps: int = None,
        use_explicit: bool = False,
        use_slot_att: bool = False,
        num_view_images: int = None,
        n_exp_toks_view: int = None,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.attention_implementation = attention_implementation
        self.n_action_steps = n_action_steps
        self.num_view_images = num_view_images

        self.use_explicit = use_explicit
        self.use_slot_att = use_slot_att
        self.n_exp_toks_view = n_exp_toks_view

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    # "vision_use_head": False,
                    "vision_use_head": True,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        # if self.train_expert_only and not self.freeze_vision_encoder:
        #     raise ValueError(
        #         "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
        #     )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Tracker_Model(nn.Module):
    def __init__(self, hidden_dim, num_obs_per_image):
        super().__init__()
        
        self.NUM_MASK_TOKEN = 256
        self.num_obs_per_image = num_obs_per_image
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) 
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.image_decoder_obs_pred_projector = nn.Linear(hidden_dim, hidden_dim)
        
        self.image_decoder_position_embedding = nn.Parameter(torch.zeros(1, num_obs_per_image + self.NUM_MASK_TOKEN, hidden_dim), requires_grad=False) 
        image_decoder_position_embedding_obs = get_2d_sincos_pos_embed(hidden_dim, int(num_obs_per_image**.5), cls_token=False)
        image_decoder_position_embedding_mask = get_2d_sincos_pos_embed(hidden_dim, int(self.NUM_MASK_TOKEN**.5), cls_token=False)
        image_decoder_position_embedding = np.concatenate((image_decoder_position_embedding_obs, image_decoder_position_embedding_mask), axis=0)
        self.image_decoder_position_embedding.data.copy_(torch.from_numpy(image_decoder_position_embedding).float().unsqueeze(0))
        
        self.image_decoder = nn.Sequential(
            Block(hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            Block(hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            )
        self.image_decoder_norm = nn.LayerNorm(hidden_dim)
        self.image_decoder_pred = nn.Linear(hidden_dim, 1152)

    def forward(self, track_tokens):

        # track_tokens: [B, num_views*num_tokens_per_images, dim]

        B, NUM_OBS_TOKEN, dim = track_tokens.shape
        assert NUM_OBS_TOKEN % self.num_obs_per_image == 0

        NUM_OBS_TOKEN_PER_IMAGE = self.num_obs_per_image

        obs_pred_embedding = self.image_decoder_obs_pred_projector(track_tokens.reshape(-1, track_tokens.shape[-1]))
        obs_pred_embedding = obs_pred_embedding.view(B * (NUM_OBS_TOKEN // NUM_OBS_TOKEN_PER_IMAGE), NUM_OBS_TOKEN_PER_IMAGE, dim)
        mask_tokens = self.mask_token.repeat(B * (NUM_OBS_TOKEN // NUM_OBS_TOKEN_PER_IMAGE), self.NUM_MASK_TOKEN, 1)
        image_decoder_input = torch.cat((obs_pred_embedding, mask_tokens), dim=1)
        image_decoder_input = image_decoder_input + self.image_decoder_position_embedding
        image_decoder_output = self.image_decoder(image_decoder_input) 
        image_pred_feature = image_decoder_output[:, -self.NUM_MASK_TOKEN:, :] 
        image_pred_feature = self.image_decoder_norm(image_pred_feature.reshape(-1, dim))
        image_pred = self.image_decoder_pred(image_pred_feature)  
        image_pred = image_pred.view(B, NUM_OBS_TOKEN // NUM_OBS_TOKEN_PER_IMAGE, 1, self.NUM_MASK_TOKEN, -1)  
        return image_pred

class Qformer(nn.Module):
    def __init__(self, num_slots, dim, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots

        self.slots_img = nn.Parameter(torch.randn(1, self.num_slots, dim))
        self.slots_wrist = nn.Parameter(torch.randn(1, self.num_slots, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        hidden_dim = max(dim // 4, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def attn(self, Q, K, V):
        # Q: [B, M, d], K,V: [B, N, d]
        scores = (Q @ K.transpose(-1, -2)) / (self.dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return weights @ V   # [B, M, d]

    def forward(self, inputs, typ):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        if typ == "img": 
            slots = self.norm_slots(self.slots_img.expand(b, self.num_slots, self.dim))
        elif typ == "wrist":
            slots = self.norm_slots(self.slots_wrist.expand(b, self.num_slots, self.dim))
        q = self.to_q(slots)

        outs = self.attn(q, k, v)
        outs = outs + self.mlp(self.norm_pre_ff(outs))

        return outs

class VAPT(nn.Module):
   
    def __init__(self, d:int, Np:int, Nprime:int, r:int):
        """
        d       = embedding dimension
        Np      = number of prompt tokens to generate
        Nprime  = N' = number of conv tokens = H'*W'
        r       = hidden dim of feature projector (r << d)
        """
        super().__init__()
        self.Np = Np
        self.Nprime = Nprime
        self.d = d

        # token-wise projectors α: (Np, N')
        self.alpha_img = nn.Parameter(torch.randn(Np, Nprime) * 0.02)
        self.alpha_wrist = nn.Parameter(torch.randn(Np, Nprime) * 0.02)

        # shared feature projector g
        self.W1 = nn.Linear(d, r, bias=True)
        self.W2 = nn.Linear(r, d, bias=True)
        self.act = nn.ReLU()

        # init
        nn.init.xavier_uniform_(self.W1.weight); nn.init.zeros_(self.W1.bias)
        nn.init.xavier_uniform_(self.W2.weight); nn.init.zeros_(self.W2.bias)

    def forward(self, x, typ):
        """
        x: (B, N', d)
        return: P: (B, Np, d)
        """
        B, Np_, d_ = x.shape
        assert d_ == self.d
        assert Np_ == self.Nprime

        # G: (B, Np, d) = alpha (Np,N') ⋅ x (B,N',d)
        if typ == "img":
            G = torch.einsum('jk,bkd->bjd', self.alpha_img, x)
        elif typ == "wrist":
            G = torch.einsum('jk,bkd->bjd', self.alpha_wrist, x)

        H = self.act(self.W1(G))
        P = self.W2(H) + G

        return P

class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)

        if self.config.use_explicit:
            print("################################# Use Explicit Module #################################")
            num_obs_per_image = self.config.n_exp_toks_view
            self.num_obs_token = num_obs_per_image*self.config.num_view_images
            track_hidden_dim = self.paligemma.language_model.model.layers[17].self_attn.q_proj.out_features
            if self.config.use_slot_att == "slot":
                self.slot_attn = SlotAttention(
                    num_slots = num_obs_per_image,
                    dim = track_hidden_dim,
                    iters = 3   # iterations of attention, defaults to 3
                )
            elif self.config.use_slot_att == "qformer":
                self.slot_attn = Qformer(num_slots = num_obs_per_image, 
                                        dim = track_hidden_dim)
            elif self.config.use_slot_att == "mlp":
                self.slot_attn = VAPT(d = track_hidden_dim, Np = num_obs_per_image, 
                                        Nprime = 256, r = track_hidden_dim // 4)
            else:
                self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.num_obs_token, track_hidden_dim)) # (1, 1, 18, 2048)
            self.tracker_model = Tracker_Model(hidden_dim=track_hidden_dim, num_obs_per_image=num_obs_per_image)

        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
            "tracker_model",
            "slot_attn"
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def get_image_features(self, pixel_values, return_raw_pool):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.paligemma.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.paligemma.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.paligemma.config.text_config.hidden_size**0.5)

        # pooler output
        pooler_output = image_outputs.pooler_output
        if return_raw_pool:
            return image_features, pooler_output
        else:
            pooler_output_features = self.paligemma.multi_modal_projector(pooler_output)
            pooler_output_features = pooler_output_features / (self.paligemma.config.text_config.hidden_size**0.5)
            return image_features, pooler_output_features

    def embed_image_return_pooling(self, image: torch.Tensor, return_raw_pool):
        return self.get_image_features(image, return_raw_pool)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.model.embed_tokens(tokens)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        models = [self.paligemma.language_model.model, self.gemma_expert.model]

        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        if self.config.use_explicit:
            if inputs_embeds[0] is not None:
                if self.config.use_slot_att in ["slot", "qformer", "mlp"]:
                    print(f"==============Use slot attention: {self.config.use_slot_att}=========")
                    imgs_only = inputs_embeds[0][:, 0:256, :]
                    # imgs_text = torch.cat([imgs_only, inputs_embeds[0][:, 512:, :]], dim=1)
                    imgs_text = imgs_only
                    # wrist_text = inputs_embeds[0][:, 256:, :]
                    wrist_text = inputs_embeds[0][:, 256:512, :]
                    imgs_slots = self.slot_attn(imgs_text, typ = "img")
                    wrist_slots = self.slot_attn(wrist_text, typ = "wrist")
                    obs_tokens = torch.cat([imgs_slots, wrist_slots], dim=1)
                    # obs_tokens = self.slot_attn(inputs_embeds[0])
                    inputs_embeds[0] = torch.cat((inputs_embeds[0], obs_tokens), dim = 1)
                else:
                    # print("==============Do not use slot attention=========")
                    inputs_embeds[0] = torch.cat((inputs_embeds[0], self.obs_tokens.repeat(batch_size, 1, 1, 1).flatten(1, 2)), dim = 1)
            # attention_mask = insert_track_attention_mask(attention_mask, self.num_obs_token)

        # RMSNorm
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        head_dim = self.paligemma.config.text_config.head_dim
        outputs_embeds_after_o_proj = []
        attention_mask_ori = attention_mask.clone()
        
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]
                # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                # hidden_states = hidden_states * normalizer
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask, batch_size, head_dim, query_states, key_states, 
                value_states
            )
            att_output = att_output.to(dtype=torch.bfloat16)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # TODO: first dropout (by default 0.0)

                    # first residual
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout (by default 0.0)

                    # second residual
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)
                    if layer_idx == num_layers-1:
                        outputs_embeds_after_o_proj.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)
                    if layer_idx == num_layers-1:
                        outputs_embeds_after_o_proj.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        decoder_output = None
        if self.config.use_explicit and outputs_embeds[0] is not None and outputs_embeds[1] is not None:

            track_output = outputs_embeds[0][:, -self.num_obs_token:, :]
            decoder_output = self.tracker_model(track_output)

        return outputs_embeds, past_key_values, decoder_output, outputs_embeds_after_o_proj

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        elif self.config.attention_implementation == "flex":
            attention_interface = flex_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        
        num_att_heads = self.config.paligemma_config.text_config.num_attention_heads
        num_key_value_heads = self.config.paligemma_config.text_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads
        
        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)


        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)

        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
