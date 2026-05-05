# -*- coding: utf-8 -*-
# Training script for Spatial-TTT (Streaming Visual-based Spatial Intelligence with Test-Time Training)
# Adapted from train_qwen.py

import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformers.feature_extraction_utils import BatchFeature

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.insert(0, str(project_root / "models"))
from types import MethodType
from typing import Union

import numpy as np
import safetensors.torch
import torch
import transformers
from peft import PeftModel
from PIL import Image
from safetensors import safe_open
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    TensorType,
)
from transformers.video_utils import group_videos_by_shape, reorder_videos

transformers.logging.set_verbosity_error()

# Import LaCT layer
from trainer import replace_qwen2_vl_attention_class
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.utils import is_torchdynamo_compiling
from transformers.video_utils import load_video

from models.causal_swa_lact import Qwen3VLLaCTSWIGLULayer
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import ModelArguments

TRAINING_ARGS_NAME = "training_args.bin"

DEFAULT_LACT_CONFIG = {
    "num_lact_heads": 4,
    "inter_multi": 1.0,
    "lact_chunk_size": 2560,
    "window_size": 2560,
    "qkv_silu": True,
    "no_v_silu": False,
    "use_muon": True,
    "lr_dim": 1,
    "lr_parameterization": "mamba",
    "learnable_ttt_scale": True,
    "w0_w2_low_rank": 32,
    "fw_init_gain": 0.5,
    "use_momentum": True,
    "ttt_prenorm": True,
    "ttt_nope": False,
    "use_fused_kernel": True,
    "fp32_states": True,
}


@dataclass
class SpatialTTTArguments:
    lact_enable: bool = True
    num_lact_heads: int = 4
    inter_multi: float = 1.0
    lact_chunk_size: int = 2560
    window_size: int = 2560
    window_decay: bool = False
    qkv_silu: bool = True
    no_v_silu: bool = False
    use_muon: bool = True
    learnable_ttt_scale: bool = True
    w0_w2_low_rank: int = 32
    use_momentum: bool = True
    ttt_prenorm: bool = True
    ttt_nope: bool = False
    use_fused_kernel: bool = True
    fp32_states: bool = True
    lact_layers: Optional[str] = None
    lact_lr: Optional[float] = None
    use_conv_layer: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    optim: str = "adamw_torch"
    model_max_length: int = 512
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    min_lr_rate: float = 0.1


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    biased_sampling: bool = False
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = 0  # deprecated
    video_min_pixels: int = 0  # deprecated
    video_fps: float = 2
    resize_height: int = 480
    resize_width: int = 640


def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
    backend = "torchcodec"
    if isinstance(video_url_or_urls, list):
        return list(
            zip(
                *[
                    self.fetch_videos(x, sample_indices_fn=sample_indices_fn)
                    for x in video_url_or_urls
                ]
            )
        )

    if video_url_or_urls.endswith(".pt"):
        torchfile = torch.load(video_url_or_urls, weights_only=False)
        return torchfile["video"], torchfile["metadata"]

    return load_video(
        video_url_or_urls, backend=backend, sample_indices_fn=sample_indices_fn
    )


def processor_call(
    self,
    images=None,
    text=None,
    videos=None,
    **kwargs,
):
    output_kwargs = self._merge_kwargs(
        Qwen3VLProcessorKwargs,
        tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        **kwargs,
    )

    if images is not None:
        assert isinstance(images[0], list)
        videos = [
            [
                Image.open(img_path).convert("RGB")
                for img_path in img_seq
                for _ in range(2)
            ]
            for img_seq in images
        ]
        videos_inputs = self.video_processor(
            videos=videos, **output_kwargs["videos_kwargs"]
        )
        video_grid_thw = videos_inputs["video_grid_thw"]

        if "return_metadata" not in kwargs:
            video_metadata = videos_inputs.pop("video_metadata")
        else:
            video_metadata = videos_inputs["video_metadata"]
        video_grid_thw = videos_inputs["video_grid_thw"]

    image_inputs = {}
    image_grid_thw = None

    if videos is not None:
        videos_inputs = self.video_processor(
            videos=videos, **output_kwargs["videos_kwargs"]
        )
        video_grid_thw = videos_inputs["video_grid_thw"]

        # If user has not requested video metadata, pop it
        if "return_metadata" not in kwargs:
            video_metadata = videos_inputs.pop("video_metadata")
        else:
            video_metadata = videos_inputs["video_metadata"]
        video_grid_thw = videos_inputs["video_grid_thw"]
    else:
        videos_inputs = {}
        video_grid_thw = None

    if not isinstance(text, list):
        text = [text]

    text = text.copy()  # below lines change text in-place
    if image_grid_thw is not None:
        merge_length = self.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                num_image_tokens = image_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace(
                    self.image_token, "<|placeholder|>" * num_image_tokens, 1
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

    if video_grid_thw is not None:
        merge_length = self.video_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while self.video_token in text[i]:
                metadata = video_metadata[index]
                if metadata.fps is None:
                    metadata.fps = 24 if metadata.fps is None else metadata.fps

                # if timestamps are not provided, calculate them
                curr_timestamp = self._calculate_timestamps(
                    metadata.frames_indices,
                    metadata.fps,
                    self.video_processor.merge_size,
                )

                video_placeholder = ""
                frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                for frame_idx in range(video_grid_thw[index][0]):
                    curr_time = curr_timestamp[frame_idx]
                    video_placeholder += f"<{curr_time:.1f} seconds>"
                    video_placeholder += (
                        self.vision_start_token
                        + "<|placeholder|>" * frame_seqlen
                        + self.vision_end_token
                    )
                if (
                    f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                    in text[i]
                ):
                    text[i] = text[i].replace(
                        f"{self.vision_start_token}{self.video_token}{self.vision_end_token}",
                        video_placeholder,
                        1,
                    )
                else:
                    # vllm may input video token directly
                    text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                index += 1

            text[i] = text[i].replace("<|placeholder|>", self.video_token)

    return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
    return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
        "return_mm_token_type_ids", None
    )
    text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
    self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

    if return_mm_token_type_ids:
        array_ids = np.array(text_inputs["input_ids"])
        mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
        mm_token_type_ids[array_ids == self.image_token_id] = 1
        text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

    return BatchFeature(
        data={**text_inputs, **image_inputs, **videos_inputs},
        tensor_type=return_tensors,
    )


def _save_func(self, output_dir: Optional[str] = None, state_dict=None):
    # If we are executing this function, we are the process zero, so we don't check for that.
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving model checkpoint to {output_dir}")

    supported_classes = (PreTrainedModel, PeftModel)
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    if not isinstance(self.model, supported_classes):
        if state_dict is None:
            state_dict = self.model.state_dict()

        if isinstance(
            self.accelerator.unwrap_model(self.model, keep_torch_compile=False),
            supported_classes,
        ):
            self.accelerator.unwrap_model(
                self.model, keep_torch_compile=False
            ).save_pretrained(
                output_dir,
                state_dict=state_dict,
                max_shard_size="20GB",
                safe_serialization=self.args.save_safetensors,
            )
        else:
            logging.info(
                "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
            )
            if self.args.save_safetensors:
                safetensors.torch.save_file(
                    state_dict,
                    os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            max_shard_size="20GB",
            safe_serialization=self.args.save_safetensors,
        )

    if self.processing_class is not None:
        self.processing_class.save_pretrained(output_dir)
    elif (
        self.data_collator is not None
        and hasattr(self.data_collator, "tokenizer")
        and self.data_collator.tokenizer is not None
    ):
        logging.info(
            "Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`"
        )
        self.data_collator.tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model, lact_args=None):
    # freeze vision encoder
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # freeze/unfreeze merger (MLP projector)
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    # freeze/unfreeze LLM (including LaCT-wrapped attention layers)
    if model_args.tune_mm_llm:
        # Enable all language model parameters
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = True
        # Enable lm_head
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        # Freeze all language model parameters first
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False

        # Then selectively enable LaCT parameters if LaCT is enabled
        if lact_args is not None and lact_args.lact_enable:
            lact_param_keywords = [
                "w0",
                "w1",
                "w2",
                "lr_proj",
                "q_scale",
                "q_offset",
                "k_scale",
                "k_offset",
                "ttt_scale_proj",
                "ttt_norm",
                "momentum_proj",
            ]
            for n, p in model.named_parameters():
                if any(kw in n for kw in lact_param_keywords):
                    p.requires_grad = True


class WindowDecayCallback(transformers.TrainerCallback):
    def __init__(self, max_ws, min_ws, bias_step, base_step=8000):
        self.max_ws = max_ws
        self.min_ws = min_ws
        self.bias_step = bias_step
        self.base_step = base_step

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        # assert state.global_step >= self.bias_step, (
        #     "global step should be greater than bias step"
        # )
        # linear decay scheduler
        ws = self.min_ws + (self.max_ws - self.min_ws) * (
            1 - max(min((state.global_step - self.bias_step) / self.base_step, 1), 0)
        )
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers
        elif hasattr(model, "language_model"):
            layers = model.language_model.layers
        else:
            raise ValueError("Cannot find language model layers in the model")

        for layer in layers:
            layer.self_attn.window_size = int(ws)


def qwen3vl_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    cache_position=None,
    **kwargs,
):
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )
        video_embeds = torch.cat(video_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(
                visual_pos_masks.sum(), img_embed.shape[-1]
            ).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask
            if not isinstance(attention_mask, dict)
            else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(
                attention_mask_tensor[:, 0], dim1=1, dim2=2
            )
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (
            prefill_compiled_stage or prefill_noncompiled_stage
        ) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    kwargs["video_grid_thw"] = video_grid_thw
    kwargs["video_mask"] = video_mask

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


def wrap_model_with_lact(model, lact_args):
    if not lact_args.lact_enable:
        rank0_print("SpatialTTT is disabled, using standard attention")
        return model

    # parse layer indices if specified
    if lact_args.lact_layers is not None:
        lact_layer_indices = set(
            int(x.strip()) for x in lact_args.lact_layers.split("/")
        )
    else:
        lact_layer_indices = None  # apply to all layers

    # access the decoder layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    elif hasattr(model, "language_model"):
        layers = model.language_model.layers
    else:
        raise ValueError("Cannot find language model layers in the model")

    num_layers = len(layers)
    rank0_print(f"Found {num_layers} decoder layers")

    # wrap each attention layer with SpatialTTT
    wrapped_count = 0
    for layer_idx, layer in enumerate(layers):
        if lact_layer_indices is not None and layer_idx not in lact_layer_indices:
            continue
        attn_layer = layer.self_attn
        lact_layer = Qwen3VLLaCTSWIGLULayer(
            attn_layer=attn_layer,
            num_lact_heads=lact_args.num_lact_heads,
            inter_multi=lact_args.inter_multi,
            window_size=lact_args.window_size,
            lact_chunk_size=lact_args.lact_chunk_size,
            qkv_silu=lact_args.qkv_silu,
            no_v_silu=lact_args.no_v_silu,
            use_muon=lact_args.use_muon,
            use_momentum=lact_args.use_momentum,
            learnable_ttt_scale=lact_args.learnable_ttt_scale,
            ttt_prenorm=lact_args.ttt_prenorm,
            ttt_nope=lact_args.ttt_nope,
            w0_w2_low_rank=lact_args.w0_w2_low_rank,
            use_fused_kernel=lact_args.use_fused_kernel,
            fp32_states=lact_args.fp32_states,
            use_conv_layer=lact_args.use_conv_layer,
        )
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            lact_layer.rotary_emb = model.model.language_model.rotary_emb

        model_dtype = next(attn_layer.parameters()).dtype
        model_device = next(attn_layer.parameters()).device
        lact_layer = lact_layer.to(dtype=model_dtype, device=model_device)

        layer.self_attn = lact_layer
        wrapped_count += 1
        rank0_print(f"Wrapped layer {layer_idx} with SpatialTTT")

    rank0_print(f"Wrapped {wrapped_count} layers with SpatialTTT")
    return model


def create_lact_optimizer(trainer, lact_args):
    opt_model = trainer.model

    if trainer.optimizer is None:
        decay_parameters = trainer.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        lact_param_names = []
        for name, _ in opt_model.named_parameters():
            if any(
                lact_param in name
                for lact_param in [
                    "w0",
                    "w1",
                    "w2",
                    "lr_proj",
                    "q_scale",
                    "q_offset",
                    "k_scale",
                    "k_offset",
                    "ttt_scale_proj",
                    "ttt_norm",
                    "momentum_proj",
                ]
            ):
                lact_param_names.append(name)

        lact_lr = (
            lact_args.lact_lr
            if lact_args.lact_lr is not None
            else trainer.args.learning_rate
        )

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n in decay_parameters
                    and n not in lact_param_names
                    and p.requires_grad
                ],
                "weight_decay": trainer.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n not in decay_parameters
                    and n not in lact_param_names
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n in decay_parameters
                    and n in lact_param_names
                    and p.requires_grad
                ],
                "weight_decay": trainer.args.weight_decay,
                "lr": lact_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n not in decay_parameters
                    and n in lact_param_names
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": lact_lr,
            },
        ]

        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters if len(group["params"]) > 0
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            trainer.args
        )
        trainer.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs
        )

    return trainer.optimizer


def print_lact_parameters(model):
    total_params = 0
    lact_params = 0
    trainable_params = 0
    lact_trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

        is_lact = any(
            lact_param in name
            for lact_param in [
                "w0",
                "w1",
                "w2",
                "lr_proj",
                "q_scale",
                "q_offset",
                "k_scale",
                "k_offset",
                "ttt_scale_proj",
                "ttt_norm",
                "momentum_proj",
            ]
        )
        if is_lact:
            lact_params += param.numel()
            if param.requires_grad:
                lact_trainable_params += param.numel()

    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    print(
        f"SpatialTTT parameters: {lact_params:,} ({100 * lact_params / total_params:.2f}%)"
    )
    print(f"SpatialTTT trainable parameters: {lact_trainable_params:,}")


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, SpatialTTTArguments)
    )
    model_args, data_args, training_args, lact_args = (
        parser.parse_args_into_dataclasses()
    )
    if training_args.lr_scheduler_type == "cosine_with_min_lr":
        training_args.lr_scheduler_kwargs = {"min_lr_rate": training_args.min_lr_rate}

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print(f"Loading model from {model_args.model_name_or_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    model.model.forward = MethodType(qwen3vl_forward, model.model)
    data_args.model_type = "qwen3vl"

    rank0_print(f"Model class: {model.__class__.__name__}")

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    type(processor).__call__ = processor_call

    def _preprocess_fn(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                resized_height, resized_width = (
                    data_args.resize_height,
                    data_args.resize_width,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_videos = stacked_videos.view(
                    B, T, C, resized_height, resized_width
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(
                stacked_videos[0], channel_dim=ChannelDimension.FIRST
            )

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(
            processed_videos_grouped, grouped_videos_index
        )
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    type(
        processor.video_processor
    )._preprocess = _preprocess_fn  # adjust resolution directly
    # type(processor.video_processor).fetch_videos = fetch_videos

    if lact_args.lact_enable:
        rank0_print("Wrapping model with SpatialTTT layers...")
        rank0_print(
            f"SpatialTTT config: num_heads={lact_args.num_lact_heads}, "
            f"chunk_size={lact_args.lact_chunk_size}, "
            f"window_size={lact_args.window_size}, "
            f"use_muon={lact_args.use_muon}, "
            f"w0_w2_low_rank={lact_args.w0_w2_low_rank}"
        )
        model = wrap_model_with_lact(model, lact_args)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()

    checkpoint_path = Path(model_args.model_name_or_path)
    safetensors_path = checkpoint_path / "model.safetensors"
    rank0_print(f"Loading checkpoint from {safetensors_path}...")
    state_dict = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        rank0_print(f"Missing keys: {missing}")
    if unexpected:
        rank0_print(f"Unexpected keys: {unexpected}")
    rank0_print(f"Loaded {len(state_dict)} parameters from checkpoint")

    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, TaskType, get_peft_model

        rank0_print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        if lact_args.lact_enable:
            for n, p in model.named_parameters():
                if any(
                    lact_param in n
                    for lact_param in [
                        "w0",
                        "w1",
                        "w2",
                        "lr_proj",
                        "q_scale",
                        "q_offset",
                        "k_scale",
                        "k_offset",
                        "ttt_scale_proj",
                        "ttt_norm",
                        "momentum_proj",
                    ]
                ):
                    p.requires_grad = True
    else:
        set_model(model_args, model, lact_args)

    # Enable input require grads AFTER setting trainable parameters
    # This must be done after LoRA/parameter freezing to ensure gradient flow
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # Disable gradient checkpointing for frozen vision encoder to avoid warnings
        # "None of the inputs have requires_grad=True"
        if not model_args.tune_mm_vision:
            if training_args.lora_enable:
                base_model = (
                    model.base_model.model if hasattr(model, "base_model") else model
                )
            else:
                base_model = model
            if hasattr(base_model, "visual"):
                # Disable gradient checkpointing for vision encoder
                for module in base_model.visual.modules():
                    if hasattr(module, "gradient_checkpointing"):
                        module.gradient_checkpointing = False
                    if hasattr(module, "_gradient_checkpointing_func"):
                        module._gradient_checkpointing_func = None

    if torch.distributed.get_rank() == 0:
        print_lact_parameters(model)
        # Print vision module trainable parameters (only if method exists)
        if training_args.lora_enable:
            # PEFT model: unwrap to get base model
            base_model = (
                model.base_model.model if hasattr(model, "base_model") else model
            )
        else:
            # Non-PEFT model: use directly
            base_model = model
        if hasattr(base_model, "visual") and hasattr(
            base_model.visual, "print_trainable_parameters"
        ):
            base_model.visual.print_trainable_parameters()

    data_module = make_supervised_data_module(processor, data_args=data_args)

    callbacks = []
    if lact_args.window_decay:
        window_decay_callback = WindowDecayCallback(
            max_ws=5400, min_ws=lact_args.window_size, bias_step=50, base_step=3000
        )
        callbacks.append(window_decay_callback)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    type(trainer)._save = _save_func

    if lact_args.lact_enable and lact_args.lact_lr is not None:
        trainer.create_optimizer = lambda: create_lact_optimizer(trainer, lact_args)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("Checkpoint found, resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
