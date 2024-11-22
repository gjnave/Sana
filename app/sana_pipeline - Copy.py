import os
import torch
import torch.nn as nn
import yaml
from typing import Tuple
from diffusion import DPMS, FlowEuler
from diffusion.data.datasets.utils import *
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.utils.logger import get_root_logger

def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])

def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        guidance_type = "classifier-free"
    return guidance_type

class SanaPipeline(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = get_root_logger()
        self.progress_fn = lambda progress, desc: None

        self.image_size = self.get_config('model', 'image_size', default=1024)
        self.latent_size = self.image_size // self.get_config('vae', 'vae_downsample_rate', default=32)
        self.max_sequence_length = self.get_config('text_encoder', 'model_max_length', default=300)
        self.flow_shift = self.get_config('scheduler', 'flow_shift', default=3.0)
        self.weight_dtype = torch.bfloat16
        self.base_ratios = eval(f"ASPECT_RATIO_{self.image_size}_TEST")
        self.vis_sampler = self.get_config('scheduler', 'vis_sampler', default='flow_dpm-solver')
        self.logger.info(f"Sampler {self.vis_sampler}, flow_shift: {self.flow_shift}")
        self.guidance_type = self.get_guidance_type()
        self.logger.info(f"Inference with {self.weight_dtype}, PAG guidance layer: {self.get_config('model', 'pag_applied_layers', default=[])}")

        self.vae = self.build_vae()
        self.tokenizer, self.text_encoder = self.build_text_encoder()
        self.model = self.build_sana_model()
        self.compute_null_embedding()

    def get_config(self, *keys, default=None):
        config = self.config
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        return config

    def get_guidance_type(self):
        guidance_type = "classifier-free_PAG"
        pag_scale = self.get_config('train', 'pag_scale', default=2.0)
        attn_type = self.get_config('model', 'attn_type', default='linear')
        if not (pag_scale > 1.0 and attn_type == "linear"):
            guidance_type = "classifier-free"
        return guidance_type

    def build_vae(self):
        vae_config = self.get_config('vae')
        return get_vae(vae_config['vae_type'], vae_config['vae_pretrained'], self.device).to(self.weight_dtype)

    def build_text_encoder(self):
        text_encoder_config = self.get_config('text_encoder')
        return get_tokenizer_and_text_encoder(name=text_encoder_config['text_encoder_name'], device=self.device)

    def build_sana_model(self):
        model_config = self.get_config('model', 'model')
        
        pred_sigma = self.get_config('scheduler', 'pred_sigma', default=True)
        learn_sigma = self.get_config('scheduler', 'learn_sigma', default=True) and pred_sigma
        
        default_work_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(default_work_dir, exist_ok=True)
        
        config_with_work_dir = {
            'work_dir': default_work_dir,
            **self.config
        }
        
        model_kwargs = {
            "input_size": self.latent_size,
            "pe_interpolation": self.get_config('model', 'pe_interpolation', default=1.0),
            "config": config_with_work_dir,
            "model_max_length": self.max_sequence_length,
            "qk_norm": self.get_config('model', 'qk_norm', default=False),
            "micro_condition": self.get_config('model', 'micro_condition', default=False),
            "caption_channels": self.text_encoder.config.hidden_size,
            "y_norm": self.get_config('text_encoder', 'y_norm', default=True),
            "attn_type": self.get_config('model', 'attn_type', default='linear'),
            "ffn_type": self.get_config('model', 'ffn_type', default='glumbconv'),
            "mlp_ratio": self.get_config('model', 'mlp_ratio', default=2.5),
            "mlp_acts": self.get_config('model', 'mlp_acts', default=['silu', 'silu', None]),
            "in_channels": self.get_config('vae', 'vae_latent_dim', default=32),
            "y_norm_scale_factor": self.get_config('text_encoder', 'y_norm_scale_factor', default=0.01),
            "use_pe": self.get_config('model', 'use_pe', default=False),
            "pred_sigma": pred_sigma,
            "learn_sigma": learn_sigma,
            "use_fp32_attention": self.get_config('model', 'fp32_attention', default=False) and self.get_config('model', 'mixed_precision') != "bf16",
        }
        
        model = build_model(model_config, **model_kwargs)
        return model.to(self.device)

    def compute_null_embedding(self):
        with torch.no_grad():
            null_caption_token = self.tokenizer(
                "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.weight_dtype)

        self.logger.info(f"Loaded model from: {model_path}")
        self.logger.warning(f"Missing keys: {missing}")
        self.logger.warning(f"Unexpected keys: {unexpected}")

    def register_progress_bar(self, progress_fn=None):
        self.progress_fn = progress_fn if progress_fn is not None else self.progress_fn

    @torch.inference_mode()
    def forward(
        self,
        prompt=None,
        height=1024,
        width=1024,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=5,
        pag_guidance_scale=2.5,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(42),
        latents=None,
    ):
        self.ori_height, self.ori_width = height, width
        self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        self.latent_size_h, self.latent_size_w = (
            self.height // self.get_config('vae', 'vae_downsample_rate', default=32),
            self.width // self.get_config('vae', 'vae_downsample_rate', default=32),
        )
        self.guidance_type = guidance_type_select(
            self.guidance_type, 
            pag_guidance_scale, 
            self.get_config('model', 'attn_type', default='linear')
        )

        if negative_prompt != "":
            null_caption_token = self.tokenizer(
                negative_prompt,
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]

        if prompt is None:
            prompt = [""]
        prompts = prompt if isinstance(prompt, list) else [prompt]
        samples = []

        for prompt in prompts:
            prompts, hw, ar = (
                [],
                torch.tensor([[self.image_size, self.image_size]], dtype=torch.float, device=self.device).repeat(
                    num_images_per_prompt, 1
                ),
                torch.tensor([[1.0]], device=self.device).repeat(num_images_per_prompt, 1),
            )
            for _ in range(num_images_per_prompt):
                with torch.no_grad():
                    prompts.append(
                        prepare_prompt_ar(prompt, self.base_ratios, device=self.device, show=False)[0].strip()
                    )

                    if not self.get_config('text_encoder', 'chi_prompt'):
                        max_length_all = self.get_config('text_encoder', 'model_max_length')
                        prompts_all = prompts
                    else:
                        chi_prompt = "\n".join(self.get_config('text_encoder', 'chi_prompt'))
                        prompts_all = [chi_prompt + prompt for prompt in prompts]
                        num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
                        max_length_all = (
                            num_chi_prompt_tokens + self.get_config('text_encoder', 'model_max_length') - 2
                        )

                    caption_token = self.tokenizer(
                        prompts_all,
                        max_length=max_length_all,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(device=self.device)
                    select_index = [0] + list(range(-self.get_config('text_encoder', 'model_max_length') + 1, 0))
                    caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
                        :, :, select_index
                    ].to(self.weight_dtype)
                    emb_masks = caption_token.attention_mask[:, select_index]
                    null_y = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None].to(self.weight_dtype)

                    n = len(prompts)
                    if latents is None:
                        z = torch.randn(
                            n,
                            self.get_config('vae', 'vae_latent_dim', default=32),
                            self.latent_size_h,
                            self.latent_size_w,
                            generator=generator,
                            device=self.device,
                            dtype=self.weight_dtype,
                        )
                    else:
                        z = latents.to(self.weight_dtype).to(self.device)
                    model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
                    if self.vis_sampler == "flow_euler":
                        flow_solver = FlowEuler(
                            self.model,
                            condition=caption_embs,
                            uncondition=null_y,
                            cfg_scale=guidance_scale,
                            model_kwargs=model_kwargs,
                        )
                        sample = flow_solver.sample(
                            z,
                            steps=num_inference_steps,
                        )
                    elif self.vis_sampler == "flow_dpm-solver":
                        scheduler = DPMS(
                            self.model,
                            condition=caption_embs,
                            uncondition=null_y,
                            guidance_type=self.guidance_type,
                            cfg_scale=guidance_scale,
                            pag_scale=pag_guidance_scale,
                            pag_applied_layers=self.get_config('model', 'pag_applied_layers', default=[]),
                            model_type="flow",
                            model_kwargs=model_kwargs,
                            schedule="FLOW",
                        )
                        scheduler.register_progress_bar(self.progress_fn)
                        sample = scheduler.sample(
                            z,
                            steps=num_inference_steps,
                            order=2,
                            skip_type="time_uniform_flow",
                            method="multistep",
                            flow_shift=self.flow_shift,
                        )

            sample = sample.to(self.weight_dtype)
            with torch.no_grad():
                sample = vae_decode(self.get_config('vae', 'vae_type'), self.vae, sample)

            sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)
            samples.append(sample)

        return samples[0] if len(samples) == 1 else samples