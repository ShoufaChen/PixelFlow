from einops import rearrange
import math
from typing import List, Optional, Union
import time
import torch
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_2d_rotary_pos_embed


class PixelFlowPipeline:
    def __init__(
        self,
        scheduler,
        transformer,
        text_encoder=None,
        tokenizer=None,
        max_token_length=512,
    ):
        super().__init__()
        self.class_cond = text_encoder is None or tokenizer is None
        self.scheduler = scheduler
        self.transformer = transformer
        self.patch_size = transformer.patch_size
        self.head_dim = transformer.attention_head_dim
        self.num_stages = scheduler.num_stages

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    @torch.autocast("cuda", enabled=False)
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = "",
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        use_attention_mask: bool = False,
        max_length: int = 512,
    ):
        # Determine the batch size and normalize prompt input to a list
        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Process prompt embeddings if not provided
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            prompt_attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=prompt_attention_mask if use_attention_mask else None
            )[0]

        # Determine dtype from available encoder
        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        # Move prompt embeddings to desired dtype and device
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)

        # Handle classifier-free guidance for negative prompts
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            # Normalize negative prompt to list and validate length
            if isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list):
                if len(negative_prompt) != batch_size:
                    raise ValueError(f"The negative prompt list must have the same length as the prompt list, but got {len(negative_prompt)} and {batch_size}")
                uncond_tokens = negative_prompt
            else:
                raise ValueError(f"Negative prompt must be a string or a list of strings, but got {type(negative_prompt)}")

            # Tokenize and encode negative prompts
            uncond_inputs = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_input_ids = uncond_inputs.input_ids.to(device)
            negative_prompt_attention_mask = uncond_inputs.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(
                negative_input_ids,
                attention_mask=negative_prompt_attention_mask if use_attention_mask else None
            )[0]

        if do_classifier_free_guidance:
            # Duplicate negative prompt embeddings and attention mask for each generation
            seq_len_neg = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        # Concatenate negative and positive embeddings and their masks
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        return prompt_embeds, prompt_attention_mask

    def sample_block_noise(self, bs, ch, height, width, eps=1e-6):
        gamma = self.scheduler.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 - gamma) + torch.ones(4, 4) * gamma + eps * torch.eye(4))
        block_number = bs * ch * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c h w) (p q) -> b c (h p) (w q)',b=bs,c=ch,h=height//2,w=width//2,p=2,q=2)
        return noise

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height,
        width,
        num_inference_steps=30,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        device=None,
        shift=1.0,
        use_ode_dopri5=False,
    ):
        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * self.num_stages

        if use_ode_dopri5:
            assert self.class_cond, "ODE (dopri5) sampling is only supported for class-conditional models now"
            from pixelflow.solver_ode_wrapper import ODE
            sample_fn = ODE(t0=0, t1=1, sampler_type="dopri5", num_steps=num_inference_steps[0], atol=1e-06, rtol=0.001).sample
        else:
            # default Euler
            sample_fn = None

        self._guidance_scale = guidance_scale
        batch_size = len(prompt)
        if self.class_cond:
            prompt_embeds = torch.tensor(prompt, dtype=torch.int32).to(device)
            negative_prompt_embeds = 1000 * torch.ones_like(prompt_embeds)
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale > 1,
                "",
                prompt_embeds=None,
                negative_prompt_embeds=None,
                use_attention_mask=True,
                max_length=self.max_token_length,
            )

        init_factor = 2 ** (self.num_stages - 1)
        height, width =  height // init_factor, width // init_factor
        shape = (batch_size * num_images_per_prompt, 3, height, width)
        latents = randn_tensor(shape, device=device, dtype=torch.float32)

        for stage_idx in range(self.num_stages):
            stage_start = time.time()
            # Set the number of inference steps for the current stage
            self.scheduler.set_timesteps(num_inference_steps[stage_idx], stage_idx, device=device, shift=shift)
            Timesteps = self.scheduler.Timesteps

            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                original_start_t = self.scheduler.original_start_t[stage_idx]
                gamma = self.scheduler.gamma
                alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
                beta = alpha * (1 - original_start_t) / math.sqrt(- gamma)

                # bs, ch, height, width = latents.shape
                noise = self.sample_block_noise(*latents.shape)
                noise = noise.to(device=device, dtype=latents.dtype)
                latents = alpha * latents + beta * noise

            size_tensor = torch.tensor([latents.shape[-1] // self.patch_size], dtype=torch.int32, device=device)
            pos_embed = get_2d_rotary_pos_embed(
                embed_dim=self.head_dim,
                crops_coords=((0, 0), (latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size)),
                grid_size=(latents.shape[-1] // self.patch_size, latents.shape[-1] // self.patch_size),
            )
            rope_pos = torch.stack(pos_embed, -1)

            if sample_fn is not None:
                # dopri5
                model_kwargs = dict(class_labels=prompt_embeds, cfg_scale=self.guidance_scale(None, stage_idx), latent_size=size_tensor, pos_embed=rope_pos)
                if stage_idx == 0:
                    latents = torch.cat([latents] * 2)
                stage_T_start = self.scheduler.Timesteps_per_stage[stage_idx][0].item()
                stage_T_end = self.scheduler.Timesteps_per_stage[stage_idx][-1].item()
                latents = sample_fn(latents, self.transformer.c2i_forward_cfg_torchdiffq, stage_T_start, stage_T_end, **model_kwargs)[-1]
                if stage_idx == self.num_stages - 1:
                    latents = latents[:latents.shape[0] // 2]
            else:
                # euler
                for T in Timesteps:
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    timestep = T.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                    if self.class_cond:
                        noise_pred = self.transformer(latent_model_input, timestep=timestep, class_labels=prompt_embeds, latent_size=size_tensor, pos_embed=rope_pos)
                    else:
                        encoder_hidden_states = prompt_embeds
                        encoder_attention_mask = prompt_attention_mask

                        noise_pred = self.transformer(
                            latent_model_input,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            timestep=timestep,
                            latent_size=size_tensor,
                            pos_embed=rope_pos,
                        )

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale(T, stage_idx) * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(model_output=noise_pred, sample=latents)
            stage_end = time.time()

        samples = (latents / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        return samples

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    def guidance_scale(self, step=None, stage_idx=None):
        if not self.class_cond:
            return self._guidance_scale
        scale_dict = {0: 0, 1: 1/6, 2: 2/3, 3: 1}
        return (self._guidance_scale - 1) * scale_dict[stage_idx] + 1

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0
