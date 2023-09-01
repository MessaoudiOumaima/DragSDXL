# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
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
# *************************************************************************

import os
import torch
import cv2
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.io import read_image
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from pytorch_lightning import seed_everything
from torchvision.transforms import functional as Fn
from torchvision.transforms import InterpolationMode
#from diffusers.utils import functional


intermediate_features = []


# override unet forward
# The only difference from diffusers:
# return intermediate UNet features of all UpSample blocks


def override_forward(self):

    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        last_up_block_idx: int = None,
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        print('prepare attention_mask')
        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        
        print('center input if necessar')
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        print('time')
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        print('broadcast to batch dimension')
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        print('time project')
        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        print('time embedding')
        emb = self.time_embedding(t_emb, timestep_cond)
        
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)
                emb = self.time_embedding(t_emb, timestep_cond, cross_attention=cross_attention_kwargs) #change added

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        print('pre process')
        # 2. pre-process
        sample = self.conv_in(sample)
        

        print('down')
        # 3. down
        down_block_res_samples = (sample,)
        i=0
        j=0
        for downsample_block in self.down_blocks:
            i+=1
            print('down block number', i)
            #print(downsample_block)
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                print('in drag pipeline')
                print ('smaple  ', sample.size())
                print('embedding ' ,emb.size())
                print('hidden state  ',encoder_hidden_states.size())
                print('attenstion mask  ',attention_mask)
                print('cross attention  ' , cross_attention_kwargs)
                

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                print('skiped downsmapling block')

                sample, res_samples = downsample_block(hidden_states=sample, temb=emb) 
                print ('smaple  ', sample.size())
                print('res_samples ' ,len(res_samples))
                

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            print('entered own_block_additional_residuals')

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                j+=1
                print('residual block', j)
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        print('mid')
        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        print('up')
        
        # 5. up
        # only difference from diffusers:
        # save the intermediate features of unet upsample blocks
        all_intermediate_features = []
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            all_intermediate_features.append(sample)
            # return early to save computation time if needed
            if last_up_block_idx is not None and i == last_up_block_idx:
                return all_intermediate_features

        print('post process')
        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # only difference from diffusers, return intermediate results
        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward


class DragXLPipeline(StableDiffusionXLPipeline): #StableDiffusionPipeline

    # must call this function when initialize
    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sample of the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=50, #changed from 77 to 50
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.cuda())[0]
        #text_embeddings=text_embeddings[:,:640] #change added
        return text_embeddings

    # get all intermediate features and then do bilinear interpolation
    # return features in the layer_idx list
    def forward_unet_features(self, z, t, encoder_hidden_states,added_cond_kwargs, layer_idx=[1,0], interp_res=256): #changed 768 to 256
        
        def hook_fn(module, input, output):
             global intermediate_features
             intermediate_features.append(output) 

        self.unet.register_forward_hook(hook_fn)
        unet_output = self.unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states, 
            added_cond_kwargs=added_cond_kwargs,
            #return_intermediates=True #this is not available for unet2dCondition
        )[0]


        all_return_features = []
        #intermediate_features=torch.zeros(1,1, 4, 64, 64).to('cuda')
        for idx in layer_idx:
            feat = intermediate_features[idx][0] #size of feat = ([1, 4, 64, 64])
            feat = Fn.resize(feat, (interp_res, interp_res), interpolation=InterpolationMode.BILINEAR )
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)
        return unet_output, return_features

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        prompt_embeds=None, # whether text embedding is directly provided.
        batch_size=5,
        height=512,
        width=512,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if prompt_embeds is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size

            # text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=50, #changed 77 to 50
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
            #text_embeddings=text_embeddings[:,:640] #change added

        else:
            batch_size = prompt_embeds.shape[0]
            text_embeddings = prompt_embeds
        print("input text embeddings :", text_embeddings.shape) #(1,50,768)

        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
            latents = torch.randn(latents_shape, device=DEVICE)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=50, #changed 77 to 50
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            #text_embeddings=text_embeddings[:,:640] #change added

            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            #print('text_embeddings --->',text_embeddings)


        print("latents shape: ", latents.shape) #([1, 4, 64, 64])
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        original_size: Optional[Tuple[int, int]] = (256,256),
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = (256,256),
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        #timestep: Union[torch.Tensor, float, int]

        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        
        # text embeddings
        #text_input = self.tokenizer(
        #    prompt,
        #    padding="max_length",
        #    max_length=50, #changed 77 to 50
        #    return_tensors="pt"
        #)
        #text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        #text_emb=text_embeddings
        
        #print('input_ids',text_input.input_ids.size())
        #text_embeddings=text_embeddings[:,:640] #change added
        #print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        
        latents = self.image2latent(image)
        start_latents = latents

        

        # unconditional embedding for classifier free guidance
        '''if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=50, #changed 77 to 50
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to('cuda'))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0).to('cuda')
            #text_embeddings=text_embeddings[:,:640] #change added]'''


        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]

                    # Prepare added time ids & embeddings
        '''if pooled_prompt_embeds == None:
            pooled_prompt_embeds= text_embeddings
        if prompt_embeds == None:
            prompt_embeds = text_embeddings
            
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
        original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
         )
        #timesteps and the embedding dim
        timesteps=self.scheduler.timesteps
        embeding_dim=add_text_embeds.size()[2]
        #time_embeddings= self.get_time_ids(timesteps,embeding_dim)


        #add_time_embeds= self._get_add_time_ids(add_text_embeds,timesteps,embeding_dim, dtype=prompt_embeds.dtype) #(text_embeddings, time_embeddings, embedding_dim)
        #print('added time embeding created with size',add_time_embeds.size() )

        #prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to('cuda')
        #add_time_ids = add_time_ids.to('cuda').repeat(batch_size , 1)
        #add_time_ids=add_time_ids.unsqueeze(2)
        #time_added_temp=torch.zeros((1,2,6))
        #time_added_temp[:,1,:]=add_time_ids
        #print('add_time_temp',time_added_temp.size())'''
        
         
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            #added args for unet





            #added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids.to('cuda')}
            #print('add_text_embeds',add_text_embeds.size())
            #print('add_time_ids',add_time_ids.size())
    

            # predict the noise
            #print('###############')
            #print('model_inputs', model_inputs.size())
            #print('encoder_hidden_states',text_embeddings.size())
            #print('t',t.size())
            #print('###############')

            #text_embeddings=text_embeddings[:,:640] #change added
            #print('before pred')
            model_inputs=model_inputs.to('cuda')
            t=t.to('cuda')
            #text_embeddings=text_embeddings.to('cuda')
            #text_emb=text_emb.to('cuda')
            
            
            #from SDXL pipeline
            from transformers import AutoTokenizer, PretrainedConfig
            def tokenize_prompt(tokenizer, prompt):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                    )
                text_input_ids = text_inputs.input_ids
                return text_input_ids

            def encode_prompt(prompt, text_input_ids_list=None):
                prompt_embeds_list = []
                text_input_ids_list = []
                
                
                # text ids
                text_input_ids_0 = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=50, #changed 77 to 50
                    truncation=True,
                    return_tensors="pt")
                
                text_input_ids_1 = self.tokenizer_2(
                    prompt,
                    padding="max_length",
                    max_length=50, #changed 77 to 50
                    truncation=True,
                    return_tensors="pt")
                
                # txt embed
                prompt_embeds_0=self.text_encoder(text_input_ids_0.input_ids.to('cuda'),output_hidden_states=True)
                prompt_embeds_1=self.text_encoder_2(text_input_ids_1.input_ids.to('cuda'),output_hidden_states=True)

                # pooled embeding
                #pooled_prompt_embeds_0 = prompt_embeds_lis[0][0]
                pooled_prompt_embeds_1 = prompt_embeds_1[0]

                #hidden state
                prompt_embeds_0 = prompt_embeds_0.hidden_states[-2]
                prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]

                #bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds_0 = prompt_embeds_0.view(prompt_embeds_0.shape[0], prompt_embeds_0.shape[1], -1)
                prompt_embeds_1 = prompt_embeds_1.view(prompt_embeds_1.shape[0], prompt_embeds_1.shape[1], -1)

                prompt_embeds = torch.concat([prompt_embeds_0,prompt_embeds_1], dim=-1)
                pooled_prompt_embeds = pooled_prompt_embeds_1.view(prompt_embeds_1.shape[0], -1)
                return prompt_embeds, pooled_prompt_embeds









                '''for i, text_encoder in enumerate(text_encoders):
                    if tokenizers is not None:
                        tokenizer = tokenizers[i]
                        text_input_ids = tokenize_prompt(tokenizer, prompt)
                    else:
                        assert text_input_ids_list is not None
                        text_input_ids = text_input_ids_list[i]

                    prompt_embeds = text_encoder(
                        text_input_ids.to(text_encoder.device),
                        output_hidden_states=True,
                    )
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                    bs_embed, seq_len, _ = prompt_embeds.shape
                    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                    prompt_embeds_list.append(prompt_embeds)

                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
                pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
                return prompt_embeds, pooled_prompt_embeds'''


            def import_model_class_from_model_name_or_path(
                pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
                text_encoder_config = PretrainedConfig.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, revision=revision
                )
                model_class = text_encoder_config.architectures[0]

                if model_class == "CLIPTextModel":
                    from transformers import CLIPTextModel
                    return CLIPTextModel
                elif model_class == "CLIPTextModelWithProjection":
                    from transformers import CLIPTextModelWithProjection
                    return CLIPTextModelWithProjection
                else:
                    raise ValueError(f"{model_class} is not supported.")

            def compute_text_embeddings(prompt, text_encoders=None, tokenizers=None):
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt)
                    prompt_embeds = prompt_embeds.to('cuda')
                    pooled_prompt_embeds = pooled_prompt_embeds.to('cuda')
                return prompt_embeds, pooled_prompt_embeds
               
                #text encoders x 2 and tokenizers x 2
            #pretrained_model_path="stabilityai/stable-diffusion-xl-base-1.0",
    
            #text_encoder_cls_one = import_model_class_from_model_name_or_path(
            #   pretrained_model_path, revision=None)
            #text_encoder_one = self.text_encoder(text_input.input_ids.to(DEVICE))
            #text_encoder_cls_two = import_model_class_from_model_name_or_path(
            #    pretrained_model_path,revision=None, subfolder="text_encoder_2")
            #text_encoder_two= self.text_encoder_2(text_input.input_ids.to(DEVICE))
            '''text_encoder_one = text_encoder_cls_one.from_pretrained(
                pretrained_model_path, subfolder="text_encoder", revision=None)
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                pretrained_model_path, subfolder="text_encoder_2", revision=None)'''

            #tokenizer_one = AutoTokenizer.from_pretrained(
            #    pretrained_model_path, subfolder="tokenizer", revision=None, use_fast=False )
            '''tokenizer_one=self.tokenizer( 
                [""] * batch_size,
                padding="max_length",
                max_length=50, #changed 77 to 50
                return_tensors="pt"
                )

            #print('tokenizer_one',tokenizer_one)
            #tokenizer_two = AutoTokenizer.from_pretrained(
            #    pretrained_model_path, subfolder="tokenizer_2", revision=None, use_fast=False)
            tokenizer_two=self.tokenizer_2(
                 [""] * batch_size,
                padding="max_length",
                max_length=50, #changed 77 to 50
                return_tensors="pt"
            )  '''         
            
            #tokenizers = [tokenizer_one, tokenizer_two]
            #text_encoders = [text_encoder_one, text_encoder_two]

            instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
            prompt)

            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds
            
            add_time_ids =  self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype) 

            elems_to_repeat = model_inputs.shape[0]

            unet_added_conditions = {
                        "time_ids": add_time_ids.repeat(elems_to_repeat, 1).to('cuda'),
                        "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat, 1).to('cuda'),
                    }
            prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat, 1, 1).to('cuda')
            noise_pred = self.unet(
                        model_inputs,
                        t,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,)[0]

            '''noise_pred = self.unet(
                    model_inputs,
                    t,
                    encoder_hidden_states=text_embeddings,#prompt_embeds,
                    #cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,)[0]'''



            #noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings, added_cond_kwargs={"text_embeds": text_embeddings,"time_ids":t_emb}).to('cuda')
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents
