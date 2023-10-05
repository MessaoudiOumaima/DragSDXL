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

import imageio
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_prompt(model,prompt, text_input_ids_list=None):
    prompt_embeds_list = []
    text_input_ids_list = []
                
                
                # text ids
    text_input_ids_0 = model.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=50, #changed 77 to 50
                    truncation=True,
                    return_tensors="pt")
                
    text_input_ids_1 = model.tokenizer_2(
                    prompt,
                    padding="max_length",
                    max_length=50, #changed 77 to 50
                    truncation=True,
                    return_tensors="pt")
                
                # txt embed
    prompt_embeds_0=model.text_encoder(text_input_ids_0.input_ids.to('cuda'),output_hidden_states=True)
    prompt_embeds_1=model.text_encoder_2(text_input_ids_1.input_ids.to('cuda'),output_hidden_states=True)

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


def compute_text_embeddings(model,prompt, text_encoders=None, tokenizers=None):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(model,prompt)
        prompt_embeds = prompt_embeds.to('cuda')
        pooled_prompt_embeds = pooled_prompt_embeds.to('cuda')
    return prompt_embeds, pooled_prompt_embeds
               
             

def point_tracking(F0, F1, handle_points, handle_points_init, args):
    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = int(pi[0])-args.r_p, int(pi[0])+args.r_p+1
            c1, c2 = int(pi[1])-args.r_p, int(pi[1])+args.r_p+1
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            # WARNING: no boundary protection right now
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[i][0] = pi[0] - args.r_p + row
            handle_points[i][1] = pi[1] - args.r_p + col
        return handle_points

def check_handle_reach_target(handle_points, target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 1.0).all()

# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat, y, x, r):
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    Ia = feat[:, :, y0-r:y0+r+1, x0-r:x0+r+1]
    Ib = feat[:, :, y1-r:y1+r+1, x0-r:x0+r+1]
    Ic = feat[:, :, y0-r:y0+r+1, x1-r:x1+r+1]
    Id = feat[:, :, y1-r:y1+r+1, x1-r:x1+r+1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def drag_diffusion_update(model, init_code, t, handle_points, target_points, mask,accelerate, args, original_size = (256,256),
        crops_coords_top_left = (0, 0),
        target_size = (256,256),):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"

    #text_emb = model.get_text_embeddings(args.prompt).detach()
    # the init output feature of unet
    with torch.no_grad():
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(model,args.prompt)
        prompt_embeds = instance_prompt_hidden_states
        unet_add_text_embeds = instance_pooled_prompt_embeds
        add_time_ids =  model._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype) 

        

        elems_to_repeat = 1

        unet_added_conditions = {
            "time_ids": add_time_ids.repeat(elems_to_repeat, 1).to('cuda'),
            "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat, 1).to('cuda'),
            }
        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat, 1, 1).to('cuda')
        
        unet_output, F0 = model.forward_unet_features(init_code, t, encoder_hidden_states=prompt_embeds_input, added_cond_kwargs=unet_added_conditions,
            layer_idx=args.unet_feature_idx, interp_res=args.sup_res)


        x_prev_0, _ = model.step(unet_output, t, init_code)



    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr) #changed Adam to SGD
    optimizer = accelerate.prepare_optimizer(optimizer)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    
    #allocated_memory = torch.cuda.memory_allocated()
    #print(f"GPU memory allocated: {allocated_memory / (1024**3):.2f} GB")
    #cached_memory = torch.cuda.memory_cached()
    #print(f"GPU memory cached: {cached_memory / (1024**3):.2f} GB")
    
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            #allocated_memory_before = torch.cuda.memory_allocated()
            #print(f"GPU memory allocated before training: {allocated_memory_before / (1024**3):.2f} GB")
            
            unet_output, F1 = model.forward_unet_features(init_code, t, encoder_hidden_states=prompt_embeds_input, added_cond_kwargs=unet_added_conditions,
                layer_idx=args.unet_feature_idx, interp_res=args.sup_res)
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
                print('new handle points', handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 1:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
                loss += ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += args.lam * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f'%(loss.item()))

        print('before backward loss ')
        #print('estimated memory',torch.cuda.memory_summary(device=None, abbreviated=False))

        #scaler.scale(loss).backward()
        accelerate.backward(loss)
        print('backward loss done')
        optimizer.step()
        #scaler.update()
        optimizer.zero_grad()

    return init_code, prompt_embeds_input
