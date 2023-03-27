import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import stylegan2.dnnlib as dnnlib
import stylegan2.legacy as legacy
from os import path
from paths import path_to_project_LB_TD as path_to_project
from tqdm import tqdm
import datetime
# This code will reconstruct an image masked part with StyleGan2-ADA

number_image_to_modify = 0

save_video = False
l2_weight = 1
name_t = "" # Name to had into the result folder
name_loss = "perc" # You can use 'perc' or 'l2' or 'vgg' or '5050' if 'all' it will do all the loss with reg and without reg
reg = True # If true, the noise of the synthesis will be regularized
translate = False # If true, it will take the translated image, inside dataset/translated_images
ffhq = False # If true, the image will be taken from the ffhq_modi folder
perc_change = 0.4 # The percentage of change between the original image and the modified image
num_steps                  = 1000 
w_avg_samples              = 10000
initial_learning_rate      = 0.1
initial_noise_factor       = 0.05
lr_rampdown_length         = 0.25
lr_rampup_length           = 0.05
noise_ramp_length          = 0.75
regularize_noise_weight    = 10000
verbose                    = False
device: torch.device
seed = 0
network_pkl = r"https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

if ffhq:
    ffhq_n = "_FFHQ"
else:
    ffhq_n = ""

if translate :
    translate_n = "_translated"
else:
    translate_n = ""

outdir_global = path.join(path_to_project,'output',f"IMAGE{ffhq_n}_{number_image_to_modify}{translate_n}")



def projector_vggl2(name_loss,reg):
    print('------------------------------')
    print('name_loss',name_loss)
    print('reg',reg)


    if reg :
        to_add = "_reg"
    else:
        to_add = "_noreg"
    outdir = path.join(outdir_global, name_loss + to_add + f"_photo_{number_image_to_modify}_projected_{num_steps}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    if translate and not(ffhq):
        original_image_path = path.join(path_to_project,'dataset',"translated_images",f'photo_{number_image_to_modify}',f"original_image_translated_{number_image_to_modify}.png")
        masked_image_path = path.join(path_to_project,'dataset',"translated_images",f'photo_{number_image_to_modify}',f"masked_image_translated_{number_image_to_modify}.png")
        mask_path = path.join(path_to_project,'dataset',"translated_images",f'photo_{number_image_to_modify}',f"mask_translated_{number_image_to_modify}.npy")
    elif not(translate) and not(ffhq):
        original_image_path = path.join(path_to_project,'dataset',"photos",f"photo_{number_image_to_modify}.png")
        masked_image_path = path.join(path_to_project,'dataset',"photos",f"photo_{number_image_to_modify}_modified.png")
        mask_path = path.join(path_to_project,'dataset',"masks",f"mask_{number_image_to_modify}.npy")
    elif ffhq and not(translate):
        original_image_path = path.join(path_to_project,'dataset',"ffhq_modi",f'photo_{number_image_to_modify}',f"photo_{number_image_to_modify}.png")
        masked_image_path = path.join(path_to_project,'dataset',"ffhq_modi",f'photo_{number_image_to_modify}',f"photo_{number_image_to_modify}_modified.png")
        mask_path = path.join(path_to_project,'dataset',"ffhq_modi",f'photo_{number_image_to_modify}',f"mask_{number_image_to_modify}.npy")
    else:
        original_image_path = path.join(path_to_project,'dataset',"translated_images_FFHQ",f'photo_{number_image_to_modify}',f"original_image_translated_{number_image_to_modify}.png")
        masked_image_path = path.join(path_to_project,'dataset',"translated_images_FFHQ",f'photo_{number_image_to_modify}',f"masked_image_translated_{number_image_to_modify}.png")
        mask_path = path.join(path_to_project,'dataset',"translated_images_FFHQ",f'photo_{number_image_to_modify}',f"mask_translated_{number_image_to_modify}.npy")

    mask = np.load(mask_path)
    mask = mask.astype(np.uint8)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(masked_image_path).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images_bf = target.unsqueeze(0).to(device).to(torch.float32)
    target_images = target_images_bf / 255 * 2 - 1

    if target_images.shape[2] > 256:
        target_images_small = F.interpolate(target_images_bf, size=(256, 256), mode='area')
    target_features = vgg16(target_images_small, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # convert mask to tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device, requires_grad=False)
    mask_tensor = mask_tensor.permute(2, 0, 1)
    mask_tensor = mask_tensor.view(1, 3, 1024, 1024)

    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        perc = step / num_steps

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        # ws = (w_opt).repeat([1, G.mapping.num_ws, 1])
        synth_images_e = G.synthesis(ws, noise_mode='const')

        synth_images = synth_images_e * mask_tensor

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_small = (synth_images + 1) * (255/2)
        if synth_images_small.shape[2] > 256:
            synth_images_small = F.interpolate(synth_images_small, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_small, resize_images=False, return_lpips=True)
    
        dist_vgg = (target_features - synth_features).square().sum()


        dist_l2 = ((target_images - synth_images_e)* (mask_tensor)).square().mean()
        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss_vgg = dist_vgg 
        loss_l2 = dist_l2 
        
        
        # Total loss.

        if name_loss == 'perc' :
            loss = (1-perc)*loss_vgg  + perc*loss_l2 * l2_weight 

        if name_loss == 'l2' :
            loss = loss_l2 * l2_weight 

        if name_loss == 'vgg' :
            loss = loss_vgg

        if name_loss == '5050' :
            if step > num_steps * perc_change:
                loss = loss_l2 * l2_weight 
            else:
                loss = loss_vgg 
        
        if reg :
            loss = loss + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist l2 {dist_l2:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()


    projected_w_steps = w_out.repeat([1, G.mapping.num_ws, 1])

    # Save an image every 200 steps.
    # create a folder for the images
    os.makedirs(f'{outdir}/images_list', exist_ok=True)
    for i in range(0, num_steps, 200):
        projected_w = projected_w_steps[i]
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/images_list/proj{i:04d}.png')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())
    # save the modifed image + the part of the projected image that fill the mask
    synth_image_numpy = PIL.Image.open(f'{outdir}/proj.png').convert('RGB')
    synth_image_numpy = np.array(synth_image_numpy, dtype=np.uint8)
    target_image_numpy = PIL.Image.open(f'{outdir}/target.png').convert('RGB')
    target_image_numpy = np.array(target_image_numpy, dtype=np.uint8)

    #Just to test lets load a image, convert it to numpy and then save it as a png

    new_synth_image = synth_image_numpy * (1-mask) + target_image_numpy * mask
    test = synth_image_numpy * (1-mask)
    PIL.Image.fromarray(test, 'RGB').save(f'{outdir}/the_part_missing_synthetised.png')
    test2 = target_image_numpy * mask
    PIL.Image.fromarray(test2, 'RGB').save(f'{outdir}/original_image_with_missing_part.png')
    PIL.Image.fromarray(new_synth_image, 'RGB').save(f'{outdir}/proj_fusion.png')


    list_images = [PIL.Image.open(f'{outdir}/proj.png'), PIL.Image.open(f'{outdir}/target.png')]
    line_image = PIL.Image.new('RGB', (list_images[0].width * len(list_images), list_images[0].height))
    for i in range(len(list_images)):
        line_image.paste(list_images[i], (list_images[0].width * i, 0))
    line_image.save(f'{outdir}/plot_in_line.png')

        
    list_images = [PIL.Image.open(f'{outdir}/proj.png'), PIL.Image.open(f'{outdir}/target.png')]
    line_image = PIL.Image.new('RGB', (list_images[0].width * len(list_images), list_images[0].height))
    for i in range(len(list_images)):
        line_image.paste(list_images[i], (list_images[0].width * i, 0))
    line_image.save(f'{outdir}/plot_in_line.png')

    # Evaluate the vgg feature distance between the projected image and the original image
    # Load the original image
    original_pil = PIL.Image.open(original_image_path).convert('RGB')
    original_uint8 = np.array(original_pil, dtype=np.uint8)
    original_uint8 = np.transpose(original_uint8, [2, 0, 1])
    original_uint8 = torch.tensor(original_uint8[np.newaxis], device=device)
    original_uint8 = original_uint8.to(torch.float32) / 255 * 2 - 1
    original_uint8 = original_uint8.to(device)
    # Load the projected image
    projected_pil = PIL.Image.open(f'{outdir}/proj.png').convert('RGB')
    projected_uint8 = np.array(projected_pil, dtype=np.uint8)
    projected_uint8 = np.transpose(projected_uint8, [2, 0, 1])
    projected_uint8 = torch.tensor(projected_uint8[np.newaxis], device=device)
    projected_uint8 = projected_uint8.to(torch.float32) / 255 * 2 - 1
    projected_uint8 = projected_uint8.to(device)

    # Compute the vgg feature distance
    original_reshaped = (original_uint8  + 1) * (255/2)
    if original_uint8.shape[2] > 256:
        original_uint8 = F.interpolate(original_uint8, size=(256, 256), mode='area')
    original_features = vgg16(original_uint8, resize_images=False, return_lpips=True)
    projected_reshaped = (projected_uint8  + 1) * (255/2)
    if projected_uint8.shape[2] > 256:
        projected_uint8 = F.interpolate(projected_uint8, size=(256, 256), mode='area')
    projected_features = vgg16(projected_uint8, resize_images=False, return_lpips=True)
    vgg_distance = (target_features - synth_features).square().sum()

    # Compute the L2 distance between the projected image and the original image
    l2_distance = torch.nn.functional.mse_loss(original_uint8, projected_uint8)

    # Compute the PSNR between the projected image and the original image
    psnr = 10 * torch.log10(1 / torch.mean((original_uint8 - projected_uint8)**2))

    # Compute the date
    date = datetime.datetime.now()

    # Save the results and the parameters in a txt file
    text = f'name {name_t} \n \n vgg_distance = {vgg_distance} \n l2_distance = {l2_distance} \n psnr = {psnr}  \n n_step = {num_steps} \n date = {date}'

    # save the text in a txt file
    with open(path.join(outdir, 'results.txt'), 'w') as f:
        f.write(text)


if name_loss = 'all' :
    for name_loss in ['vgg','l2','perc','5050'] : 
        for reg in [False,True] :
            projector_vggl2(name_loss,reg)

else :
    projector_vggl2(name_loss,reg)
