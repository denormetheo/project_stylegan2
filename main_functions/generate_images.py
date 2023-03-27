
import numpy as np
import PIL.Image
import torch
import warnings

import os
from os import path
from tqdm import tqdm
from paths import path_to_project_LB_TD
import sys

import stylegan2.legacy as legacy
import stylegan2.dnnlib as dnnlib

size = 200
seed = 135

warnings.filterwarnings("ignore", category=UserWarning)

network = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
outdir = path.join(path_to_project_LB_TD, "dataset")

print('Loading networks from "%s"...' % network)

device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device "%s"...' % device)
with dnnlib.util.open_url(network) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

os.makedirs(outdir, exist_ok=True)
os.makedirs(path.join(outdir,"photos"), exist_ok=True)
os.makedirs(path.join(outdir,"latent_vectors_w"), exist_ok=True)
os.makedirs(path.join(outdir,"latent_vectors_z"), exist_ok=True)

truncation_psi = 1
noise_mode = "const"

# Generate images.
label = torch.zeros([1, G.c_dim], device=device)

z_vectors = np.random.RandomState(seed).randn(size, G.z_dim)

w_samples = G.mapping(torch.from_numpy(z_vectors).to(device),label, truncation_psi=truncation_psi,  truncation_cutoff=None)  # [N, L, C]
w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)

list_z_of_points = [z_vectors[i] for i in range(len(z_vectors))]
list_w_of_points = [w_samples[i] for i in range(len(w_samples))]


list_w_of_points_torch = [torch.from_numpy(point).to(device) for point in list_w_of_points]
list_z_of_points_torch = [torch.from_numpy(point).to(device) for point in list_z_of_points]

for index,(z,w) in enumerate(tqdm(zip(list_z_of_points_torch,list_w_of_points_torch),total=len(list_z_of_points_torch))):
    '''save the latent vector in the latent_vector folder'''
    np.save(path.join(outdir,'latent_vectors_w',f'latent_vector_{index}.npy'),w.cpu().numpy())
    np.save(path.join(outdir,'latent_vectors_z',f'latent_vector_{index}.npy'),w.cpu().numpy())
    
    w = w.repeat(1,G.mapping.num_ws,1)
    if device == torch.device('cpu') :
        img = G.synthesis(w, force_fp32=True)
    else:
        img = G.synthesis(w)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    '''save the image in the photos folder'''
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(path.join(outdir,'photos',f'photo_{index}.png'))
