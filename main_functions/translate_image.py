## This file will do an image translation, and save the the result in a folder called translated_images or translated_images_FFHQ
## inside the dataset folder. 
import os
import sys
import argparse
import numpy as np
from paths import path_to_project_LB_TD as path_to_project
from os import path
from PIL import Image
from matplotlib import pyplot as plt

translation_pixel = 25 # The number of pixel to translate the image
number_image_to_modify = 65 # The number of the image to modify
ffhq = True # If true, the image will be taken from the ffhq_modi folder

if ffhq:
    ffhq_n = "_FFHQ"
else:
    ffhq_n = ""

folder_name = "translated_images" + ffhq_n

outdir_global = path.join(path_to_project,'dataset',folder_name)
if not os.path.exists(outdir_global):
    os.makedirs(outdir_global)

outdir_global = path.join(outdir_global, f'photo_{number_image_to_modify}')
if not os.path.exists(outdir_global):
    os.makedirs(outdir_global)


if ffhq:
    original_image_path = path.join(path_to_project,'dataset',"ffhq_modi",f"photo_{number_image_to_modify}",f"photo_{number_image_to_modify}.png")
    masked_image_path = path.join(path_to_project,'dataset',"ffhq_modi",f"photo_{number_image_to_modify}",f"photo_{number_image_to_modify}_modified.png")
    mask_path = path.join(path_to_project,'dataset',"ffhq_modi",f"photo_{number_image_to_modify}",f"mask_{number_image_to_modify}.npy")
else:
    # Load the image to translate
    original_image_path = path.join(path_to_project,'dataset',"photos",f"photo_{number_image_to_modify}.png")
    masked_image_path = path.join(path_to_project,'dataset',"photos",f"photo_{number_image_to_modify}_modified.png")
    mask_path = path.join(path_to_project,'dataset',"masks",f"mask_{number_image_to_modify}.npy")

#Translation of the mask
mask = np.load(mask_path)
mask = np.roll(mask, -translation_pixel, axis=1)
mask[:, -translation_pixel:, :] = mask[:, -translation_pixel-1:-translation_pixel, :]

# Translation of the image, with replication of the border
original_image_o = Image.open(original_image_path).convert('RGB')
original_image = np.array(original_image_o)
original_image = np.roll(original_image, -translation_pixel, axis=1)
original_image[:, -translation_pixel:, :] = original_image[:, -translation_pixel-1:-translation_pixel, :]

# Translation of the masked image, with replication of the border
masked_image_o = Image.open(masked_image_path).convert('RGB')
masked_image = np.array(masked_image_o)
masked_image = np.roll(masked_image, -translation_pixel, axis=1)
masked_image[:, -translation_pixel:, :] = masked_image[:, -translation_pixel-1:-translation_pixel, :]

#Save the new mask
np.save(path.join(outdir_global,f'mask_translated_{number_image_to_modify}'),mask)

#Save the new images
original_image = Image.fromarray(original_image)
masked_image = Image.fromarray(masked_image)
original_image.save(path.join(outdir_global,f'original_image_translated_{number_image_to_modify}.png'))
masked_image.save(path.join(outdir_global,f'masked_image_translated_{number_image_to_modify}.png'))

# Create a new image with the combined width and height of the 4 images
width, height = original_image_o.size

# Save the masked next to the translated image
new_masked_image = Image.new("RGB", (2*width, height))
new_masked_image.paste(masked_image_o, (0, 0))
new_masked_image.paste(masked_image, (width, 0))
new_masked_image.save(path.join(outdir_global,f'masked_{translation_pixel}_{number_image_to_modify}.png'))


# Save the original next to the translated image
new_original_image = Image.new("RGB", (2*width, height))
new_original_image.paste(original_image_o, (0, 0))
new_original_image.paste(original_image, (width, 0))
new_original_image.save(path.join(outdir_global,f'original_{translation_pixel}_{number_image_to_modify}.png'))

# Convert NumPy arrays back to PIL images
original_image_pil = Image.open(path.join(outdir_global,f'original_image_translated_{number_image_to_modify}.png'))
masked_image_pil = Image.open(path.join(outdir_global,f'masked_image_translated_{number_image_to_modify}.png'))


# Save the plot
plt.savefig(path.join(outdir_global,f'plot_translated_images_{number_image_to_modify}.png'))
