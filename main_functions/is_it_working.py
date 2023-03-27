"""Now that we have hyperplanes linked to caracteristics, we can estimate the direction of the vector linked to a photo
and then we can transform the photo to be a man or a woman for example"""

import json
import numpy as np
from os import path
import torch
import numpy as np
import PIL.Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from os import path
from tqdm import tqdm
import stylegan2.dnnlib as dnnlib
import stylegan2.legacy as legacy
from paths import path_to_project_LB_TD
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Example of what we have in the data.json file
# "0": {"gender": "women", "age": "40-50", "hair_lenght": "short human hair", "hat": "bare head", "smile": "Smile "}

path_json_hyperplan_z = path.join(path_to_project_LB_TD,"dataset","hyperplan_z.json")
path_json_hyperplan_w = path.join(path_to_project_LB_TD,"dataset","hyperplan_w.json")
path_json = path.join(path_to_project_LB_TD,"dataset","data.json")
folder_images =path.join(path_to_project_LB_TD,"dataset")
network = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

list_caracteristics = ["smile", "hat", "hair_lenght", "age", "gender"]

nb_samples = 20
pas = 6
z_or_w = "w"



with open(path_json, 'r') as fp:
    data = json.load(fp)

if z_or_w == "z":
    with open(path_json_hyperplan_z, 'r') as fp:
        hyperplan_dict = json.load(fp)
elif z_or_w == "w":
    with open(path_json_hyperplan_w, 'r') as fp:
        hyperplan_dict = json.load(fp)

if nb_samples == len(data):
    random_number_list = range(0,len(data))
else:
    random_number_list = np.random.randint(0,len(data), nb_samples)



print('Loading networks from "%s"...' % network)
device = torch.device('cuda')
with dnnlib.util.open_url(network) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore


 # Generate images.
label = torch.zeros([1, G.c_dim], device=device)
truncation_psi = 1
noise_mode = "const"


for what_we_modify in list_caracteristics :
    if what_we_modify in hyperplan_dict.keys():
        if what_we_modify == "smile":
            extractor = AutoFeatureExtractor.from_pretrained("Minapotheo/smileornot")
            model = AutoModelForImageClassification.from_pretrained("Minapotheo/smileornot")
        elif what_we_modify == "hat":
            extractor = AutoFeatureExtractor.from_pretrained("Minapotheo/hatornot")
            model = AutoModelForImageClassification.from_pretrained("Minapotheo/hatornot")
        elif what_we_modify == "age" :
            extractor = AutoFeatureExtractor.from_pretrained("ibombonato/vit-age-classifier")
            model = AutoModelForImageClassification.from_pretrained("ibombonato/vit-age-classifier")
        elif what_we_modify == "gender" :
            extractor = AutoFeatureExtractor.from_pretrained("Leilab/gender_class")
            model = AutoModelForImageClassification.from_pretrained("Leilab/gender_class")
        elif what_we_modify == "hair_lenght" :
            extractor = AutoFeatureExtractor.from_pretrained("Leilab/hair_lenght")
            model = AutoModelForImageClassification.from_pretrained("Leilab/hair_lenght")



        count_changed = 0
        bad = []
        for number_image_to_modify in tqdm(random_number_list):
            vector_of_the_image = np.load(path.join(path_to_project_LB_TD,'dataset',"latent_vectors_"+z_or_w,f"latent_vector_{number_image_to_modify}.npy"))
            image_base_label = data[str(number_image_to_modify)][what_we_modify]

            if what_we_modify == 'age':
                young = ['0-10', '10-20', '20-30','30-40']
                if image_base_label in young :
                    image_base_label = young
                else:
                    image_base_label = ['40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
                if data[str(0)][what_we_modify] in image_base_label :
                    coef_ref = 1
                else:
                    coef_ref = -1

            if data[str(0)][what_we_modify] == image_base_label :
                coef_ref = 1
            else:
                coef_ref = -1

            if z_or_w == "z" and vector_of_the_image.shape[0] != 1:
                vector_of_the_image = vector_of_the_image.reshape(1,512)

            hyperplan = np.array(hyperplan_dict[what_we_modify]['hyperplan'])
            intercept = np.array(hyperplan_dict[what_we_modify]['intercept'])

            c = (np.dot(hyperplan,vector_of_the_image[0]) -intercept) / np.linalg.norm(hyperplan)**2

            y = vector_of_the_image + c * hyperplan

            ''' y is the vector of the image in the hyperplan, we can now transform the image by adding a vector to y'''

            distance_between_vectors = 1
            norm_diff = np.linalg.norm(y - vector_of_the_image)

            '''Check if y- vector_of_the_image is orthogonal to hyperplan'''

            # vector = (vector_of_the_image - coef_ref*pas*(distance_between_vectors/norm_diff)  * (y - vector_of_the_image))
            vector = (vector_of_the_image - pas*(distance_between_vectors/norm_diff)  * (y - vector_of_the_image))
        
            torch_vector = torch.from_numpy(vector).to(device)
            if z_or_w == "w":
                z = torch_vector.repeat(1,G.mapping.num_ws,1)
                
                if device == torch.device('cuda'):
                    img = G.synthesis(z)
                else :
                    img = G.synthesis(z, force_fp32=True)
            else:
                z = torch_vector
                if device == torch.device('cuda'):
                    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                else :
                    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode,force_fp32=True)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
            inputs = extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            label = model.config.id2label[predicted_class_idx]
            if label != data[str(number_image_to_modify)][what_we_modify] :
                count_changed += 1
            else :
                bad.append(number_image_to_modify)
            

        print("What we modify : ", what_we_modify)
        print("z or w : ", z_or_w)
        print("Number of images changed : ", count_changed)
        print("Number of images : ", nb_samples)
        percentage = count_changed/nb_samples
        print("Percentage of images changed : ", percentage)
        print("Bad images indexes : ", np.sort(np.array(bad)))
    else :
        print(f"The caracteristic {what_we_modify} is not in the hyperplan dictionary")
