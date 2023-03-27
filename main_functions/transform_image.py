"""Now that we have hyperplanes linked to caracteristics, we can estimate the direction of the vector linked to a photo
and then we can transform the photo to be a man or a woman for example"""

import json
import numpy as np
from os import path
import torch
import os
import numpy as np
import PIL.Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from paths import path_to_project_LB_TD
import warnings
import stylegan2.dnnlib as dnnlib
import stylegan2.legacy as legacy

# example of data.json for the image 0
# "0": {"gender": "women", "age": "40-50", "hair_lenght": "short human hair", "hat": "bare head", "smile": "Smile "}



what_we_modify = "gender"
number_image_to_modify = 0
pas = 5
z_or_w = "w"



warnings.filterwarnings("ignore", category=UserWarning)

path_json_hyperplan_z = path.join(path_to_project_LB_TD,"dataset","hyperplan_z.json")
path_json_hyperplan_w = path.join(path_to_project_LB_TD,"dataset","hyperplan_w.json")

path_json = path.join(path_to_project_LB_TD,"dataset","data.json")

folder_to_save = path.join(path_to_project_LB_TD,"output")

network = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

list_caracteristics = ["smile", "hat", "hair_lenght", "age", "gender"]



dict_model = {}

extractor = AutoFeatureExtractor.from_pretrained("Minapotheo/smileornot")
model = AutoModelForImageClassification.from_pretrained("Minapotheo/smileornot")
dict_model["smile"] = extractor,model

extractor = AutoFeatureExtractor.from_pretrained("Minapotheo/hatornot")
model = AutoModelForImageClassification.from_pretrained("Minapotheo/hatornot")
dict_model["hat"] = extractor,model

extractor = AutoFeatureExtractor.from_pretrained("ibombonato/vit-age-classifier")
model = AutoModelForImageClassification.from_pretrained("ibombonato/vit-age-classifier")
dict_model["age"] = extractor,model

extractor = AutoFeatureExtractor.from_pretrained("Leilab/gender_class")
model = AutoModelForImageClassification.from_pretrained("Leilab/gender_class")
dict_model['gender'] = extractor,model

extractor = AutoFeatureExtractor.from_pretrained("Leilab/hair_lenght")
model = AutoModelForImageClassification.from_pretrained("Leilab/hair_lenght")
dict_model['hair_lenght'] = extractor,model

dict_accuracy = {}
beginning_class = {}
for caracteristic in list_caracteristics:
    dict_accuracy[caracteristic] = []
    beginning_class[caracteristic] = ""


def get_accuracy(image,dict_accuracy=dict_accuracy,dict_model = dict_model,beginning_class = beginning_class):
    image = image.resize((224, 224))
    
    for caracteristic in list_caracteristics:
        extractor,model = dict_model[caracteristic]
        inputs = extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        percentage_how_sure = torch.nn.functional.softmax(logits, dim=1)[0]
        if caracteristic == 'age':
            if predicted_class_idx in [0,1,2,3]:
                if len(beginning_class[caracteristic]) == 0:
                    beginning_class[caracteristic] = "young"
                    accuracy_young = sum([percentage_how_sure[i].item() for i in [0,1,2,3]])
                    dict_accuracy[caracteristic].append(accuracy_young)
                else:
                    if beginning_class[caracteristic] == "young":
                        accuracy_young = sum([percentage_how_sure[i].item() for i in [0,1,2,3]])
                        dict_accuracy[caracteristic].append(accuracy_young)
                    else:
                        accuracy_young = sum([percentage_how_sure[i].item() for i in [0,1,2,3]])
                        dict_accuracy[caracteristic].append(1-accuracy_young)
            else :
                if len(beginning_class[caracteristic]) == 0:
                    beginning_class[caracteristic] = "old"
                    accuracy_old = sum([percentage_how_sure[i].item() for i in [4,5,6,7,8,9]])
                    dict_accuracy[caracteristic].append(accuracy_old)
                else:
                    if beginning_class[caracteristic] == "old":
                        accuracy_old = sum([percentage_how_sure[i].item() for i in [4,5,6,7,8,9]])
                        dict_accuracy[caracteristic].append(accuracy_old)
                    else:
                        accuracy_old = sum([percentage_how_sure[i].item() for i in [4,5,6,7,8,9]])
                        dict_accuracy[caracteristic].append(1-accuracy_old)
        else :
            if len(beginning_class[caracteristic]) != 0:
                if model.config.id2label[predicted_class_idx] == beginning_class[caracteristic]:
                    dict_accuracy[caracteristic].append(percentage_how_sure[predicted_class_idx].item())
                else:
                    dict_accuracy[caracteristic].append(1-percentage_how_sure[predicted_class_idx].item())
            else:
                beginning_class[caracteristic] = model.config.id2label[predicted_class_idx]
                dict_accuracy[caracteristic].append(percentage_how_sure[predicted_class_idx].item())

def distance_from_hyperplans(np_list_vectors,hyperplan_dict,list_caracteristics):
    print('Calculating distance from hyperplans...')
    dict_distance = {}
    for what_we_modify in list_caracteristics:
        if what_we_modify in hyperplan_dict.keys():
            dict_distance[what_we_modify] = []
            for vector_of_the_image in np_list_vectors :
                hyperplan = np.array(hyperplan_dict[what_we_modify]['hyperplan'])
                intercept = np.array(hyperplan_dict[what_we_modify]['intercept'])
                c = (np.dot(hyperplan,vector_of_the_image[0]) -intercept) / np.linalg.norm(hyperplan)**2
                y = vector_of_the_image[0] + c * hyperplan
                norm_diff = np.linalg.norm(y - vector_of_the_image[0])
                orientation = np.sign(np.dot(hyperplan,y-vector_of_the_image[0]))
                dict_distance[what_we_modify].append(orientation*norm_diff)
    return dict_distance

if z_or_w == "z":
    with open(path_json_hyperplan_z, 'r') as fp:
        hyperplan_dict = json.load(fp)
elif z_or_w == "w":
    with open(path_json_hyperplan_w, 'r') as fp:
        hyperplan_dict = json.load(fp)

vector_of_the_image = np.load(path.join(path_to_project_LB_TD,'dataset',"latent_vectors_"+z_or_w,f"latent_vector_{number_image_to_modify}.npy"))

if z_or_w == "z" and vector_of_the_image.shape[0] != 1:
    vector_of_the_image = vector_of_the_image.reshape(1,512)

linspace = np.linspace(0,1,10)

hyperplan = np.array(hyperplan_dict[what_we_modify]['hyperplan'])
intercept = np.array(hyperplan_dict[what_we_modify]['intercept'])

c = (np.dot(hyperplan,vector_of_the_image[0]) -intercept) / np.linalg.norm(hyperplan)**2

y = vector_of_the_image + c * hyperplan


''' y is the vector of the image in the hyperplan, we can now transform the image by adding a vector to y'''

distance_between_vectors = 1
norm_diff = np.linalg.norm(y - vector_of_the_image)

'''Check if y- vector_of_the_image is orthogonal to hyperplan'''

np_list_vectors = []

for i in linspace:
    vector = (vector_of_the_image - pas*(distance_between_vectors/norm_diff) * i * (y - vector_of_the_image))
   
    np_list_vectors.append(vector)



print("distance between vectors ",[np.linalg.norm(np_list_vectors[i] - np_list_vectors[i+1]) for i in range(len(np_list_vectors)-1)][0])

print('Loading networks from "%s"...' % network)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with dnnlib.util.open_url(network) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

os.makedirs(folder_to_save, exist_ok=True)
folder_inside = f"image_{z_or_w}_{ number_image_to_modify}_{what_we_modify}_pas_{pas}"
final_path = path.join(folder_to_save,folder_inside)
os.makedirs(final_path, exist_ok=True)

# Generate images.
label = torch.zeros([1, G.c_dim], device=device)


truncation_psi = 1
noise_mode = "const"

list_of_points_torch = [torch.from_numpy(point).to(device) for point in np_list_vectors]
for index,z in enumerate(tqdm(list_of_points_torch)):
    if z_or_w == "w":
        z = z.repeat(1,G.mapping.num_ws,1)
        if device == torch.device('cpu') :
            img = G.synthesis(z, force_fp32=True)
        else:
            img = G.synthesis(z)
    else:
        if device == torch.device('cpu') :
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
        else:
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    get_accuracy(PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB'))
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{final_path}/image_{index:04d}.png')

'''Plot the ten images in a row and save the plot in the same folder'''

list_images = [Image.open(f'{final_path}/image_{i:04d}.png') for i in range(0, 10 )]
line_image = Image.new('RGB', (list_images[0].width * len(list_images), list_images[0].height))
for i in range(len(list_images)):
    line_image.paste(list_images[i], (list_images[0].width * i, 0))
line_image.save(f'{final_path}/plot_in_line.png')

'''Creation of a gif with the ten images'''

images = []
for i in range(0, 10 ):
    images.append(Image.open(f'{final_path}/image_{i:04d}.png'))
images[0].save(f'{final_path}/plot_in_gif.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

'''Lets plot the latent traversial curves'''

fig = plt.figure(figsize=(20,2))
for caracteristic in list_caracteristics :
    plt.plot(linspace, dict_accuracy[caracteristic], label = caracteristic+" : "+ beginning_class[caracteristic])
plt.title("Latent traversial curves, we modify : "+what_we_modify)
plt.legend()
plt.savefig(f'{final_path}/plot_latent_traversial_{z_or_w}.png')

'''Lets plot the distance between the vectors and the hyperplans'''
dict_distance = distance_from_hyperplans(np_list_vectors,hyperplan_dict,list_caracteristics)
fig = plt.figure(figsize=(20,2))
'''plot first a horizontal lign at y=0'''

plt.plot(linspace,np.zeros(len(linspace)), linestyle='dashed' )

for caracteristic in list_caracteristics :
    if caracteristic in hyperplan_dict.keys() :
        plt.plot(linspace, dict_distance[caracteristic], label = caracteristic+" : "+ beginning_class[caracteristic])
plt.title("Distance between the vectors and the hyperplans, we modify : "+what_we_modify)
plt.legend()
plt.savefig(f'{final_path}/plot_distance_hyperplans_{z_or_w}.png')


