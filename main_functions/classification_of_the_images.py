import torch
from os import path
import glob
import json
from PIL import Image
from os import path
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from tqdm import tqdm
from paths import path_to_project_LB_TD


list_caracteristics = ["smile","hat","age","gender","hair_lenght"]

name_json_file = "data.json"

folder = path.join(path_to_project_LB_TD, "dataset", "photos")

for what_we_classify in list_caracteristics:
    print(f"we are classifying {what_we_classify}")
    if what_we_classify == "smile":
        extractor = AutoFeatureExtractor.from_pretrained("Minapotheo/smileornot")
        model = AutoModelForImageClassification.from_pretrained("Minapotheo/smileornot")
    elif what_we_classify == "hat":
        extractor = AutoFeatureExtractor.from_pretrained("Minapotheo/hatornot")
        model = AutoModelForImageClassification.from_pretrained("Minapotheo/hatornot")
    elif what_we_classify == "age" :
        extractor = AutoFeatureExtractor.from_pretrained("ibombonato/vit-age-classifier")
        model = AutoModelForImageClassification.from_pretrained("ibombonato/vit-age-classifier")
    elif what_we_classify == "gender" :
        extractor = AutoFeatureExtractor.from_pretrained("Leilab/gender_class")
        model = AutoModelForImageClassification.from_pretrained("Leilab/gender_class")
    elif what_we_classify == "hair_lenght" :
        extractor = AutoFeatureExtractor.from_pretrained("Leilab/hair_lenght")
        model = AutoModelForImageClassification.from_pretrained("Leilab/hair_lenght")

    """create a list of the path of the images in the folder images"""

    list_path = []

    for filename in glob.glob(folder + '/*.png'):
        list_path.append(filename)

    """ Function that take the path of an image and return the number of the image in the folder images"""

    def get_number_image(path_image):
        return str(path_image.split("\\")[-1].split("_")[-1].split(".")[0])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

    # Create a dictionary to store the name of the image and the prediction of the model or open an existing one
    if path.exists(path.join(path_to_project_LB_TD,"dataset",name_json_file)):
        with open(path.join(path_to_project_LB_TD,"dataset",name_json_file), 'r') as fp:
            dict = json.load(fp)
    else:
        dict = {}
        for i in range(len(list_path)):
            dict[str(i)] = {}

    for i, path_image in enumerate(tqdm(list_path)):
        image = Image.open(path_image)
        image = image.resize((224, 224))
        inputs = extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        percentage_how_sure = torch.nn.functional.softmax(logits, dim=1)[0]
        # Store the prediction of the model in the dictionary
        dict[get_number_image(path_image)][what_we_classify] = model.config.id2label[predicted_class_idx]
        dict[get_number_image(path_image)][what_we_classify+"_percentage"] = percentage_how_sure[predicted_class_idx].item()

    # Create a json file with the dictionary
    with open(path.join(path_to_project_LB_TD,"dataset",'data.json'), 'w') as fp:
        json.dump(dict, fp)



