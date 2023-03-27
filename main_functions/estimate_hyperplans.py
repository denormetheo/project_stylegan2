'''create a svm on the vectors of the images of the folder 1000_images and find the hyperplane that separate the images of the class we want to classify from the others.'''

import numpy as np
import json
from os import path
from tqdm import tqdm
from paths import path_to_project_LB_TD
from sklearn import svm
from sklearn.metrics import accuracy_score


#Example of the data.json for the image 0 :
#  "0": {"gender": "women", "age": "40-50", "hair_lenght": "short human hair", "hat": "bare head", "smile": "Smile "}

list_caracteristics = ['gender', 'age', 'hair_lenght', 'hat', 'smile']

z_or_w = "w"

seuil = 0

name_json_file = f"hyperplan_{z_or_w}.json"


path_json = path.join(path_to_project_LB_TD,"dataset","data.json")


if path.exists(path.join(path_to_project_LB_TD,"dataset",name_json_file)):
    with open(path.join(path_to_project_LB_TD,"dataset",name_json_file), 'r') as fp:
        dict_hyperplan = json.load(fp)
else:
    dict_hyperplan = {}
    

with open(path_json, 'r') as fp:
    data = json.load(fp)


for what_we_classify in list_caracteristics:
    print(f"For {what_we_classify} :")
    # Create a list of the vectors of the images of the folder images
    list_vectors = []
    list_labels = []
    """We need to distinguish the class age, because it is not a binary classification"""
    if what_we_classify == "age":
        reference = data[str(0)][what_we_classify]
        if reference in ['0-10','10-20','20-30', '30-40']:
            reference = ['0-10','10-20','20-30', '30-40']
        elif reference in ['40-50','50-60','60-70', '70-80', '80-90','90-100']:
            reference = ['40-50','50-60','60-70', '70-80', '80-90','90-100']
        for i in tqdm(range(len(data))):
            vector_path = path.join(path_to_project_LB_TD,'dataset',f"latent_vectors_{z_or_w}",f"latent_vector_{i}.npy")
            vector = np.load(vector_path)
            if z_or_w == "z":
                list_vectors.append(vector)
            else:
                list_vectors.append(vector[0])
            list_labels.append(int(data[str(i)][what_we_classify] in reference))
    else:
        for i in tqdm(range(len(data))):
            if data[str(i)][what_we_classify + "_percentage"] > seuil:
                vector_path = path.join(path_to_project_LB_TD,'dataset',f"latent_vectors_{z_or_w}",f"latent_vector_{i}.npy")
                vector = np.load(vector_path)
                if z_or_w == "z":
                    list_vectors.append(vector)
                else:
                    list_vectors.append(vector[0])
                list_labels.append(int(data[str(i)][what_we_classify] == data[str(0)][what_we_classify]))

    print("number total of images that have been selected : ", len(list_vectors))
    if what_we_classify == "age":
        print(f"number of images of classes {reference} that have been selected : ", sum([int(x) for x in list_labels]))
    else:
        print(f"number of images of class {data[str(0)][what_we_classify]} that have been selected : ", sum([int(x) for x in list_labels]))


    # fit the model
    if len(set(list_labels)) == 1:
        print("The model can't be trained because there is only one class")
        continue
    else :
        clf = svm.LinearSVC()
        clf.fit(np.array(list_vectors), np.array(list_labels))

        # get the hyperplan
        hyperplan = clf.coef_[0]

        # get the intercept
        intercept = clf.intercept_[0]

        # Save it in a json file
        dict_hyperplan[what_we_classify] = {}
        dict_hyperplan[what_we_classify]["hyperplan"] = hyperplan.tolist()
        dict_hyperplan[what_we_classify]["intercept"] = intercept.tolist()

        with open(path.join(path_to_project_LB_TD,'dataset',name_json_file), 'w') as fp:
            json.dump(dict_hyperplan, fp)

        # Test the model

        y_pred = clf.predict(np.array(list_vectors))
        print("accuracy",accuracy_score(np.array(list_labels), y_pred))




