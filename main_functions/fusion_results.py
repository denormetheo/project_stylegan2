import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paths import path_to_project_LB_TD as path_to_project
from os import path
import os

# If you put a number in number_image_to_modify and you have ran projector with name_loss = 'all' before, this will create a a txt file with the results of all the methods in a table, it will be saved in the folder results_txt in the project folder
number_image_to_modify = 56
ffhq = True
translate = False

if ffhq:
    ffhq_n = "_FFHQ"
else:
    ffhq_n = ""

if translate:
    translate_n = "_translated"
else:
    translate_n = ""

def fusion_results(list_path_text_to_fusion, output_file):
    # Create a list to store the results
    results = []
    
    # Loop through each result file
    for path, method in list_path_text_to_fusion:
        # Read the contents of the result file
        with open(path, "r") as f:
            contents = f.read()
        
        # Extract the values from the contents
        vgg_distance = float(contents.split("vgg_distance = ")[1].split("\n")[0])
        l2_distance = float(contents.split("l2_distance = ")[1].split("\n")[0])
        psnr = float(contents.split("psnr = ")[1].split("\n")[0])
        # n_step = int(contents.split("n_step = ")[1].split("\n")[0])
        # date = contents.split("date = ")[1].strip()
        
        # Add the values to the results list
        results.append([method, vgg_distance, l2_distance, psnr])
    
    # Sort the results list by vgg_distance
    results.sort(key=lambda x: x[1])
    
    # Write the results to the output file
    with open(output_file, "w") as f:
        f.write("Method\tVGG Distance\tL2 Distance\tPSNR\tN Steps\tDate\n")
        for result in results:
            f.write("\t".join([str(x) for x in result]) + "\n")


path_folder_to_save = path.join(path_to_project,"results_txt")
# Check if the folder exists, if not create it

if not os.path.exists(path_folder_to_save):
    os.makedirs(path_folder_to_save)


name_s =f'result_report_{number_image_to_modify}{ffhq_n}{translate_n}'

list_path_txt_to_fusion =[]
for reg in ['reg_','noreg_']:
    for method in ['vgg','l2','5050','perc']:
        list_path_txt_to_fusion.append((path.join(path_to_project,'output',f'IMAGE{ffhq_n}_{number_image_to_modify}{translate_n}',f'{translate_n}{method}_{reg}photo_{number_image_to_modify}_projected_1000/results.txt'),f'{method}_{reg}'))

fusion_results(list_path_txt_to_fusion,path.join(path_folder_to_save,f'{name_s}.txt'))







