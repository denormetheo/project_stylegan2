import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paths import path_to_project_LB_TD as path_to_project
from os import path
import os

# If you put a number in number_image_to_modify and you have ran projector with name_loss = 'all' before, this will create a beautiful plot of the results of all the methods in two lines
number_image_to_modify = None
ffhq = True
translate = True


# If number_image_to_modify is None, you can put the path of the images you want to fusion in line and give a name to the fusion
list_path_image_to_fusion = ["path_to_image_1","path_to_image_2"]
name = "ffhq_56"

# The created image will be saved in the folder fusion_for_the_report in the project folder

if ffhq:
    ffhq_n = "_FFHQ"
else:
    ffhq_n = ""

if translate:
    translate_n = "_translated"
else:
    translate_n = ""


path_folder_to_save = path.join(path_to_project,"fusion_for_the_report")
# Check if the folder exists, if not create it

if not os.path.exists(path_folder_to_save):
    os.makedirs(path_folder_to_save)


def create_image(name, width=1024, height=50):
    # Create a white background image
    image = Image.new("RGB", (width, height), (255, 255, 255))
    
    # Create a draw object to draw on the image
    draw = ImageDraw.Draw(image)
    
    # Choose a font and specify its size
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    
    # Measure the size of the text
    text_width, text_height = font.getsize(name)
    
    # Calculate the position to center the text
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    
    # Draw the text on the image
    draw.text((x, y), name, (0, 0, 0), font=font)
    
    return image



if number_image_to_modify == None :
    # Load the images
    list_image = []
    for path_image in list_path_image_to_fusion:
        list_image.append(Image.open(path_image).convert('RGB'))

    # Create a new image with the combined width and height of the len(list_image) images
    width, height = list_image[0].size
    new_image = Image.new("RGB", (len(list_image)*width, height))

    # Save the images next to each other
    for i in range(len(list_image)):
        new_image.paste(list_image[i], (i*width, 0))

    # Save the new image
    new_image.save(path.join(path_folder_to_save,f'{name}.png'))


else :
    name_s =f'all_methods_in_line{number_image_to_modify}{ffhq_n}{translate_n}'
    common_ = f'photo_{number_image_to_modify}_projected_1000/proj.png'
    if ffhq and not(translate):
        list_path_image_to_fusion =[(path.join(path_to_project,'dataset','ffhq_modi',f'photo_{number_image_to_modify}',f'photo_{number_image_to_modify}.png'),'Goal'),(path.join(path_to_project,'dataset','ffhq_modi',f'photo_{number_image_to_modify}',f'photo_{number_image_to_modify}_modified.png'),'Input')]
    elif not(ffhq) and translate:
        list_path_image_to_fusion =[(path.join(path_to_project,'dataset','translated_images',f'photo_{number_image_to_modify}',f'original_image_translated_{number_image_to_modify}.png'),'Goal'),(path.join(path_to_project,'dataset','translated_images',f'photo_{number_image_to_modify}',f'masked_image_translated_{number_image_to_modify}.png'),'Input')]
    elif ffhq and translate:
        list_path_image_to_fusion =[(path.join(path_to_project,'dataset','translated_images_FFHQ',f'photo_{number_image_to_modify}',f'original_image_translated_{number_image_to_modify}.png'),'Goal'),(path.join(path_to_project,'dataset','translated_images_FFHQ',f'photo_{number_image_to_modify}',f'masked_image_translated_{number_image_to_modify}.png'),'Input')]
    else:
        list_path_image_to_fusion =[(path.join(path_to_project,'dataset','photos',f'photo_{number_image_to_modify}.png'),'Goal'),(path.join(path_to_project,'output',f'IMAGE_{number_image_to_modify}',f'5050_reg_photo_{number_image_to_modify}_projected_1000/target.png'),'Input')]
    for reg in ['reg_','noreg_']:
        for method in ['vgg','l2','5050','perc']:
            if not(translate):
                list_path_image_to_fusion.append((path.join(path_to_project,'output',f'IMAGE{ffhq_n}_{number_image_to_modify}{translate_n}',f'{method}_{reg}photo_{number_image_to_modify}_projected_1000/proj.png'),f'{method}_{reg}'))
            elif translate and ffhq:
                list_path_image_to_fusion.append((path.join(path_to_project,'output',f'IMAGE{ffhq_n}_{number_image_to_modify}{translate_n}',f'{translate_n}{method}_{reg}photo_{number_image_to_modify}_projected_1000/proj.png'),f'{method}_{reg}'))
    # Load the images
    list_image = []
    for path_image in list_path_image_to_fusion:
        image = Image.open(path_image[0]).convert('RGB')
        name = path_image[1]
        # create a new image of the string name of the method writted on a white background
        width, height = image.size
        text_image = create_image(name, width, 50)
        # Create a fusion of the image and the text_image
        new_image = Image.new("RGB", (width, height+50))
        new_image.paste(image, (0, 0))
        new_image.paste(text_image, (0, height))
        list_image.append(new_image)

    # Create a new image with the combined width and height of the len(list_image) images
    # It will be in 2 lines
    width, height = list_image[0].size
    new_image = Image.new("RGB", (5*width, 2*height))

    # Save the images next to each other
    for i in range(len(list_image)//2):
        new_image.paste(list_image[i], (i*width, 0))
    for i in range(len(list_image)//2,len(list_image)):
        new_image.paste(list_image[i], ((i-len(list_image)//2)*width, height))

    # Save the new image
    new_image.save(path.join(path_folder_to_save,f'{name_s}.png'))




