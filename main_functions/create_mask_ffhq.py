import PySimpleGUI as sg
import numpy as np
from os import path
from paths import path_to_project
from PIL import Image
import os
import matplotlib.pyplot as plt


# This file will create a mask for the image, and save the mask in a folder named masks in the dataset folder
# This works ONLY with the images in the folder small_ffhq inside the dataset folder
# The mask, original image and masked image will be saved in the folder ffhq_modi, inside the dataset folder
number_image_to_modify = 65
original_image_path = path.join(path_to_project,'dataset',"small_ffhq",f"photo_{number_image_to_modify}.png")
original_image = Image.open(original_image_path)

original_image_size = original_image.size


path_folder_modified_images = path.join(path_to_project,'dataset',"modified_ffhq")
if not path.exists(path_folder_modified_images):
    os.mkdir(path_folder_modified_images)
folder_inside = path.join(path_folder_modified_images,f"photo_{number_image_to_modify}")
if not path.exists(folder_inside):
    os.mkdir(folder_inside)

"""save the image in size 512*512"""
resized_image = original_image.resize((512,512))
downsampling_factor = original_image_size[0]/512
resized_image.save(path.join(folder_inside,f"photo_{number_image_to_modify}_512.png"))
new_path = path.join(folder_inside,f"photo_{number_image_to_modify}_512.png")


"""
    Demo - Drag a rectangle to draw it
    This demo shows how to use a Graph Element to (optionally) display an image and then use the
    mouse to "drag a rectangle".  This is sometimes called a rubber band and is an operation you
    see in things like editors
"""
image_file = new_path

layout = [[sg.Graph(
    canvas_size=(512,512) ,
    graph_bottom_left=(0, 0),
    graph_top_right=(512,512),
    key="-GRAPH-",
    change_submits=True,  # mouse click events
    background_color='lightblue',
    drag_submits=True), ],
    [sg.Text(key='info', size=(60, 1))]]

window = sg.Window("draw rect on image", layout , size=(600,600), finalize=True)
# get the graph element for ease of use later
graph = window["-GRAPH-"]  # type: sg.Graph

graph.draw_image(image_file, location=(0,512)) if image_file else None
dragging = False
start_point = end_point = prior_rect = None
start_point_for_the_mask = end_point_for_the_mask = None

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break  # exit

    if event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
        x, y = values["-GRAPH-"]
        if not dragging:
            start_point = (x, y)
            start_point_for_the_mask = (int(x*downsampling_factor),int(y*downsampling_factor))
            dragging = True
        else:
            end_point = (x, y)
            end_point_for_the_mask = (int(x*downsampling_factor),int(y*downsampling_factor))
        if prior_rect:
            graph.delete_figure(prior_rect)
        if None not in (start_point, end_point):
            prior_rect = graph.draw_rectangle(start_point, end_point, line_color='red')

    elif event.endswith('+UP'):  # The drawing has ended because mouse up
        info = window["info"]
        info.update(value=f"grabbed rectangle from {start_point} to {end_point}")
        start_point, end_point = None, None  # enable grabbing a new rect
        dragging = False

    else:
        print("unhandled event", event, values)

"""Now we create the mask"""
##Check if the mask folder exists
if not path.exists(path.join(path_to_project,'dataset',"masks")):
    os.makedirs(path.join(path_to_project,'dataset',"masks"))

"""Check if start_point_for_the_mask and end_point_for_the_mask are well placed"""
corrected_start,corrected_end = start_point_for_the_mask,end_point_for_the_mask
if start_point_for_the_mask[0] > end_point_for_the_mask[0] and start_point_for_the_mask[1] > end_point_for_the_mask[1]:
    corrected_start,corrected_end = end_point_for_the_mask,start_point_for_the_mask
elif start_point_for_the_mask[0] > end_point_for_the_mask[0] and start_point_for_the_mask[1] < end_point_for_the_mask[1]:
    corrected_start,corrected_end = (end_point_for_the_mask[0],start_point_for_the_mask[1]),(start_point_for_the_mask[0],end_point_for_the_mask[1])
elif start_point_for_the_mask[0] < end_point_for_the_mask[0] and start_point_for_the_mask[1] > end_point_for_the_mask[1]:
    corrected_start,corrected_end = (start_point_for_the_mask[0],end_point_for_the_mask[1]),(end_point_for_the_mask[0],start_point_for_the_mask[1])


mask = np.ones((*original_image_size,3))
mask[start_point_for_the_mask[1]:end_point_for_the_mask[1],( start_point_for_the_mask[0]):(end_point_for_the_mask[0]),:] = np.zeros((np.abs(end_point_for_the_mask[1]-start_point_for_the_mask[1]),np.abs(end_point_for_the_mask[0]-start_point_for_the_mask[0]),3))
mask = np.flip(mask,axis=0)


np.save(path.join(folder_inside,f"mask_{number_image_to_modify}.npy"),mask)

"""Now we can use the mask to modify the original image"""
original_image = np.array(original_image)
modified_image = original_image*(mask)
modified_image = Image.fromarray(modified_image.astype('uint8'))
modified_image.save(path.join(folder_inside,f"photo_{number_image_to_modify}_modified.png"))

#Suppression of the 512*512 image
os.remove(new_path)

"""Plot the mask, the original image and the modified image, without the axis"""

plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.imshow(mask)
plt.title("Mask")
plt.subplot(1,3,2)
plt.imshow(original_image)
plt.title("Original image")
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(modified_image)
plt.title("Modified image")
plt.axis('off')
plt.show()


