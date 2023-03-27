# InterfaceGan using StyleGan 2
 
! THE INPAINTING PART IS BELOW (it is the next part)!
## Setup 

Create a new conda environment and activate it:
```bash
conda create --name iin_project python==3.7.15
conda activate lcc
```
Install torch 1.12.1 with or without cuda depending on your hardware (for example here I have GPU with cuda 11.6 installed, refer to [this page](https://pytorch.org/) for installation)
``` 
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
Install `requirements.txt`
```
pip install -r requirements.txt
```
Be sure to put the path to the folder inside `path.py`, you can in vscode copy the path directly by left-clicking on the folder `project_LB_TD` and click on 'copy path'. For example for me it is :
```
path_to_project_LB_TD = r"C:\Users\denor\Desktop\introduction_a_l'imagerie numérique_projet\project_LB_TD"
```
We are using [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) for our code, their functions are in our folder `stylegan2`, that can be obtained by unzipping `stylegan2.zip`.

The main functions are all in the folder `main_functions`, make sure to execute the functions in the logic order :

First, generate the dataset of images, with their associated vectors in z and w. You can control the size of the dataset by changing the variable ``size`` at the beginning of the file `generate_images.py` (default 20 to test). Then run :

```
python -m main_functions.generate_images
```

Then you need to create the `data.json` file inside the `dataset` folder, this is a dictonary that contains the class predicted for each caracteristic and for each photo of the dataset. Then run :

```
python -m main_functions.classification_of_the_images
```

You can have all the statistics about what was created in the notebook ``statistics.ipynb``. 


Then with those predictions, we can compute the hyperplans in the z space or in the w space, in `estimate_hyperplans.py` you can control the ceil of surety to choose (by default it is 0) by changing the variable `seuil`. You can choose if you compute the hyperplan on z or on w (by default w). Then run :
```
python -m main_functions.estimate_hyperplans
```
Finaly you can use the biggest function we have : `transform_image.py`. This function takes :
- An index of a generated image (change `number_image_to_modify`)
- A caracteristic to change (you can change "gender", "age", "hair_lenght", "hat": "bare head", "smile"
) (change `what_we_modify`)
- A intensity of the modification (change `pas`)

This function will :
- Create ten images that are generate from a vector z or w that is modified by a certain amount. The amount of modification is controlled by the variable `pas` (default 6). 
- Create a gif that shows the evolution of the image from the original to the final image.
- Create a plot in line of the ten images.
- Create the plot of the latent traversial metric.
- Create a plot of the distance of the vector with the hyperplans.

Run :

```
python -m main_functions.transform_image
```

Finaly you can test if our method is working by running the function `is_it_working.py`, this function will test the porportion of images that are correctly classified after the transformation. You can change the number of images to test by changing the variable `nb_samples` (default 20 here). Then run :

```
python -m main_functions.is_it_working
```

-----------------------

# Inpainting with StyleGan2

First you will need to have created a dataset of  generated images with the function `generate_images.py` (see above).

Then you will have to download the small_ffhq dataset from here https://www.kaggle.com/datasets/tommykamaz/faces-dataset-small. Put everything in the folder `dataset` and rename the folder to `small_ffhq` if it is not already the case.

Then you will have to choose a mask for an image, you can use the function `create_mask.py` to create a mask for an image of the dataset generated by StyleGan2 of use `create_mask_ffhq.py` to create a mask for an image of the small_ffhq dataset. Be careful, when you choose the box, you have to select the box from the bottom left corner to the top right corner. 

Don't be on a remote server when you run the function `create_mask.py` or `create_mask_ffhq.py` because it will open a window to select the box, and the window will not be displayed on the remote server.

``` 
python -m main_functions.create_mask
python -m main_functions.create_mask_ffhq
```

Then you can run the function `inpainting.py` to inpaint the image with the mask. You can see how to use it inside the file in comments, if you struggle to understand it, contact me at denorme.theo@gmail.com. I advise you to use name_loss = 'all' to have the results of all the methods and then perform the other scripts.



```
python -m main_functions.inpainting

```

Now to view a beautiful plot of all the reconstructed images with the different methods, you can run the function `fusion_images.py`. You can see how to use it inside the file in comments. The function can also be used to create a fusion between two images, you can see how to use it inside the file in comments.

```
python -m main_functions.fusion_images
```

You can also run `fusion_results.py` to create a txt file with the results of all the methods on an image (in a table). You can see how to use it inside the file in comments.

```
python -m main_functions.fusion_results
```

Now you can do a translation of an image with the function `translation.py`. You can see how to use it inside the file in comments. After you can go back to the file `inpainting.py` and put translation = True to inpaint the image with the translation.

```
python -m main_functions.translation
```





