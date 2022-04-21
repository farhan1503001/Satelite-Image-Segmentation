from cv2 import cvtColor
import numpy as np
import cv2
mask_directory=r'D:\Personal Info\Python Projects\Person Segmentation Using Unet and DeepLabv3+\Satelite Imageprocessing\output.png'
img_directory=r'D:\Personal Info\Python Projects\Person Segmentation Using Unet and DeepLabv3+\Satelite Imageprocessing\image.png'


def find_color_pixels(mask):
    #Using these function we will try to find out the uniques colors in a image
    print("Finding the mask unique pixel values")
    new_shape_mask=np.unique(mask.reshape(-1,mask.shape[2]),axis=0)
    print(new_shape_mask)
def find_color_second_method(image):
    print("Color pixel values in another way")
    unique_pixels=set( tuple(v) for m2d in image for v in m2d ) 
    print("Unique pixel are: ")
    print(unique_pixels)
if __name__ == '__main__':
    #image=cv2.imread(img_directory,1)
    mask=cv2.imread(mask_directory,1)
    if mask is not None:
        print("Image reading completed successfully")
        find_color_pixels(mask)
        find_color_second_method(mask)
    else:
        print("Sorry!! Couldn't read actually")
