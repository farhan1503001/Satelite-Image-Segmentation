import os
import cv2
import numpy as np
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler=MinMaxScaler()
#Now path clearation
patch_size=512
def create_patches(path,store_path,mask_path,mask_store_path):
    #image_dataset=[]
    image=cv2.imread(path,1)
    mask=cv2.imread(mask_path,1)
            ####image patching############
    size_x=(image.shape[1]//patch_size)*patch_size#Nearest size which is patchable
    size_y=(image.shape[0]//patch_size)*patch_size#Nearest size which is patchable
            ### mask patching#############
            #Now converting to a pil image
    image=Image.fromarray(image)
    mask=Image.fromarray(mask)
            #Now we just crop the image for creating patches for training
    image=image.crop((0,0,size_x,size_y))
    mask=mask.crop((0,0,size_x,size_y))
            #now converting image to a numpy array for easy understanding and visiblity
    image=np.array(image)
    mask=np.array(mask)
            #Now we will start patchifying the images
    print("Now patchifying image and masks")
    patch_images=patchify(image,patch_size=(patch_size,patch_size,3),step=patch_size)
    patch_masks=patchify(mask,patch_size=(patch_size,patch_size,3),step=patch_size)
            #Now setting all patched images in a dataset
    counter=0
    for i in range(patch_images.shape[0]):
        for j in range(patch_images.shape[1]):
                #getting single patch image and setting them in dataset array
            single_patch_img=patch_images[i,j,:,:]
            single_patch_mask=patch_masks[i,j,:,:]
            #single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
            #Dropping the extra unnecessary dimension
            single_patch_img=single_patch_img[0]
            single_patch_mask=single_patch_mask[0]
            #image_dataset.append(single_patch_img)
            writing=store_path+str(counter)+'.png'
            mwriting=mask_store_path+str(counter)+'.png'
            cv2.imwrite(writing,single_patch_img)
            cv2.imwrite(mwriting,single_patch_mask)
            counter+=1
        
if __name__=='__main__':
    #creating training images
    mp=r'D:\Personal Info\Python Projects\Person Segmentation Using Unet and DeepLabv3+\Satelite Imageprocessing\output.png'
    ip=r'D:\Personal Info\Python Projects\Person Segmentation Using Unet and DeepLabv3+\Satelite Imageprocessing\input.png'
    store=r'D:\Personal Info\Python Projects\Person Segmentation Using Unet and DeepLabv3+\Satelite Imageprocessing\Dataset\Image'
    mstore=r'D:\Personal Info\Python Projects\Person Segmentation Using Unet and DeepLabv3+\Satelite Imageprocessing\Dataset\Mask'
    create_patches(ip,store,mp,mstore)