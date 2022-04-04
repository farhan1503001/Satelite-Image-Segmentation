import tensorflow as tf
import os 
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip,Crop,CenterCrop,OpticalDistortion,CoarseDropout,ChannelShuffle,Rotate
import cv2
#library ported

#Now creating a function for saving data after all pre-processing
def create_dir(path):
    if not os.path.exists(path):
        #If path doesnot exist meaning there is no file then create one
        os.makedirs(path)
        
#Now writing a function for just loading data from folders
def load_data(path,split=0.1):
    #Now reading images and labels
    X=sorted(glob(os.path.join(path,'images','*.jpg')))
    Y=sorted(glob(os.path.join(path,'masks','*.png')))
    #Now we will split the dataset
    split_size=int(len(X)*split) #Finding the total number that we wanna split
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=split_size,random_state=42)
    return (x_train,y_train),(x_test,y_test)

#Now we will preprocess and try to augment the images
#Each image will be of size 512*512
def augment(images,masks,savepath,augment=True):
    height=512
    width=512
    #Now loop over each and every image
    count=0
    for x,y in tqdm(zip(images,masks),total=len(images)):
        #It will find the name of the images
        #Now reading an image
        
        image=cv2.imread(x,cv2.IMREAD_COLOR)
        mask=cv2.imread(y,cv2.IMREAD_COLOR)
        #Now our augmentation part
        
        if augment==True:
            #Horizontal augmentation
            aug=HorizontalFlip(p=1.0)
            augmented=aug(image=image,mask=mask)
            x1=augmented['image']
            y1=augmented['mask']
            
            x2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            y2 = mask
            
            #Now channel shuffle
            shuff=ChannelShuffle(p=1.0)
            shuffled=shuff(image=image,mask=mask)
            x3=shuffled['image']
            y3=shuffled['mask']
            
            #Now coarse dropout
            coarse_dropout=CoarseDropout(max_height=32,max_width=32,max_holes=10,min_holes=3,p=1)
            coarsed=coarse_dropout(image=image,mask=mask)
            x4=coarsed['image']
            y4=coarsed['mask']
            
            #Now Rotate
            rotate=Rotate(limit=45,p=1)
            rotated=rotate(image=image,mask=mask)
            x5=rotated['image']
            y5=rotated['mask']
            
            X=[image,x1,x2,x3,x4,x5]
            Y=[mask,y1,y2,y3,y4,y5]
            
        else:
            X=[image]
            Y=[mask]
            
        #Now we will resize each image to 512,512 size
        index=0
        for img,mask in zip(X,Y):
            #First we will try center cropping if not possible then we will just reize it
            try:
                #Center cropping
                cropper=CenterCrop(height=height,width=width,p=1)
                cropped=cropper(image=img,mask=mask)
                img=cropped['image']
                mask=cropped['mask']
            except:
                img=cv2.resize(img,(width,height))
                mask=cv2.resize(mask,(width,height))
            #Now creating temporary image name:
            temp_image_name=f"{count}_{index}.png"
            temp_mask_name=f"{count}_{index}.png"
            
            #Now writing the new images path
            img_path=os.path.join(savepath,"image",temp_image_name)
            mask_path=os.path.join(savepath,"mask",temp_mask_name)
            
            #Now we will write
            cv2.imwrite(img_path,img)
            cv2.imwrite(mask_path,mask)
            
            index+=1
        count+=1
       