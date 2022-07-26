from time import sleep
import numpy as np
from PIL import Image
import cv2
def find_color_pixels(mask):
    #Using these function we will try to find out the uniques colors in a image
    print("Finding the mask unique pixel values")
    new_shape_mask=np.unique(mask.reshape(-1,mask.shape[2]),axis=0)
    print(new_shape_mask)
    return new_shape_mask
    
def find_max_color(im,cl):
    counter=[0,0,0,0,0,0]
    for h in range(im.shape[0]):
        for w in range(im.shape[1]):
            if im[h, w, 0] == cl[0][0] and im[h, w, 1] == cl[0][1] and im[h, w, 2] == cl[0][2]:
                counter[0]+=1
            elif im[h, w, 0] == cl[1][0] and im[h, w, 1] == cl[1][1] and im[h, w, 2] == cl[1][2]:
                counter[1]+=1
            elif im[h, w, 0] == cl[2][0] and im[h, w, 1] == cl[2][1] and im[h, w, 2] == cl[2][2]:
                counter[2]+=1
            elif im[h, w, 0] == cl[3][0] and im[h, w, 1] == cl[3][1] and im[h, w, 2] == cl[3][2]:
                counter[3]+=1
            elif im[h, w, 0] == cl[4][0] and im[h, w, 1] == cl[4][1] and im[h, w, 2] == cl[4][2]:
                counter[4]+=1
            elif im[h, w, 0] == cl[5][0] and im[h, w, 1] == cl[5][1] and im[h, w, 2] == cl[5][2]:
                counter[5]+=1
    max_index=counter.index(max(counter))
    return max_index
#Read the image
filename=r'D:\Personal Info\Python Projects\Person Segmentation Using Unet and DeepLabv3+\Satelite Imageprocessing\Groundtruth_color.png'
label=Image.open(filename)
gt=np.array(label)
print("Image size: ",gt.shape)
#Now finding unique pixel values
color_values=find_color_pixels(gt)
cl_values=[[  0,0 ,0],
 [  0,0,255],
 [  0 ,255, 255],
 [255,0,0],
 [255,255,0]]
""""
cl_values=[[ 0,0,0],
 [ 0,0,233],
 [ 0,233,0],
 [ 0,233,233],
 [233,0,0],
 [233,233,0]]
"""
print(cl_values)
blank_img=np.zeros_like(gt)
def fill_values(index,row,col,rext,cext):
    for i in range(row,row+rext):
        for j in range(col,col+cext):
            blank_img[i,j,0]=cl_values[index][0]
            blank_img[i,j,1]=cl_values[index][1]
            blank_img[i,j,2]=cl_values[index][2]
    #print(blank_img[row:row+rowext,col:col+cext,:])
    
for row in range(0,(gt.shape[0]//486)*486,486):
    for col in range(0,(gt.shape[1]//461)*461,461):
        temp=gt[row:row+486,col:col+461,:]
        index=find_max_color(temp,cl_values)
        rowext=temp.shape[0]
        colext=temp.shape[1]
        #print(index)
        fill_values(index,row,col,rowext,colext)
        print(row,col)
    temp=gt[row:row+486,col:,:]
    print(temp.shape)
    
    index=find_max_color(temp,cl_values)
    rowext=temp.shape[0]
    colext=temp.shape[1]
    fill_values(index,row,col,rowext,colext)
    
#Now for extra portion
for coloumext in range(0,(gt.shape[1]//461)*461,461):
    temp=gt[row:,coloumext:coloumext+461,:]
    print(temp.shape)
    
    index=find_max_color(temp,cl_values)
    rowext=temp.shape[0]
    colext=temp.shape[1]
    fill_values(index,row,coloumext,rowext,colext)
#Final extension 
temp=gt[row:,coloumext:,:]
print(temp.shape)    
index=find_max_color(temp,cl_values)
rowext=temp.shape[0]
colext=temp.shape[1]
fill_values(index,row,coloumext,rowext,colext)

from PIL import Image
im = Image.fromarray(blank_img)
im.save("buildingcategorization.png")

