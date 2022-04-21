# Improting Image class from PIL module 
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 3500000000000000000  
def image_splitter(image,data_type,out_dir):
    # Opens a image in RGB mode 
    img = Image.open(image)

    # Size of the image in pixels (size of orginal image) 
    # (This is not mandatory) 
    width, height = img.size 
    print(width,height)
    # Setting the points for cropped image 
    left1 = (30/100)*width
    top1 = 0
    right1 = width
    bottom1 = height 

    # Cropped image of above dimension 
    # (It will not change orginal image) 
    img1 = img.crop((left1, top1, right1, bottom1)) 
    #img1.save('ptrain.png')
    img1.save('mtrain.png')

    left2 = 0
    top2 = 0
    right2 = (30/100)*width
    bottom2 = height

    # Cropped image of above dimension 
    # (It will not change orginal image) 
    img2 = img.crop((left2, top2, right2, bottom2)) 
    #img2.save('ptest.png')
    img2.save('mtest.png')

img = 'Model'
data_type = 'input'
img_dir =r'D:\Personal Info\Python Projects\Person Segmentation Using Unet and DeepLabv3+\Satelite Imageprocessing\mask_final.png'
image_splitter(img_dir,data_type,img)