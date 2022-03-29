import preprocess

if __name__=='__main__':
    load_path='people_segmentation'
    (x_train,y_train),(x_test,y_test)=preprocess.load_data(load_path)
    
    print("Training dataset length",len(x_train))
    print("Testing dataset length",len(x_test))
    #print("Train image size; ")
    
    #Now creating directories for saving images after preprocessing
    preprocess.create_dir('newdata/train/image/')
    preprocess.create_dir('newdata/train/mask/')
    preprocess.create_dir('newdata/test/image/')
    preprocess.create_dir('newdata/test/mask/')
    
    #Now we will really write resized data
    preprocess.augment(x_train,y_train,'newdata/train/',True)
    preprocess.augment(x_test,y_test,'newdata/test/',False)