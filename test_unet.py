from tf_unet import unet, util, image_util
import glob
train_dirs=glob.glob( "/Users/davidknowles/Downloads/segmentation_training/*/" )
train_dirs=[ td + "training-set/image*" for td in train_dirs ]
data_provider = image_util.ImageDataProvider( train_dirs, 
    data_suffix=".png", mask_suffix="_mask.png") 
    
net = unet.Unet(layers=3, channels=3, n_class=2) # features_root=64, 
trainer = unet.Trainer(net)
path = trainer.train(data_provider, "/Users/davidknowles/Dropbox/miccai_nuclei/results/", training_iters=32, epochs=100)
# /Users/davidknowles/Dropbox/miccai_nuclei/tf_unet

import numpy as np
from PIL import Image
a=np.array(Image.open("/Users/davidknowles/Downloads/segmentation_training/hnsc/training-set/image01_mask.png"))