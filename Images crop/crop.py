import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time
import statistics

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib

from stone import stone

# Importing stone config
sys.path.append(os.path.join(ROOT_DIR, "samples/stone/"))

# Directory to save the logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
STONE_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_stone_0001.h5")
 
# Download stone trained weights from Releases if needed
if not os.path.exists(STONE_MODEL_PATH):
    utils.download_trained_weights(STONE_MODEL_PATH)

# When running the Demo on the images in which the model is trained, use the second directory. 
# When running the Demo on Random images, uncomment the first direcotry and comment the second directory.
    
# Directory to run the dataset images
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/stone_dataset/stone/val")

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images/Dataset_images")

class InferenceConfig(stone.StoneConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def create_model_and_train():
    # Create model object in inference mode.
    create_model_and_train.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    create_model_and_train.model.load_weights(STONE_MODEL_PATH, by_name=True)

def load_run_object_detection():
    
    time_list = []
    # Load a random image from the images folder
    #file_names = next(os.walk(specs.IMAGE_DIR))[2]
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    all_images = []
    for filename in os.listdir(IMAGE_DIR):
        image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))
        
        # Time started when the image is being loaded
        start = time.time()

        # Run detection
        results = create_model_and_train.model.detect([image], verbose=1)
        
        # Time ended when the detection and masking was completed
        end = time.time()
        
        print("The execution time for this image was:", end-start)
        
        time_list.append(end-start)
    
        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
    
        class_id = 1
        images_cropped = []
        class_fltr = r['class_ids'] == class_id
        boxes = r['rois'][class_fltr, :]
        for box in boxes:
            y1, x1, y2, x2 = box
            cropped = image[y1: y2, x1: x2]
            images_cropped.append(cropped)
         
        #columns = 10
        #rows = 10
        #fig=plt.figure(figsize=(8, 8))
    
        '''
        for i in range(len(images_cropped)):
            img = images_cropped[i]
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img)
   '''
        
        if image is not None:
            all_images.append(image)
            
    print(statistics.mean(time_list))
    
    return(all_images)

        
        
        
if __name__ == '__main__':
    
    #InferenceConfig()
    config = InferenceConfig()
    config.display()
    create_model_and_train()
    
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'stone']
    
    load_run_object_detection()
    
  