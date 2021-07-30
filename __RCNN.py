#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:45:46 2020

@author: tes520
"""

from xml.etree import ElementTree as ET
from mrcnn.utils import Dataset, compute_ap
from mrcnn.config import Config
from mrcnn.model import load_image_gt, mold_image
from matplotlib.patches import Rectangle
from glob import glob

import os
import random

import numpy as np
import matplotlib.pyplot as plt

# class that defines and loads the kangaroo dataset
class MakeDataset(Dataset):
	# load the dataset definitions
    
    def load_dataset(self, dataset_dir, is_train=True, train_size = 0.2, model_name = 'tree'):
        #add classes to be identified by the model
        self.add_class('dataset',1,'Tree')
        self.add_class('dataset',2,'Shrub')
        
        #add images to the class
        images_dir = dataset_dir+'/images'
        annotations_dir = dataset_dir+'/annots'
        
        #pick out a random selection for testing
#        img_files = os.listdir(images_dir)
        img_files = [x.split('/')[-1] for x in glob(images_dir+'/*.jpg')]

        inds = np.arange(len(img_files))
#        test_inds = random.sample(list(inds), int(len(inds)*train_size))
        test_inds = random.sample(list(inds), 1)
        
        test_imagefiles = [img_files[x] for x in test_inds]
        
                
        #find all the images
        for filename in img_files:
            print(filename)
            #get image info
            image_id = filename.split('/')[-1].split('.')[0]
            img_path = images_dir+'/'+filename
            ann_path = annotations_dir+'/'+filename.split('.')[0]+'.xml'
            
            #ignore those images destined for testing
            if is_train:
                if filename in test_imagefiles:
                    pass
                else:
                    #add this image to the dataset
                    self.add_image('dataset', image_id, img_path, annotation=ann_path)
            else:
                if filename in test_imagefiles:
                    self.add_image('dataset', image_id, img_path, annotation=ann_path)
                else:
                    pass
        
	# function to extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
       	# load and parse the file
       	tree = ET.parse(filename)
       	# get the root of the document
       	root = tree.getroot()
       	# extract each bounding box
       	boxes = list()
       	for box in root.findall('.//bndbox'):
       		xmin = int(box.find('xmin').text)
       		ymin = int(box.find('ymin').text)
       		xmax = int(box.find('xmax').text)
       		ymax = int(box.find('ymax').text)
       		coors = [xmin, ymin, xmax, ymax]
       		boxes.append(coors)
       	# extract image dimensions
       	width = int(root.find('.//size/width').text)
       	height = int(root.find('.//size/height').text)
    
        classname = [x.text for x in root.findall('.//name')]
       	return boxes, width, height, classname
        
    
    def load_mask(self, image_id, extract_boxes = extract_boxes):
        #get image info
        info = self.image_info[image_id]
        #path to xml file
        xml_path = info['annotation']
        #load xml
        boxes, w, h, classname = extract_boxes(self, xml_path)
        
        #create one array for all masks, each one in a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        
        #create masks
        class_ids = []
        for i in range(len(boxes)):
            box = boxes[i]
            #get the extent of the mask
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            # print(classname[i])
            if classname[i] == 'Tree':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index(classname[i]))
            elif classname[i] == 'Shrub':
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index(classname[i]))

        return masks, np.asarray(class_ids)

	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# class that defines and loads the kangaroo dataset
class TestDataset(Dataset):
 	# load the dataset definitions
    
    def load_dataset(self, imagepath, ann_path, model_name = 'tree'):
        #add classes to be identified by the model
        self.add_class('dataset',1,'Tree')
        self.add_class('dataset',2,'Shrub')      
                

        #get image info
        image_id = imagepath.split('/')[-1]

        self.add_image('dataset', image_id, imagepath, annotation = ann_path)

        
 	# function to extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
       	# load and parse the file
       	tree = ET.parse(filename)
       	# get the root of the document
       	root = tree.getroot()
       	# extract each bounding box
       	boxes = list()
       	for box in root.findall('.//bndbox'):
       		xmin = int(box.find('xmin').text)
       		ymin = int(box.find('ymin').text)
       		xmax = int(box.find('xmax').text)
       		ymax = int(box.find('ymax').text)
       		coors = [xmin, ymin, xmax, ymax]
       		boxes.append(coors)
       	# extract image dimensions
       	width = int(root.find('.//size/width').text)
       	height = int(root.find('.//size/height').text)
    
        classname = [x.text for x in root.findall('.//name')]
       	return boxes, width, height, classname
    
    
    def load_mask(self, image_id, classname='Tree', extract_boxes = extract_boxes):
        #get image info
        info = self.image_info[image_id]
        #path to xml file
        xml_path = info['annotation']
        #load xml
        boxes, w, h, classname = extract_boxes(self, xml_path)
        
        #create one array for all masks, each one in a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        
        #create masks
        class_ids = []
        for i in range(len(boxes)):
            box = boxes[i]
            #get the extent of the mask
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            # print(classname[i])
            if classname[i] == 'Tree':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index(classname[i]))
            elif classname[i] == 'Shrub':
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index(classname[i]))

        return masks, np.asarray(class_ids)

 	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
#model config class   
class ModelConfig(Config):
    #name the config
    NAME='model_cfg'
    #number of classes (including background)
    NUM_CLASSES = 1+2

    #number of training steps per Epoch
    STEPS_PER_EPOCH = 71
    
#define the prediction configuration
class PredictConfig(Config):
    #name the config
    NAME='model_cfg'
    #number of classes (including background)
    NUM_CLASSES = 1+2
    #simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = np.expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = np.mean(APs)
	return mAP

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = np.expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# define subplot
		plt.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		plt.imshow(image)
		plt.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			plt.imshow(mask[:, :, j], cmap='gray', alpha=0.999)
		# get the context for drawing boxes
		plt.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		plt.imshow(image)
		plt.title('Predicted')
		ax = plt.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	plt.show()
    
    
