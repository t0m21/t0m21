#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 08:59:39 2020

@author: tes520
"""

import os
import pickle
import subprocess

from __RCNN import TestDataset, PredictConfig
from mrcnn.model import MaskRCNN, mold_image
from glob import glob
from osgeo import gdal
from __GEO import UpdateGT, GetGeoTransform
from scipy.ndimage import label
from skimage.measure import regionprops
from tqdm import tqdm
from scipy import misc
from PIL import Image

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



class Trees():
    """
    class of functions for the tree-counting algorithm
    """

    def __init__(self, path, model_path = '/Volumes/LaCie/Mapping/modules/Mask_RCNN/'):
        
        #working directory
        self.path = path
        self.model_path = model_path
        
        
    def set_vals(self, 
                 pixel_thresholds = [200,1.5e3,1e6],
                 DEM_threshold = 2,
                 sieve_num = 100,
                 sieve_num_foliage = 100,
                 box_overlap_threshold = 0.4,
                 area_overlap_threshold = 0.5,
                 pixel_area = 0.05**2,
                 tree_threshold = 3e3,
                 shrub_threshold = 1e4,
                 weight_rcnn = 2,
                 weight_obia = 1,
                 include_models = ['trees', 'shrubs','new_run'],
                 tree_side_ratio = 1.8,
                 num_rcnn_runs = 10,
                 use_OBIA = True,
                 OBIA_path = None,
                 im_path = None,
                 jpeg_path = None):
        
        """
        Set a number of parameters and thresholds for the algorithm
        """
        
        #pixel size thresholds 
        self.pixel_threshold_lower = pixel_thresholds[0]  #objects smaller than this are ignored
        self.pixel_threshold_mid = pixel_thresholds[1]    #objects smaller than this are likely shrubs, larger are likely trees
        self.pixel_threshold_upper = pixel_thresholds[2]  #objects larger than this are ignored 
        
        #DEM elevation threshold
        self.DEM_threshold = DEM_threshold #objects with height greater than this are classed as trees
        
        #sieve lower limits
        self.sieve_num = sieve_num
        
        #overlap thresholds for boxes
        self.box_overlap_threshold = box_overlap_threshold        #percentage shared overlap threshold for a box to be considered the child of another
        self.area_overlap_threshold = area_overlap_threshold  #as above, if neither box is partially contained within the other
        
        #pixel area
        self.pixel_area = pixel_area #area, in m2, of a single pixel
        
        #weights of models
        self.weight_rcnn = weight_rcnn  #weighting of RCNN detection
        self.weight_obia = weight_obia  #weighting of OBIA detection
        
        #RCNN trained models to include
        self.include_models = include_models
        
        #number of runs to use
        self.num_rcnn_runs = num_rcnn_runs
        
        #shrub and tree thresholds for class contrasting
        self.tree_threshold = tree_threshold * self.pixel_area
        self.shrub_threshold = shrub_threshold * self.pixel_area
        
        #ratio of box sides above which the box is likely to be a tree
        self.tree_side_ratio = tree_side_ratio
        
        #set whether to use OBIA in algorithm
        self.use_OBIA = use_OBIA
        self.OBIA_path = OBIA_path
        
        #image path
        self.im_path = im_path
        self.jpeg_path = jpeg_path
        
        
    def make_jpeg(self, path = None):
        """
        Make a jpeg file from the stack

        Returns
        -------
        None.
        """
        
        if path == None:
            im_path = glob(self.path+'/pre-fire/indices/*stack_total*')[0]
        else:
            im_path = path
        stack = np.swapaxes(np.swapaxes(gdal.Open(im_path).ReadAsArray()[5:],0,2),0,1)
        
        # plt.imsave('/'.join(im_path.split('/')[:-3])+'/pre_fire/'+im_path.split('/')[7]+'_'+im_path.split('/')[8]+'_pre.jpg', stack)
        plt.imsave('/'.join(im_path.replace('.tif','.jpg'), stack))
        # misc.toimage(stack, cmin=0.0, cmax=255).save(path+'/pre_fire/'+im_path.split('/')[6]+'_'+im_path.split('/')[7]+'_pre.jpg')
        
    def run_rcnn(self, KMP_duplicate = True, 
                 jpeg_path = '/Volumes/LaCie/Data_Analysis/trees/hires_images',
                 image_path = None):
        """
        Runs the variations on the RCNN model to output a set of boxes where trees or shrubs are detected.
        It assumes a data structure as follows:
        MODEL_PATH:
            -model_name1
            -model_name2
            ...
            -model_name(n)
                -main
                    -model_rundir
                        -weightsfile_1.h5
                        ....
                        -weightsfile_n.h5
                        
        N.B only the last 'model_rundir' folder will be used!
        """
        #be sure to use the right backend
        os.environ['KERAS_BACKEND']='tensorflow'
        
        #name of the image
        path_split = [x for x in self.path.split('/') if len(x)!=0]
        if image_path is None:
            image_name = self.jpeg_path

        else:
            image_name = image_path
        #set the environment to allow for multiple instances of OMP library
        if KMP_duplicate:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
          
        #empty lists for box and class data
        boxes = []
        class_ids = []
        
        #loop through each model in turn
        for model_name in self.include_models:
            
            print('Model: '+model_name)
            
            #list models available
            weights_list = list(set(glob(self.model_path+'/'+model_name+'/main/'+os.listdir(self.model_path+'/'+model_name+'/main')[-1]+'/*.h5')))[-self.num_rcnn_runs:]
            print(weights_list)
            #create config
            testcfg = PredictConfig()
            print('done')
    
            #define the model to be used for testing
            test_model = MaskRCNN(mode = 'inference', model_dir = self.model_path+'/'+model_name+'/main/', config = testcfg)
            print('done2')
            #search for the image in jpg form
            if image_path is None:
                # for root, dirs, files in os.walk(jpeg_path):
                    # if image_name in files:
                        # jpeg_file = root+'/'+image_name
                    # if image_name.split('.')[0]+'.xml' in files:
                        # xml_file = root+'/'+image_name.split('.')[0]+'.xml'
                jpeg_file = image_name
                xml_file = jpeg_file.replace('.jpg','.aux.xml')
            else:
                jpeg_file = image_path
                xml_file = jpeg_file.replace('.jpg', '.xml')
                print(xml_file)
            
            #run the predictor for each weights file and add the resulting boxes to the list
            for weightsfile in weights_list:
                print(weightsfile)
                
                #load the weights file into the model
                test_model.load_weights(weightsfile, by_name=True)
                print('Weights loaded')
                
                #load the desired image as a 'test' dataset
                test_img = TestDataset()
                test_img.load_dataset(jpeg_file, ann_path = xml_file)
                print('Test data loaded')

                
                #scale and fit the image to the required format
                image = test_img.load_image(0)
                print('Image loaded')
                scaled_image = mold_image(image, testcfg)
                expand_image = np.expand_dims(scaled_image, 0)
                print('Image scaled')
                
                #model output
                yhat = test_model.detect(expand_image, verbose=0)[0]
                print('Model complete!')
                
                #add box & class data to lists
                for box in yhat['rois']:
                    boxes.append(box)

                    
                for class_id in yhat['class_ids']:
                    class_ids.append(class_id)
        
        #save the boxes to file
        box_info = {}
        box_info['boxes'] = np.array(boxes)
        box_info['classes'] = class_ids
        if not os.path.exists(self.path+'/RCNN'):
            os.mkdir(self.path+'/RCNN')
        with open(self.path+'/RCNN/boxes.pickle', 'wb') as F:
            
            pickle.dump(box_info, F)
            
        
    def OBIA_trees(self):
        """
        Creates a TIFF file based on OBIA classification & DEM, 
        where shrub areas are assigned a value of 1 and tree areas a value of 2
        """
        
        #classifier and feature paths
        if self.OBIA_path == None:
            pre_fire_classifier = glob(self.path+'/pre-fire/Classification/OBIA/*corrected.tif')[0]
        else:
            pre_fire_classifier = self.OBIA_path
            
        if self.im_path == None:
            pre_fire_features = glob(self.path+'/pre-fire/indices/*total*.tif')[0]
        else:
            pre_fire_features = self.im_path
        
        #classifier and DEM arrays
        print(pre_fire_classifier)
        classifier = gdal.Open(pre_fire_classifier).ReadAsArray()
        DEM = (gdal.Open(pre_fire_features).ReadAsArray())[2]
        
        #make sure we're using the correct features TIFF
        if classifier.shape != DEM.shape:
            
            pre_fire_features = glob(self.path+'/pre-fire/indices/*total_cropped*.tif')[0]
            DEM = (gdal.Open(pre_fire_features).ReadAsArray())[2]
            
        #if its still off, crop the classifier to the features array
        if classifier.shape!= DEM.shape:
            
            #geotransform
            features_gt = GetGeoTransform(pre_fire_features)
            
            #crop command 
            crop_cmd = 'gdal_translate -projwin '+\
                        str(features_gt['ulx'])+' '+str(features_gt['uly'])+' '+str(features_gt['lrx'])+' '+str(features_gt['lry'])+\
                        ' -a_nodata -999.0 -of GTiff '+\
                        pre_fire_classifier+' '+\
                        pre_fire_classifier.split('.')[0]+'_cropped.tif'
                        
            subprocess.call(crop_cmd, shell=True)
            
            classifier = gdal.Open(pre_fire_classifier.split('.')[0]+'_cropped.tif').ReadAsArray()
 
        
        #check if a 'Trees' folder already exists
        # if not os.path.exists(self.path+'/pre-fire/Classification/Trees'):
        #     os.mkdir(self.path+'/pre-fire/Classification/Trees')
        if not os.path.exists(self.path+'/Trees'):
            os.mkdir(self.path+'/Trees')
            
        #mask all elements other than foliage in the classifier and turn into binary array
        mask = (np.ma.masked_equal(classifier, 2).mask).astype(int)
        
        #mask all elements from the DEM lower than 20cm
        mask_DEM = (np.ma.masked_greater(DEM, 0.2).mask).astype(int)
        
        #mask all elements from the DEM lower than the tree cut-off
        mask_2m = (np.ma.masked_greater_equal(DEM, self.DEM_threshold).mask).astype(int)
        
        #partition the mask into only connected pixels
        labeled_class, nlabels_class = label(mask)
        labeled_DEM, nlabels_DEM = label(mask_DEM)
        
        #set all areas where foliage is detected to 1
        labeled_class[labeled_class!=0] = 1
        
        #do the same for the DEM labels
        labeled_DEM[labeled_DEM!=0] = 1

        #add the two arrays together, such that anything over 2m is classed as a tree and assigned a value of 2
        new_labeled_DEM = labeled_DEM + mask_2m
        
        #create a new georeferenced tiff for all foliage objects
        # UpdateGT(self.path+'/pre-fire/Classification/Trees/foliage_class.tif', labeled_class, pre_fire_classifier)
        UpdateGT(self.path+'/Trees/foliage_class.tif', labeled_class, pre_fire_classifier)
        
        #filter the labeled class for objects smaller than a given threshold
        # sieve_cmd = 'gdal_sieve.py -st '+\
        #                 str(self.sieve_num)+\
        #                     ' -4 -of GTiff '+\
        #                         self.path+'/pre-fire/Classification/Trees/foliage_class.tif '+\
        #                             self.path+'/pre-fire/Classification/Trees/foliage_class_sieved.tif'                             
        sieve_cmd = 'gdal_sieve.py -st '+\
                        str(self.sieve_num)+\
                            ' -4 -of GTiff '+\
                                self.path+'/Trees/foliage_class.tif '+\
                                    self.path+'/Trees/foliage_class_sieved.tif'                             
        subprocess.call(sieve_cmd, shell=True)
        
        #open the sieved binary foliage array
        # foliage_arr = gdal.Open(self.path+'/pre-fire/Classification/Trees/foliage_class_sieved.tif').ReadAsArray()
        foliage_arr = gdal.Open(self.path+'/Trees/foliage_class_sieved.tif').ReadAsArray()
     
        #then add in any extra foliage detections as shrubs, making sure to not double count
        foliage_arr[new_labeled_DEM!=0] = 0
        output = new_labeled_DEM + foliage_arr
        
        #add this to a georeferenced TIFF
        # UpdateGT(self.path+'/pre-fire/Classification/Trees/trees_DEM.tif', output, pre_fire_classifier)
        UpdateGT(self.path+'/Trees/trees_DEM.tif', output, pre_fire_classifier)

    def BoxAlgorithmPt1(self, overwrite = False, run_rcnn = run_rcnn, OBIA_trees = OBIA_trees):
        """
        This algorithm loads in rectangle objects from OBIA and RCNN models, 
        decides which boxes are children of parent boxes, 
        and decides on the basis of several inputs which class should be assigned to a parent box.
        This is the first part, which returns a dictionary object of boxes, along with a full list of boxes and class IDs.
        """
        
        #read in the boxes file made from RCNN model
        if overwrite:
            run_rcnn(self) 
            if self.use_OBIA:
                OBIA_trees(self)
        else:
            if os.path.exists(self.path+'/pre-fire/Classification/boxes.pickle'):
                pass
            elif os.path.exists(self.path+'/RCNN/boxes.pickle'):
                pass
            #if no boxes are detected, run the RCNN model first
            else:
                run_rcnn(self)  
            if self.use_OBIA:
                #check if a classifier already exists
                if os.path.exists(self.path+'/pre-fire/Classification/Trees/trees_DEM.tif'):
                    pass
                #if no trees TIFF exists, make one!
                else:
                    OBIA_trees(self)
       
        #read in boxes from RCNN pickle file
        if os.path.exists(self.path+'/pre-fire/Classification/boxes.pickle'):
            with open(self.path+'/pre-fire/Classification/boxes.pickle', 'rb') as F:
                box_info =  pickle.load(F)
        elif os.path.exists(self.path+'/RCNN/boxes.pickle'):
            with open(self.path+'/RCNN/boxes.pickle', 'rb') as F:
                box_info =  pickle.load(F)
        
        if self.use_OBIA:
            #create boxes from OBIA classifier
            # img = gdal.Open(self.path+'/pre-fire/Classification/Trees/trees_DEM.tif').ReadAsArray()
            img = gdal.Open(self.path+'/OBIA_corrected.tif').ReadAsArray()
            #find boxes & classes using pure image segmentation
            foliage_lbls, nfoliage = label(img)
            f_boxes = []
            f_classes = []
            #for every region labelled, extract box boundas and classes
            for region in regionprops(foliage_lbls):
                
                f_boxes.append([x for x in region.bbox])
                y1,x1,y2,x2 = region.bbox
                f_classes.append(np.ma.max(img[y1:y2,x1:x2]))
                
            f_boxes = np.array(f_boxes)
    
        #concatenate boxes from RCNN and OBIA
        model_boxes = box_info['boxes']
        if self.use_OBIA:
            boxes = np.append(model_boxes, f_boxes, axis=0)
        else:
            boxes = model_boxes.copy()
        #do the same for class ids, flipping the identifier from RCNN
        #(in RCNN, Trees = 1 and Shrubs = 2)
        int_classes = np.zeros(len(box_info['classes']))
        int_classes[np.array(box_info['classes'])==1]=2
        int_classes[np.array(box_info['classes'])==2]=1
        
        if self.use_OBIA:
            class_ids = list(int_classes) + f_classes
        else:
            class_ids = list(int_classes)
        
        #list the model used for each box in order
        if self.use_OBIA:
            model_id = ['rcnn' for x in range(len(box_info['classes']))] + ['obia' for x in range(len(f_classes))]
        else:
            model_id = ['rcnn' for x in range(len(box_info['classes']))]
        #dictionary object to store box metadata
        detections = {}
        #list counter to prevent double counting & speed things up
        detection_id = [[] for x in range(len(boxes))]
        
        #initialise dictionary object for each box
        for b in range(len(boxes)):
            detections[str(b)] = {}
            detections[str(b)]['children'] = []
            detections[str(b)]['parents'] = []
            detections[str(b)]['has_parent'] = False
        
        #start of the algorithm
        for b in tqdm(range(len(boxes))):

            #box details
            y1, x1, y2, x2 = boxes[b]
            
            #pixels enclosed by rectangle
            checkbox_x_inds = np.arange(x1,x2+1)
            checkbox_y_inds = np.arange(y1,y2+1)
            
            checkbox_size = len(checkbox_x_inds)*len(checkbox_y_inds)
            
            if self.use_OBIA:
                #seperate the box into classes, including binary detection/no detection class
                zoom_img = img[y1:y2,x1:x2].copy()
                trees_img = zoom_img.copy()
                shrub_img = zoom_img.copy()
            
                trees_img[trees_img != 2] = 0
                shrub_img[shrub_img != 1] = 0
                zoom_img[zoom_img!=0] = 5
                
                #labels for the portion of the image contained within the box
                tree_labels, ntrees = label(trees_img)
                shrub_labels, nshrubs = label(shrub_img)
                label_img, nobjects = label(zoom_img)
                
                #area of each detection
                tree_sizes = [len(np.where(tree_labels==x)[0])*self.pixel_area for x in np.unique(tree_labels[tree_labels>0])]
                shrub_sizes = [len(np.where(shrub_labels==x)[0])*self.pixel_area for x in np.unique(shrub_labels[shrub_labels>0])]
            
                # detections[str(b)] = {}
                detections[str(b)]['nTrees'] = ntrees
                detections[str(b)]['nShrubs'] = nshrubs
                detections[str(b)]['nObjects'] = nobjects
                detections[str(b)]['tree_sizes'] = tree_sizes
                detections[str(b)]['shrub_sizes'] = shrub_sizes
            detections[str(b)]['class'] = class_ids[b]
            detections[str(b)]['box_area'] = checkbox_size*self.pixel_area
            detections[str(b)]['bbox'] = boxes[b]
            detections[str(b)]['model'] = model_id[b]
            detections[str(b)]['x'] = checkbox_x_inds
            detections[str(b)]['y'] = checkbox_y_inds
            
            child=False
            for j in range(len(boxes)):
            #keep looping over other boxes, unless the box is found to be the child of another
                if b==j:
                    continue
                # get coordinates
                y1, x1, y2, x2 = boxes[j]
                # get indices of array which box encases
                x_inds = np.arange(x1,x2+1)
                y_inds = np.arange(y1,y2+1)
                #box info
                box_size = len(x_inds)*len(y_inds)

                #intersection of parent box with this one, as a percentage
                x_intersect = len(np.intersect1d(checkbox_x_inds, x_inds))
                y_intersect = len(np.intersect1d(checkbox_y_inds, y_inds))
                intersect_parent = x_intersect*y_intersect/(len(checkbox_x_inds)*len(checkbox_y_inds))
                intersect_child = x_intersect*y_intersect/(len(x_inds)*len(y_inds))
        
                #is one box wholly contained within another?
                if set(x_inds).issubset(set(checkbox_x_inds)) or set(checkbox_x_inds).issubset(set(x_inds)):
                    
                    #if it is, assign the same detection ID to both boxes
                    if set(y_inds).issubset(set(checkbox_y_inds)) or set(checkbox_y_inds).issubset(set(y_inds)):
                        #if more than a given percentage of the area of the parent box ix covered by the child, then consider them the same detection
                        if intersect_parent >= self.box_overlap_threshold or intersect_child >= self.box_overlap_threshold:
                            #set the detection ID to the parent box number, avoiding double-counting
                            if j not in detection_id[b]:
                                detection_id[b].append(j)
                                detections[str(b)]['children'].append([j,class_ids[j], model_id[j]])
                                detections[str(j)]['parents'].append([b,class_ids[b], model_id[b]])
                    
                    # if not, check what percentage of the parent/child boxes intersect
                    else:
                        #if both have large overlap, consider them the same detection
                        if intersect_child >= self.box_overlap_threshold and intersect_parent >= self.box_overlap_threshold:
                            #whichever box is larger will be considered the 'parent'
                            if checkbox_size >= box_size:
                                if j not in detection_id[b]:
                                    detection_id[b].append(j)
                                    detections[str(b)]['children'].append([j,class_ids[j], model_id[j]])
                                    detections[str(j)]['parents'].append([b,class_ids[b], model_id[b]])
                            else:
                                if b not in detection_id[j]:
                                    detection_id[j].append(b)
                                    detections[str(b)]['parents'].append([j,class_ids[j], model_id[j]])
                                    detections[str(j)]['children'].append([b,class_ids[b], model_id[b]])
                                    child=True
                        
                elif set(y_inds).issubset(set(checkbox_y_inds)) or set(checkbox_y_inds).issubset(set(y_inds)):
                    #if both have large overlap, consider them the same detection
                    if intersect_child >= self.box_overlap_threshold and intersect_parent > self.box_overlap_threshold:
                        #whichever box is larger will be considered the 'parent'
                        if checkbox_size >= box_size:
                            if j not in detection_id[b]:
                                detection_id[b].append([j,class_ids[j]])
                                detections[str(b)]['children'].append([j,class_ids[j], model_id[j]])
                                detections[str(j)]['parents'].append([b,class_ids[b], model_id[b]])
                              
                        else:
                            if b not in detection_id[j]:
                                detection_id[j].append(b)
                                detections[str(b)]['parents'].append([j,class_ids[j], model_id[j]])
                                detections[str(j)]['children'].append([b,class_ids[b], model_id[b]])
                                child=True
                                
                #if no one side is contained within the other, check if both have major overlap and assign the larger box the parent role         
                else:
                     #if both have large overlap, consider them the same detection
                    if intersect_child >= self.area_overlap_threshold and intersect_parent > self.area_overlap_threshold:
                        #whichever box is larger will be considered the 'parent'
                        if checkbox_size >= box_size:
                            if j not in detection_id[b]:
                                detection_id[b].append([j,class_ids[j]])
                                detections[str(b)]['children'].append([j,class_ids[j], model_id[j]])
                                detections[str(j)]['parents'].append([b,class_ids[b], model_id[b]])
                              
                        else:
                            if b not in detection_id[j]:
                                detection_id[j].append(b)
                                detections[str(b)]['parents'].append([j,class_ids[j], model_id[j]])
                                detections[str(j)]['children'].append([b,class_ids[b], model_id[b]])
                                child=True                  
                if child:
                    detections[str(b)]['has_parent']=True
                    # break
                    
        return detections
    
    
    def BoxAlgorithm(self, detections = None,
                     overwrite = False, run_rcnn = run_rcnn, OBIA_trees = OBIA_trees,
                     BoxAlgorithmPt1 = BoxAlgorithmPt1, use_OBIA = True):
        """
        The full algorithm to detect trees and shrubs. 
        If no inputs are given, calculates them from the part 1 function. 
        """
        #if one or more of the inputs aren't given, run part 1
        if detections == None:
            detections = BoxAlgorithmPt1(self, overwrite = overwrite, run_rcnn = run_rcnn, OBIA_trees = OBIA_trees)
            
        # print(detections.keys())

        class_ids = [detections[x]['bbox'] for x in detections.keys()]
        parent_boxes = []
        loop_boxes = [x for x in detections.keys() if detections[x]['has_parent'] == False]
        # print(loop_boxes)
        
        #loop through given parent boxes and apply classification criteria to them
        for box in loop_boxes:
            # print(box)
            #filter out boxes that are too big
            if detections[box]['box_area'] < self.pixel_threshold_upper:
                #make sure boxes designated as trees are large enough
                if detections[box]['class'] == 2 and detections[box]['box_area'] < self.tree_threshold:
                    continue
                #or that boxes designated as shrubs aren't unrealistically large
                elif detections[box]['class'] == 1 and detections[box]['box_area'] > self.shrub_threshold:
                    continue
                else:
                    parent_boxes.append(box)
                    # print('test')
                    
                    # children = np.array([x[0] for x in detections[box]['children']])
                    child_classes = [x[1] for x in detections[box]['children']]
                    child_models = [x[2] for x in detections[box]['children']]
                    c = detections[box]['class']
                    model = detections[box]['model']
        
                    #give weights to the detected classed from rcnn and obia
                    if use_OBIA:
                        weights = {}
                        weights['obia'] = self.weight_obia
                        weights['rcnn'] = self.weight_rcnn
                        w = np.array([weights[x] for x in [model]+child_models])
                    
                    #decide the class of the parent box based on the weighted average of it and the child boxes
                    classes = np.append(np.array(c),child_classes)
                    # if use_OBIA:
                        # weighted_class = np.round(np.average(classes,weights=w))
                        
                    # else:
                    weighted_class = np.round(np.mean(classes))
                    
                    #the weight of the class detection depends on the number of classes with the same choice
                    uniq_class, class_counts = np.unique(classes, return_counts = True)
                    numweight = class_counts[np.where(uniq_class == weighted_class)[0]]
                    
                    #if the detected box is large but there are multiple small detections within it, dismiss the smaller ones
                    
                    #box dimensions test - if the box has one side almost twice as long as the other, strong chance it is a tree.
                    height = detections[box]['bbox'][2] - detections[box]['bbox'][0]
                    width = detections[box]['bbox'][3] - detections[box]['bbox'][1]
                    if height > self.tree_side_ratio * width or width > self.tree_side_ratio * height:
                        dimclass = 2
                        dimweight = 11
                    else:
                        dimclass = None
                        dimweight = None
                    
                    #box size test - if the box is very large and has a single detection, likely to be a tree.
                    area = detections[box]['box_area']
                    nobjects = detections[box]['nObjects']
                    if area < self.pixel_threshold_lower:
                        sizeclass = 1
                        sizeweight = 10
                    elif area > 0.4*self.pixel_threshold_upper:
                        sizeclass = 2
                        sizeweight = 10
                    elif area > 0.4*self.pixel_threshold_upper and nobjects == 1:
                        sizeclass = 2
                        sizeweight = 20
                    else:
                        sizeclass = 2
                        sizeweight = 1
                    
                    #eliminate any variables which aren't given
                    var = [weighted_class,dimclass,sizeclass]
                    wts = [numweight, dimweight, sizeweight]
                    while None in var:
                        var.remove(None)
                    while None in wts:
                        wts.remove(None)

                    #assign the new, weighted class to this box
                    newclass = np.round(np.average(np.array(var), weights = np.array(wts)))
                    class_ids[int(box)] = newclass
                    
                    detections[box]['class'] = newclass
                    
            else:
                print('Oversized box '+box+':',detections[box]['box_area'])
                
            
        #boxes with parents
        parent_detections = {}
        # print(parent_boxes)
        for box in parent_boxes:
            parent_detections[box] = detections[box]
        
        return parent_detections
    
    def plot_boxes(self, detections = None, save = True, 
                   jpg_path = '/Volumes/LaCie/Data_Analysis/trees',
                   BoxAlgorithm = BoxAlgorithm):
        """
        Overlays the detection boxes onto the plot image
        """
        
        #make a detections array if it doesn't exist
        if detections == None:
            detections = BoxAlgorithm()
            
        #path to jpeg image
        location = [x for x in self.path.split('/') if len(x) > 0][-2]
        plot = [x for x in self.path.split('/') if len(x) > 0][-1]
        
        #read in image
        img_path = jpg_path+'/hires_images/images/'+location+'_'+plot+'_pre.jpg'
        img = np.swapaxes(np.swapaxes(gdal.Open(img_path).ReadAsArray(),0,2),0,1)
        
        #add rectangles to image
        plt.figure(figsize=(30,26))
        plt.imshow(img)
        ax = plt.gca()
        
        for b in detections:
        
            # get coordinates
            y1, x1, y2, x2 = detections[b]['bbox']
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            	# create the shape
            if detections[b]['class'] == 1:
            	rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            elif detections[b]['class'] == 2:
                rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
        
        if save:
            plt.savefig(jpg_path+'/RCNN_output_images/'+location+'_'+plot+'_boxes.png')
    
    
    
    def TreeCounter(self, tile, use_OBIA = True, plot_path = None, parents = None, nodataval = -999, 
                    BoxAlgorithm = BoxAlgorithm):
        """
        Counts trees and shrubs for a given transect or tile
        """
        
        #if no parent box data is given, calculate it
        if parents == None:
            parents = BoxAlgorithm(self)
        
        #open the tile as an array, and get geotransform data
        if type(tile) == str:
            tile_array = gdal.Open(tile).ReadAsArray()
            tile_gt = GetGeoTransform(tile)
        elif type(tile) == int:
            transect_string = plot_path+'/fuel_load/transect_'+str(tile)+'_pre-fire_classes.tif'
            tile_array = gdal.Open(transect_string).ReadAsArray()
            tile_gt = GetGeoTransform(transect_string)
        else:
            raise TypeError('tile input must be either a path to a TIFF file, or transect number')
            
        #plot geotransform data
        # if plot_path is None:
        plot_gt = GetGeoTransform(self.path+'/pre-fire/Classification/OBIA/OBIA_corrected.tif')
        # else:
            # plot_gt = GetGeoTransform(plot_path)
        
        #extract lat/lon data for the plot
        plot_lons = np.arange(plot_gt['ulx'], plot_gt['lrx'], plot_gt['xres'])
        plot_lats = np.arange(plot_gt['uly'], plot_gt['lry'], plot_gt['yres'])
        
        #lat/lon data for the transect
        tile_lons = np.arange(tile_gt['ulx'], tile_gt['lrx'], tile_gt['xres'])
        tile_lats = np.arange(tile_gt['uly'], tile_gt['lry'], tile_gt['yres'])
        
        #find the indices of the transect within the plot image (including masked areas)
        try:
            lon_inds = np.arange(np.where(plot_lons>tile_lons[0])[0][0]-1, np.where(plot_lons>tile_lons[-1])[0][0])
        except IndexError:
            lon_inds = np.arange(np.where(plot_lons>tile_lons[0])[0][0]-1, len(plot_lons)-1)
        try:
            lat_inds = np.arange(np.where(plot_lats>tile_lats[0])[0][-1]-1, np.where(plot_lats>tile_lats[-1])[0][-1])
        except IndexError:
            lat_inds = np.arange(0,np.where(plot_lats>tile_lats[-1])[0][-1])
            
        # x,y = np.meshgrid(lon_intersect, lat_intersect)
        
        #mask of the nodata areas around the transect
        if type(tile) == int:
            mask = np.ma.masked_equal(tile_array, -999).mask
        else:
            mask = np.zeros(tile_array[0].shape)
            mask[mask == 0] = False
        
        #counts for shrubs and trees
        shrubcounter = 0
        treecounter = 0
        
        #check every box for overlap with the transect
        for box in parents.keys():
            
            #intersect of the box coordinates with the transect
            lon_intersect = np.intersect1d(parents[box]['x'],lon_inds)
            lat_intersect = np.intersect1d(parents[box]['y'],lat_inds)
            
            #if there is an intersection, carry on
            if len(lon_intersect) > 0 and len(lat_intersect)> 0:
            
                #pixel counts of the box and of the part of the box in the transect
                pixelsize_box = len(parents[box]['x'])*len(parents[box]['y'])
            
                #boolean array of the intersection overlaid by nodata regions
                intersect_mask = mask[np.where(lon_inds == lon_intersect[0])[0][0]:np.where(lon_inds == lon_intersect[-1])[0][0],
                                      np.where(lat_inds == lat_intersect[0])[0][0]:np.where(lat_inds == lat_intersect[-1])[0][0]]
                
                #relative occurences of nodata values in the box region
                maskval, mask_count = np.unique(intersect_mask, return_counts=True)
                
                #if less than 50% of the box is in the section, and it is a shrub, do not count it
                if mask_count[maskval==False]/pixelsize_box >= 0.5 and parents[box]['class'] == 1:
                    shrubcounter += 1
                
                #if 20% of the box is in the section and it is a tree, count it
                elif mask_count[maskval==False]/pixelsize_box >= 0.2 and parents[box]['class'] == 2:                
                    treecounter += 1
                    
        return shrubcounter, treecounter
            
        
                
                
        
            
        
            
        
            
        
                
                
                
                


        
        
        
    
        
        
        
        
        