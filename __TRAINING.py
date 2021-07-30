#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:15:53 2020

@author: tes520
"""

from osgeo import gdal
from glob import glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import subprocess
import numpy as np
import __CLASS as CL

class Training():

    def __init__(self, path, NoData = -999):
        
        self.path = path
        self.shapefiles = glob(path+'/*.shp')
        if len(self.shapefiles) == 0:
            
            raise FileNotFoundError('No training areas found in '+path)
            
        self.classes = [x.split('/')[-1].split('.')[0] for x in glob(path+'/*.shp')]
        self.nodata = NoData
        
    def Match_Extents(self, img_path, nodata = -999, overwrite = False):
        """
        Match the extents of the training areas to the original image
        """
        #geographical target data
        geotransform = CL.Classifier(img_path).GetGeoTransform()
        
        #update the subprocess command string
        extent = '-te '+str(geotransform['ulx'])+' '+str(geotransform['lry'])+' '+str(geotransform['lrx'])+' '+str(geotransform['uly'])
        res = '-tr '+str(geotransform['xres'])+' '+str(geotransform['yres'])
        #res = '-ts '+str(full_height)+' '+str(full_width)
        nodata = '-dstnodata '+str(-999)
        
        if overwrite == True:
            ovr = '-overwrite'
        else:
            ovr = ''
        
        #loop over the classes
        for cat in self.shapefiles:
            
            cmd = 'gdalwarp -of GTiff -r bilinear '+extent+' '+res+' '+nodata+' -cutline '+cat+' -crop_to_cutline '+img_path\
            +' '+self.path+cat.split('/')[-1].split('.')[0]+'.tif'+' '+ovr+' --config GDALWARP_IGNORE_BAD_CUTLINE YES'
            
            #run the command in terminal
            subprocess.call(cmd, shell=True) 
        
        
    def training_areas_as_array(self):
        """
        Get the training areas as a dictionary of arrays
        """
        x = {}
        for shp in self.shapefiles:
            
            x[shp.split('/')[-1].split('.')[0]] = gdal.Open('.'.join(shp.split('.')[:-1])+'.tif').ReadAsArray()
            
    def split_train_test(self, balance = True, shuffle = True, scale = True, compress = True, onehot = False):
        """
        Split the given training data into training and testing data
        """
        
        if len(glob(self.path+'/*.tif')) == 0:
            
            raise IOError('No rasters of the training areas found in '+self.path+'\n\nPlease check they have been created!')
            
        x = {}
        for shp in list(sorted(self.shapefiles)):
            
            x[shp.split('/')[-1].split('.')[0]] = gdal.Open('.'.join(shp.split('.')[:-1])+'.tif').ReadAsArray()
            len_img = len(x[shp.split('/')[-1].split('.')[0]])
            
            #mask, strip and flatten the training areas
            x[shp.split('/')[-1].split('.')[0]] = np.ma.masked_equal(x[shp.split('/')[-1].split('.')[0]], self.nodata)
            if compress == True:
                x[shp.split('/')[-1].split('.')[0]] = x[shp.split('/')[-1].split('.')[0]].compressed().reshape(len_img,-1)
            else:
                x[shp.split('/')[-1].split('.')[0]] = x[shp.split('/')[-1].split('.')[0]].reshape(len_img,-1)
        
        #balance the training data
        if balance == True:
            
            #find the lowest number of samples in the classes
            len_min = min([len(y[1][0]) for y in x.items()])
            
            for cs in x.keys():
    
                #number of unique pixels available
                len_sample = len(x[cs][0])
                
                #generate a random index set
                ind_rand = np.random.choice(len_sample, len_min, replace = False)
                check_ind, counts = np.unique(ind_rand, return_counts = True)
                
                #check the uniqueness of the sample
                if len(ind_rand) != len(check_ind):
                    
                    raise ValueError("Subset of training set "+cs+" is not unique!")

                
                #take a subsample
                x[cs] = x[cs][:,ind_rand]

        #assign values to the classes
        print(list(sorted(self.shapefiles)))
        LABELS = {}
        # k=0
        for cat in list(sorted(self.shapefiles)):
            # k+=1
            if 'bare_soil' in cat:
                LABELS[cat.split('/')[-1].split('.')[0]] = 1
            elif 'foliage' in cat:
                LABELS[cat.split('/')[-1].split('.')[0]] = 2
            elif 'grass' in cat:
                LABELS[cat.split('/')[-1].split('.')[0]] = 3
            elif 'woody' in cat:
                LABELS[cat.split('/')[-1].split('.')[0]] = 4
            elif 'shadow' in cat:
                LABELS[cat.split('/')[-1].split('.')[0]] = 5
            
            
        #reformat to one training data list using one-hot vectors
        training_data = []
        for label in LABELS.keys():
            if onehot:
                if balance == True:
                    for i in tqdm(range(len_min)):
                        training_data.append([x[label][:,i],np.eye(len(LABELS))[LABELS[label]]])
                else:
                    for i in tqdm(range(len(x[label][0]))):
                        training_data.append([x[label][:,i],np.eye(len(LABELS))[LABELS[label]]])
            else:
                if balance == True:
                    for i in tqdm(range(len_min)):
                        training_data.append([x[label][:,i],LABELS[label]])
                else:
                    for i in tqdm(range(len(x[label][0]))):
                        training_data.append([x[label][:,i],LABELS[label]])
                    
        if shuffle == True:
            
            #shuffle training data
            np.random.shuffle(training_data)

        X = np.array([i[0] for i in training_data])
        Y = np.array([i[1] for i in training_data])
        
        if scale == True:
            X = StandardScaler().fit_transform(X)
            
        return X, Y
    
    
        
            
        
        
        
        
    