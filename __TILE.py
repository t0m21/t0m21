#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:02:50 2020

@author: tes520
"""


import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime as dtm

from __TREES import Trees
from __CLASS import GetGeoTransform

from osgeo import gdal

import re
import pickle
import os
import math

class BM_MODEL():
    """
    Class which takes any tiff image and outputs biomass predictions based on UAV biomass model.
    """
    
    def __init__(self, 
                 img_path):
        
        
        self.img_path = img_path
        self.obia_path = img_path.replace('UAV', 'OBIA')
        self.plot_path = '/'.join(img_path.split('/')[:10])
        self.gt = GetGeoTransform(img_path)
        
        
    def set_tile_treecount(self, 
                           RCNN_pickle_path = '/Volumes/LaCie/Data_Analysis/Fuel_Load/data/RCNN_treecount/'):
        """
        Parameters
        ----------
        RCNN_pickle_path : STRING, optional
            DESCRIPTION. Path to folder containing box-algorithm pickle files. 
            The default is 'Volumes/LaCie/Data_Analysis/Fuel_Load/data/RCNN_treecount/'.

        Returns
        -------
        Tree count in the given tile

        """
        
        #Calculate RCNN treecount
        TreeClass = Trees(self.plot_path)
        
        #transect information
        country = self.img_path.split('/')[6]
        transect_numbers = re.findall(r'\d+', self.img_path)[1:3]
        
        #if there is no existing pickle file for detections in this pixel, go ahead and make one
        if not os.path.exists(RCNN_pickle_path+'Transect_data_'+transect_numbers[0]+'_'+country+'_treedata.pickle'):
            if not os.path.exists(RCNN_pickle_path+'Transect_data_'+transect_numbers[1]+'_'+country+'_treedata.pickle'):

                #set default parameters & values
                TreeClass.set_vals()
                
                #dictionary of parent detection boxes of trees & shrubs
                detections = TreeClass.BoxAlgorithmPt1()
                parents = TreeClass.BoxAlgorithm(detections)
            
                with open(RCNN_pickle_path+'Transect_data_'+transect_numbers[1]+'_'+country+'_treedata.pickle', 'wb') as picklefile:
                    pickle.dump(parents, picklefile)
            else:
                
                with open(RCNN_pickle_path+'Transect_data_'+transect_numbers[1]+'_'+country+'_treedata.pickle', 'rb') as picklefile:
                    parents = pickle.load(picklefile)
                
                
        else:
            
            with open(RCNN_pickle_path+'Transect_data_'+transect_numbers[0]+'_'+country+'_treedata.pickle', 'rb') as picklefile:
                parents = pickle.load(picklefile) 
               
               
        shrubcount, treecount = TreeClass.TreeCounter(self.plot_path, self.img_path, parents = parents)
        
        return shrubcount+treecount
    
    def set_tsf(self,
                tsf_path = '/Volumes/LaCie/Data/Sat_data/MODIS/MCD64A1.006/Time since fire/'):
        
        #read in tsf raster file to array
        tsf_raster = gdal.Open(tsf_path+'/s_a_time_since_fire_reprj.tif').ReadAsArray()
        
        with open(tsf_path+'/s_a_lons.pickle', 'rb') as F:
            tsf_lons = pickle.load(F)
            
        with open(tsf_path+'/s_a_lats.pickle', 'rb') as F:
            tsf_lats = pickle.load(F)
        
        #slice indexing information
        lat_ind = np.where(abs(tsf_lats-np.mean([self.gt['uly'],self.gt['lry']])) == 
                                    min(abs(tsf_lats-np.mean([self.gt['uly'],self.gt['lry']]))))[0][0]
        lon_ind = np.where(abs(tsf_lons-np.mean([self.gt['ulx'],self.gt['lrx']])) == 
                                    min(abs(tsf_lons-np.mean([self.gt['ulx'],self.gt['lrx']]))))[0][0]
        
        #time since fire in datetime format
        tsf_point = tsf_raster[:,lat_ind, lon_ind]
        tsf_date = dtm.datetime(year = tsf_point[0], month = 1, day = 1) + dtm.timedelta(days = float(tsf_point[1]))
        # tsf_delta = (date_of_fire - tsf_date)
        
        
        return tsf_date 
    
    def set_tile_meteorology(self, 
                             path_to_met_data = '/Volumes/LaCie/Data/Met_data/ECMWF/met_ecmwf_africa_2.nc',
                             set_tsf = set_tsf, 
                             set_tile_treecount = set_tile_treecount):
        
        #read in the meteorology dataset
        met_data = nc.Dataset(path_to_met_data)
        
        #lat/lon info
        met_lats = met_data['latitude'][:]
        met_lons = met_data['longitude'][:]
        
        #slice indexing information
        #spatial
        lat_ind = np.where(abs(met_lats-np.mean([self.gt['uly'],self.gt['lry']])) == 
                                    min(abs(met_lats-np.mean([self.gt['uly'],self.gt['lry']]))))[0][0]
        lon_ind = np.where(abs(met_lons-np.mean([self.gt['ulx'],self.gt['lrx']])) == 
                                    min(abs(met_lons-np.mean([self.gt['ulx'],self.gt['lrx']]))))[0][0]
        
        #temporal
        with open(self.plot_path+'/pre-fire/date.txt', 'r') as F:
            date_txt = [int(x) for x in F.read().split(';')]
            date_of_fire = dtm.datetime(date_txt[2], date_txt[1], date_txt[0])
        
        #time since fire
        tsf_date = set_tsf(self)
        tsf_delta = (date_of_fire - tsf_date).days
        
        #RCNN_count
        RCNN_count = set_tile_treecount(self)
        
        #most recent met datapoint previous to fire
        closest_previous_date_ind = np.where(np.array([(x - date_of_fire).days for x in [dtm.datetime(1900,1,1) + dtm.timedelta(hours=int(x)) for x in met_data['time'][:]]]) < 0)[0][-1]
        #start date for averaging
        start_date_ind = int(closest_previous_date_ind - math.ceil(12*(tsf_delta/365)))
        
        #relevant met data
        precipitation = np.ma.sum(met_data['tp'][start_date_ind:closest_previous_date_ind, lat_ind, lon_ind])
        mean_ssr = np.ma.mean(met_data['ssrd'][start_date_ind:closest_previous_date_ind, lat_ind, lon_ind])
        mean_temp = np.mean(met_data['t2m'][start_date_ind:closest_previous_date_ind, lat_ind, lon_ind])
        mean_soil_moisture = np.ma.mean(met_data['swvl1'][start_date_ind:closest_previous_date_ind, lat_ind, lon_ind])
        sum_evaporation = np.ma.sum(met_data['e'][start_date_ind:closest_previous_date_ind, lat_ind, lon_ind])
        
        #organise these into a dataframe
        met_df = pd.DataFrame({'map_tsf': tsf_delta,
                               'sum_precipitation': precipitation,
                               'mean_soil_moisture': mean_soil_moisture,
                               'sum_evaporation': sum_evaporation,
                               'mean_ssr': mean_ssr,
                               'RCNN_count': RCNN_count,
                               'mean_temp': mean_temp}, index = [0])
        
        return met_df
    
    
    def set_tile_proportions(self):
        
        #read in the classifier to array
        obia_raster = gdal.Open(self.obia_path).ReadAsArray()
        
        #mask shadows & no data values in the classifier
        classifier = np.ma.masked_equal(np.ma.masked_equal(np.ma.masked_equal(obia_raster, -999), 5),0)
        
        vals, counts = np.unique(classifier, return_counts = True)
        #mask the counts array to exclude shadows & no_data
        counts = np.ma.masked_array(counts, mask = vals.mask)
        
        #calculate the proportions of each class
        sum_pixels = np.ma.sum(counts)
        props = [x/sum_pixels for x in counts.compressed()]
        if len(props) < 4:
            props.append(0)
        
        if len(props) < 2:
            prop_df = None
        else:
            prop_df = pd.DataFrame({'Prop_bare_soil': props[0],
                                    'Prop_foliage': props[1],
                                    'Prop_grass': props[2]}, index = [0])
        
        return prop_df
        
    
    def calc_biomass(self, fuel_class, 
                     model_coeff_path = '/Volumes/LaCie/Data_Analysis/Fuel_Load/data/model_coefficients.xlsx',
                     set_tile_meteorology = set_tile_meteorology,
                     set_tile_proportions = set_tile_proportions):
        
        
        #read in coefficient dataframe for the desired fuel class
        coeffs = pd.read_excel(model_coeff_path, index_col=0).loc[fuel_class]
        
        #calculate meteorological data
        met_df = set_tile_meteorology(self)
        
        #calculate UAV data
        UAV_df = set_tile_proportions(self)
        
        if UAV_df is None:
            
            return -999
        else:
            
            #calculate biomass
            biomass = met_df['map_tsf']*coeffs['map_tsf']+\
                        met_df['sum_precipitation']*coeffs['sum_precipitation']+\
                        met_df['mean_soil_moisture']*coeffs['mean_soil_moisture']+\
                        met_df['sum_evaporation']*coeffs['sum_evaporation']+\
                        met_df['mean_ssr']*coeffs['mean_ssr']+\
                        met_df['RCNN_count']*coeffs['RCNN_count']+\
                        met_df['mean_temp']*coeffs['mean_temp']+\
                        UAV_df['Prop_bare_soil']*coeffs['Prop_bare_soil']+\
                        UAV_df['Prop_foliage']*coeffs['Prop_foliage']+\
                        UAV_df['Prop_grass']*coeffs['Prop_grass']
                        
                        
            return biomass.values
        
        
        
            
    
    
        
        
        
        