#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:35:52 2020

@author: tes520
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime as dtm
import matplotlib.pyplot as plt
import matplotlib

import xlrd
import subprocess
import os
import glob
import pickle
import itertools
import math

from tqdm import tqdm
from osgeo import ogr, gdal
from scipy.ndimage import label
from scipy import stats
from shutil import copyfile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm

from scipy.optimize import curve_fit

from __GEO import UpdateGT, GetGeoTransform


def Read_NCDF(year=None, path = '/Volumes/LaCie/Data/Met_data/ECMWF', region = 'Africa'): 

    if region == None:
        x_start = 0
        x_end = 3600       
        y_start = 0
        y_end = 1801
    # elif region == 'Africa':
    #     x_start = 0
    #     x_end = 600      
    #     y_start = 800
    #     y_end = 1400

    if year is not None:
        #read data in from given 
        data = nc.Dataset(path+'/met_ecmwf_'+str(year))
        
        #organise the data into a dictionary object
        df = {'Longitude': data['longitude'][x_start:x_end],
                           'Latitude': data['latitude'][y_start:y_end],
                           'Time': [dtm.datetime(1900,1,1) + dtm.timedelta(hours=int(x)) for x in data['time'][:]],
                            'SST': data['sst'][:,y_start:y_end,x_start:x_end],
                            'Surface Soil Temp': data['stl1'][:,y_start:y_end,x_start:x_end],
                            'Convective Precipitation': data['cp'][:,y_start:y_end,x_start:x_end],
                            'Surface Sensible Heat flux': data['sshf'][:,y_start:y_end,x_start:x_end],
                            'Cloud cover': data['tcc'][:,y_start:y_end,x_start:x_end],
                            'Surface net solar radiation': data['ssr'][:,y_start:y_end,x_start:x_end],
                            'Evaporation': data['e'][:,y_start:y_end,x_start:x_end],
                            'Total precipitation': data['tp'][:,y_start:y_end,x_start:x_end]}

    else:
        
        data = nc.Dataset(path+'/met_ecmwf_africa_2.nc')

        df = {'Longitude': data['longitude'][:],
                           'Latitude': data['latitude'][:],
                           'Time': [dtm.datetime(1900,1,1) + dtm.timedelta(hours=int(x)) for x in data['time'][:]],
                            '2m Temperature': data['t2m'][:,:,:],
                            'Soil Temp layer 1': data['stl1'][:,:,:],
                            'Evaporation': data['e'][:,:,:],
                            'Heat flux': data['sshf'][:,:,:],
                            'Solar radiation': data['ssrd'][:,:,:],
                            'Solar thermal radiation': data['strd'][:,:,:],
                            'Evaporation from bare soil': data['evabs'][:,:,:],
                            'Evaporation from vegetation': data['evavt'][:,:,:],
                            'Soil water 1': data['swvl1'][:,:,:],
                            'Soil water 2': data['swvl2'][:,:,:],
                            'Total precipitation': data['tp'][:,:,:],
                            'Wind speed':np.sqrt(data['u10'][:,:,:]**2 + data['v10'][:,:,:]**2),
                            'LAI HV':data['lai_hv'][:,:,:],
                            'LAI LV':data['lai_lv'][:,:,:]}
        
                                
    return df

def Patchiness(path):

    #read in classification files
    pre_path = path+'/pre-fire/Classification/OBIA/OBIA_corrected.tif'
    post_path = path+'/post-fire/Classification/OBIA/OBIA_corrected.tif'
    
    
    pre_classes = gdal.Open(pre_path).ReadAsArray()
    post_classes = gdal.Open(post_path).ReadAsArray()
    
    #if the arrays are already the same shape, then simply mask the vegetation classes by the unburned area
    try:
        
        unburned_mask = np.ma.masked_equal(np.round(post_classes), 1).mask
#        burned_mask = np.ma.masked_equal(np.round(post_classes), 2).mask
        unburned_classes = np.ma.masked_outside(np.ma.round(np.ma.masked_equal(np.ma.masked_array(pre_classes, mask = unburned_mask),-999)), 1.1, 4.1)
#        burned_vegetation = np.ma.masked_inside(np.ma.round(np.ma.masked_equal(np.ma.masked_array(pre_classes, mask = burned_mask),-999)), 1.1, 4.1)
        labeled_arr, nlabels = label(unburned_classes.mask)
        
        patches = np.zeros(labeled_arr.shape)
        patches[labeled_arr!=1] = 1
        
        if os.path.exists(path+'/burned_area_mask.shp'):
            cropline = path+'/burned_area_mask.shp'
        elif os.path.exists(path+'/Overlap.shp'):
            cropline = path+'/Overlap.shp'
        elif os.path.exists(path+'/overlap.shp'):
            cropline = path+'/overlap.shp'
        
        for i in [pre_path, post_path]:
            if os.path.exists('/'.join(i.split('/')[:-3])+'/Classification/Patchiness'):
                if os.path.exists('/'.join(i.split('/')[:-3])+'/Classification/Patchiness/OBIA.tif'):
                    pass
                else:
                    copyfile(i, '/'.join(i.split('/')[:-3])+'/Classification/Patchiness/OBIA.tif')
            else:
                os.mkdir('/'.join(i.split('/')[:-3])+'/Classification/Patchiness')
                copyfile(i, '/'.join(i.split('/')[:-3])+'/Classification/Patchiness/OBIA.tif')
        
        UpdateGT(path+'/post-fire/Classification/Patchiness/Patches.tif',
                 patches,
                 path+'/post-fire/Classification/Patchiness/OBIA.tif')
        
        UpdateGT(path+'/post-fire/Classification/Patchiness/post-fire_vegetation.tif',
                 unburned_classes.filled(-999),
                 path+'/post-fire/Classification/Patchiness/OBIA.tif')
        
        #sieve the results
        sieve_cmd = 'gdal_sieve.py -st 400 -4 -of GTiff '+\
                    path+'/post-fire/Classification/Patchiness/Patches.tif '+\
                    path+'/post-fire/Classification/Patchiness/Patches_sieved.tif'
        subprocess.call(sieve_cmd, shell=True)
        
        sieve_cmd = 'gdal_sieve.py -st 400 -4 -of GTiff '+\
                    path+'/post-fire/Classification/Patchiness/post-fire_vegetation.tif '+\
                    path+'/post-fire/Classification/Patchiness/post-fire_vegetation_sieved.tif'
        subprocess.call(sieve_cmd, shell=True)
        
        crop_cmd = 'gdalwarp -of GTiff -cutline '+\
                    cropline+\
                    ' -cl burned_area_mask -crop_to_cutline -overwrite -dstnodata -999.0 '+\
                    path+'/post-fire/Classification/Patchiness/Patches_sieved.tif '+\
                    path+'/post-fire/Classification/Patchiness/Patches_cropped_sieved.tif'
        
        subprocess.call(crop_cmd, shell=True)
    
        crop_cmd = 'gdalwarp -of GTiff -cutline '+\
                    cropline+\
                    ' -cl burned_area_mask -crop_to_cutline -overwrite -dstnodata -999.0 '+\
                    path+'/post-fire/Classification/Patchiness/post-fire_vegetation_sieved.tif '+\
                    path+'/post-fire/Classification/Patchiness/post-fire_vegetation_cropped_sieved.tif'
        
        subprocess.call(crop_cmd, shell=True)
    #if not, then crop to a given mask
    except np.ma.MaskError:
        
        #if there is already a 'burned area' mask file, use this
        if os.path.exists(path+'/burned_area_mask.shp'):
            unburned_area = path+'/burned_area_mask.shp'
    #        elif os.path.exists(path+'/unburned_area_mask.shp'):
    #            unburned_area = path+'/unburned_area_mask.shp'
        #otherwise, copy across the existing file to the new directory and use that
        else:
            for i in [pre_path, post_path]:
                if os.path.exists('/'.join(i.split('/')[:-3])+'/Classification/Patchiness'):
                    if os.path.exists('/'.join(i.split('/')[:-3])+'/Classification/Patchiness/OBIA.tif'):
                        pass
                    else:
                        copyfile(i, '/'.join(i.split('/')[:-3])+'/Classification/Patchiness/OBIA.tif')
                else:
                    os.mkdir('/'.join(i.split('/')[:-3])+'/Classification/Patchiness')
                    copyfile(i, '/'.join(i.split('/')[:-3])+'/Classification/Patchiness/OBIA.tif')
            
            #if there is an overlap file, cut to that line
            if os.path.exists(path+'/Overlap.shp'):
                cropline = path+'/Overlap.shp'
            else:
                cropline = path+'/overlap.shp'
                
        for i in [pre_path, post_path]:
            
            if os.path.exists('/'.join(i.split('/')[:-3])+'/Classification/Patchiness'):
                pass
            else:
                os.mkdir('/'.join(i.split('/')[:-3])+'/Classification/Patchiness')
                        
            if 'unburned_area' in locals():
                cropline = unburned_area
            
            crop_cmd = 'gdalwarp -of GTiff -r bilinear -cutline '+cropline+\
                ' -crop_to_cutline '+i+' '+\
                '/'.join(i.split('/')[:-3])+'/Classification/Patchiness/OBIA.tif -overwrite -dstnodata -999 --config GDALWARP_IGNORE_BAD_CUTLINE YES'
                
            subprocess.call(crop_cmd, shell=True)
        
        #get extents of both files
        pre_GT = GetGeoTransform(path+'/pre-fire/Classification/Patchiness/OBIA.tif')
        post_GT = GetGeoTransform(path+'/post-fire/Classification/Patchiness/OBIA.tif')
        
        ulx = min(pre_GT['ulx'], post_GT['ulx'])
        uly = min(pre_GT['uly'], post_GT['uly'])
        lrx = max(pre_GT['lrx'], post_GT['lrx'])
        lry = max(pre_GT['lry'], post_GT['lry'])
        
        extent = '-te '+str(ulx)+' '+str(lry)+' '+str(lrx)+' '+str(uly)
        
        xres = min(pre_GT['xres'], post_GT['xres'])
        yres = max(pre_GT['yres'], post_GT['yres'])
        
        res = '-tr '+str(xres)+' '+str(yres)
        
        #set the resolutions & extents to the same for both classifiers
        for i in [path+'/pre-fire/Classification/Patchiness/OBIA.tif', path+'/post-fire/Classification/Patchiness/OBIA.tif']:         
            
            cmd = 'gdalwarp -of GTiff -r bilinear '+extent+' '+res+' -cutline '+cropline+' -crop_to_cutline '+\
                    i+' '+\
                    i.split('.')[0]+'_cropped.tif -overwrite -dstnodata -999 --config GDALWARP_IGNORE_BAD_CUTLINE YES'
                
            subprocess.call(cmd, shell=True)
         
        #read in matched classifiers
        pre_classes = gdal.Open(path+'/pre-fire/Classification/Patchiness/OBIA_cropped.tif').ReadAsArray()
        post_classes = gdal.Open(path+'/post-fire/Classification/Patchiness/OBIA_cropped.tif').ReadAsArray()
        
        #mask the areas that haven't burned
        unburned_mask = np.ma.masked_equal(np.round(post_classes), 1).mask
#        burned_mask = np.ma.masked_equal(np.round(post_classes), 2).mask
        unburned_classes = np.ma.masked_outside(np.ma.round(np.ma.masked_equal(np.ma.masked_array(pre_classes, mask = unburned_mask),-999)), 1.1, 4.1)
#        burned_vegetation = np.ma.masked_inside(np.ma.round(np.ma.masked_equal(np.ma.masked_array(pre_classes, mask = burned_mask),-999)), 1.1, 4.1)
        labeled_arr, nlabels = label(unburned_classes.mask)
        
        patches = np.zeros(labeled_arr.shape)
        patches[labeled_arr!=1] = 1

        
        UpdateGT(path+'/post-fire/Classification/Patchiness/Patches.tif',
                 patches,
                 path+'/post-fire/Classification/Patchiness/OBIA_cropped.tif')
        
        UpdateGT(path+'/post-fire/Classification/Patchiness/post-fire_vegetation.tif',
                 unburned_classes.filled(-999),
                 path+'/post-fire/Classification/Patchiness/OBIA.tif')
        
        #sieve the results
        sieve_cmd = 'gdal_sieve.py -st 400 -4 -of GTiff '+\
                    path+'/post-fire/Classification/Patchiness/Patches.tif '+\
                    path+'/post-fire/Classification/Patchiness/Patches_sieved.tif'
        subprocess.call(sieve_cmd, shell=True)
        
        sieve_cmd = 'gdal_sieve.py -st 400 -4 -of GTiff '+\
                    path+'/post-fire/Classification/Patchiness/post-fire_vegetation.tif '+\
                    path+'/post-fire/Classification/Patchiness/post-fire_vegetation_sieved.tif'
        subprocess.call(sieve_cmd, shell=True)
        
        crop_cmd = 'gdalwarp -of GTiff -r bilinear -cutline '+cropline+\
                ' -crop_to_cutline '+path+'/post-fire/Classification/Patchiness/Patches_sieved.tif'+' '+\
                path+'/post-fire/Classification/Patchiness/Patches_sieved_cropped.tif -overwrite -dstnodata -999 --config GDALWARP_IGNORE_BAD_CUTLINE YES'
        
        subprocess.call(crop_cmd, shell=True)
        
        crop_cmd = 'gdalwarp -of GTiff -cutline '+\
                    cropline+\
                    ' -cl burned_area_mask -crop_to_cutline -overwrite -dstnodata -999.0 '+\
                    path+'/post-fire/Classification/Patchiness/Patches_sieved.tif '+\
                    path+'/post-fire/Classification/Patchiness/Patches_cropped_sieved.tif'
        
        subprocess.call(crop_cmd, shell=True)
    
        crop_cmd = 'gdalwarp -of GTiff -cutline '+\
                    cropline+\
                    ' -cl burned_area_mask -crop_to_cutline -overwrite -dstnodata -999.0 '+\
                    path+'/post-fire/Classification/Patchiness/post-fire_vegetation_sieved.tif '+\
                    path+'/post-fire/Classification/Patchiness/post-fire_vegetation_cropped_sieved.tif'
        
        subprocess.call(crop_cmd, shell=True)
        
        
    #percentage calculation for pure patchiness
    try:
        img = gdal.Open(path+'/post-fire/Classification/Patchiness/Patches_cropped_sieved.tif')
    except RuntimeError:
        img = gdal.Open(path+'/post-fire/Classification/Patchiness/Patches_sieved_cropped.tif')
    im_array = img.ReadAsArray()
    
    vals, counts = np.unique(im_array, return_counts=True)
    perc_unburned = 100*counts[-1]/np.sum(counts[1:])
    
    
    #proportions of vegetation types surviving
    try:
        img = gdal.Open(path+'/post-fire/Classification/Patchiness/post-fire_vegetation_cropped_sieved.tif')
    except RuntimeError:
        img = gdal.Open(path+'/post-fire/Classification/Patchiness/post-fire_vegetation_sieved.tif')
    im_array = img.ReadAsArray()
    
    vals, counts = np.unique(im_array, return_counts=True)
    total_veg = np.sum(counts[1:])
    
    pre_vals, pre_counts = np.unique(np.round(pre_classes), return_counts=True)

    percentage_veg = pd.DataFrame({'Grass': [100*counts[2]/total_veg, 100*counts[2]/pre_counts[4]],
                                    'Foliage': [100*counts[1]/total_veg, 100*counts[1]/pre_counts[3]],
                                    'Woody material': [100*counts[3]/total_veg, 100*counts[3]/pre_counts[5]]},
                                    index = ['Percentage of surviving vegetation','Percentage of class which survived'])

    
    results = pd.concat([percentage_veg, pd.DataFrame([perc_unburned], index = [0], columns = ['Total unburned'])], axis = 1)
    print('Class proportions within surviving vegetation')
    print('\n',results)
    return results


def Tree_count(path, pre_fire = True, pixel_thresholds = [200,1000,3000], DEM_threshold = 2, sieve_num = 100, sieve_num_foliage = 100):
    """
    Algorithm to determine whether objects designated as 'foliage' are trees or shrubs
    """
    pre_fire_classifier = glob.glob(path+'/pre-fire/Classification/OBIA/*corrected.tif')[0]
    pre_fire_features = glob.glob(path+'/pre-fire/indices/*total*.tif')[0]
    
    if not pre_fire:
        post_fire_classifier = glob.glob(path+'/post-fire/Classification/OBIA/*corrected.tif')[0]
        post_fire_features = glob.glob(path+'/post-fire/indices/*total*.tif')[0]
    
    #classifier in array form
    if pre_fire == True:
        classifier = gdal.Open(pre_fire_classifier).ReadAsArray()
        DEM = (gdal.Open(pre_fire_features).ReadAsArray())[2]
        if not os.path.exists(path+'/pre-fire/Classification/Trees'):
            os.mkdir(path+'/pre-fire/Classification/Trees')
    else:
        classifier = gdal.Open(post_fire_classifier).ReadAsArray()
        DEM = (gdal.Open(post_fire_features).ReadAsArray())[2]
        if not os.path.exists(path+'/post-fire/Classification/Trees'):
            os.mkdir(path+'/post-fire/Classification/Trees')
            
    #mask all elements other than foliage and turn into binary array
    mask = (np.ma.masked_equal(classifier, 2).mask).astype(int)
    mask_DEM = (np.ma.masked_greater(DEM, 0.2).mask).astype(int)
    mask_2m = (np.ma.masked_greater_equal(DEM,2).mask).astype(int)
    # inds_2m = np.where(mask_2m == 1)
    

    #partition the mask into only connected pixels
    labeled_class, nlabels_class = label(mask)
    labeled_DEM, nlabels_DEM = label(mask_DEM)
    
    labeled_class[labeled_class!=0] = 1
    
    UpdateGT(path+'/pre-fire/Classification/Trees/foliage_class.tif', labeled_class, pre_fire_classifier)
    
    sieve_cmd = 'gdal_sieve.py -st '+\
                    str(sieve_num_foliage)+\
                        ' -4 -of GTiff '+\
                            path+'/pre-fire/Classification/Trees/foliage_class.tif '+\
                                path+'/pre-fire/Classification/Trees/foliage_class_sieved.tif'
                                
    subprocess.call(sieve_cmd, shell=True)
    
    foliage_arr = gdal.Open(path+'/pre-fire/Classification/Trees/foliage_class_sieved.tif').ReadAsArray()
    
    labeled_DEM[labeled_DEM!=0] = 1
    # foliage_arr[labeled_DEM!=0] = 1
    
    
    
    #anything over 2m is classed as a tree
    new_labeled_DEM = labeled_DEM + mask_2m
    
    #if there is no foliage detected, then remove the object
    new_labeled_DEM[foliage_arr==0] = 0
    
    #then add in any extra foliage detections as shrubs, making sure to not double count
    foliage_arr[new_labeled_DEM!=0] = 0
    output = new_labeled_DEM + foliage_arr
    
    
    # inds_labeled = np.where(labeled_DEM == 10)
    
    
    UpdateGT(path+'/pre-fire/Classification/Trees/trees_DEM.tif', output, pre_fire_classifier)
    
    sieve_cmd = 'gdal_sieve.py -st '+\
                    str(sieve_num)+\
                        ' -4 -of GTiff '+\
                            path+'/pre-fire/Classification/Trees/trees_DEM.tif '+\
                                path+'/pre-fire/Classification/Trees/trees_DEM_sieved.tif'
                                
    subprocess.call(sieve_cmd, shell=True)
    
    
    
    

    # objects = []
    # mean_height = []
    # num_pixels = np.zeros(nlabels)
    # foliage_category = np.zeros(nlabels)
    # height = np.zeros(classifier.shape)
    # pixel_size = np.zeros(classifier.shape)
    # trees_shrubs = np.zeros(classifier.shape)
    # for i in tqdm(range(1, nlabels+1)):
    #     objects.append(((np.where(labeled_arr == i))))
    #     mean_height.append(np.mean(DEM[objects[i-1]]))
    
        
    #     num_pixels[i-1] = len(objects[i-1][0])

    #     if pixel_thresholds[0] < num_pixels[i-1] <= pixel_thresholds[1]:
    #         foliage_category[i-1] = 10
    #     elif pixel_thresholds[1] < num_pixels[i-1] <= pixel_thresholds[2]:
    #         foliage_category[i-1] = 20
    #     elif pixel_thresholds[2] < num_pixels[i-1]:
    #         foliage_category[i-1] = 10
    #     elif mean_height[i-1] > DEM_threshold:
    #         foliage_category[i-1] = 20
            
    #     trees_shrubs[objects[i-1]] = foliage_category[i-1]
    #     height[objects[i-1]] = mean_height[i-1]
    #     pixel_size[objects[i-1]] = num_pixels[i-1]
        
    # if pre_fire == True:
    #     UpdateGT('/'.join(pre_fire_classifier.split('/')[:-1])+'/trees_shrubs.tif',
    #                            trees_shrubs, 
    #                            pre_fire_classifier)
    #     UpdateGT('/'.join(pre_fire_classifier.split('/')[:-1])+'/object_height.tif',
    #                            height, 
    #                            pre_fire_classifier)
    #     UpdateGT('/'.join(pre_fire_classifier.split('/')[:-1])+'/object_size.tif',
    #                            pixel_size, 
    #                            pre_fire_classifier)
    # else:
    #     UpdateGT('/'.join(post_fire_classifier.split('/')[:-1])+'/trees_shrubs.tif',
    #                            trees_shrubs, 
    #                            post_fire_classifier)
    #     UpdateGT('/'.join(post_fire_classifier.split('/')[:-1])+'/object_height.tif',
    #                            height, 
    #                            post_fire_classifier)
    #     UpdateGT('/'.join(post_fire_classifier.split('/')[:-1])+'/object_size.tif',
    #                            pixel_size, 
    #                            post_fire_classifier)
        
        
    # return num_pixels, mean_height, foliage_category

class Polygon():

    def BoundingLine(lon_start, lon_end, lat_start, lat_end, dist = 5):
        """
        Returns a bounding box a given distance surrounding an input line
        """
    
        #radius of the earth (in m)
        R = 6378.137 * 1e3

        #distance from line to line in degrees
        dist_deg = np.rad2deg(dist/(2*R))
    
        #find the equation of the line
        m = (lat_start - lat_end)/(lon_start - lon_end)
        
        #find the equation of the perpendicular line passing through the midpoint
        m_perp = -1/m
        if m==0:
            m_perp = 1
        b_end = lat_end - lon_end * m_perp
        b_start = lat_start - lon_start * m_perp
        
        #find the longitudes
        x_bounds = np.array([lon_start + dist_deg/np.sqrt(1+m_perp**2),
                             lon_start - dist_deg/np.sqrt(1+m_perp**2),
                             lon_end - dist_deg/np.sqrt(1+m_perp**2),
                             lon_end + dist_deg/np.sqrt(1+m_perp**2)])
        
        y_bounds = np.array([x_bounds[0]*m_perp + b_start,
                             x_bounds[1]*m_perp + b_start,
                             x_bounds[2]*m_perp + b_end,
                             x_bounds[3]*m_perp + b_end])
    
        print(x_bounds)
        print(y_bounds)
        return x_bounds, y_bounds

    def CreatePolygon(lon_start, lon_end, lat_start, lat_end, dist = 5, BoundingLine = BoundingLine):
        """
        Create a polygon object from a given boundary box
        """
        
        #retrieve bounding coordinates
        coords = np.array(BoundingLine(lon_start, lon_end, lat_start, lat_end, dist))
        
        #define geometry
        ring = ogr.Geometry(ogr.wkbLinearRing)
        
        #add points to the polygon
        for c in range(len(coords[0])):
            ring.AddPoint(coords[0][c], coords[1][c])
        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        
        return poly.ExportToWkt(), coords
        
    def WriteSHP(poly, out_shp):
        """
        Writes a shapefile from a given polygon object
        """
    
        #convert to a shapefile with OGR    
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.CreateDataSource(out_shp)
        layer = ds.CreateLayer('', None, ogr.wkbPolygon)
        
        # Add one attribute
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()
    
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', 123)
    
        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkt(poly)
        feat.SetGeometry(geom)
    
        layer.CreateFeature(feat)
        feat = geom = None  # destroy these
    
        # Save and close everything
        ds = layer = feat = geom = None
        
        cmd = 'ogr2ogr -a_srs EPSG:4326 '+out_shp.split('.')[0]+'_updated.shp '+out_shp
        
        subprocess.run(cmd.split())

class Fuel_Load():
    
    def __init__(self, xl_filename):
        
        xls = xlrd.open_workbook(xl_filename, on_demand=True)
        
        self.filename = xl_filename
        self.sheet_names = xls.sheet_names()
 
    def by_type(self, name):
        """
        Returns a DataFrame of the fuel load category specified 
        """
        
        return pd.read_excel(self.filename, sheet_name = name)
    
    
    def by_transect(self, transects, col = 'Transect'):
        """
        Returns a dictionary of DataFrames of the fuel load taken from a given transect(s)
        """
        fl_data = {}
        per_transect = {}
        indx_dict = {}
        #loop over the sheet data 
        for sheet in self.sheet_names:
            
            #add the full dataset to a dictionary and use it to extract the desired transect data
            fl_data[sheet] = pd.read_excel(self.filename, sheet_name = sheet)
            per_transect[sheet] = pd.DataFrame(columns = fl_data[sheet].columns)
            
            #test for existance of transect data
            try:
                x = fl_data[sheet][col]
                indx_dict[sheet] = []
                #add the transect data
                for num in transects:
                    indx = np.where(fl_data[sheet][col] == num)[0]
                    indx_dict[sheet].append(indx)
                    
                    #skip sections for which there is no data
                    if len(indx)>1:
                        per_transect[sheet] = per_transect[sheet].append(fl_data[sheet].loc[indx[0]:indx[-1]])
                        
                    #allow for categories with a single data entry per transect 
                    else:
                        try:
                            per_transect[sheet] = per_transect[sheet].append(fl_data[sheet].loc[indx[0]])
                        except IndexError:
                            print('\nNo data found for transect '+str(num)+' in sheet '+sheet)
            except KeyError:
                print('\nSheet named "'+sheet+'" has no "'+col+'" data')
            
            #remove empty sheets
            if np.size(per_transect[sheet]) == 0:
                del per_transect[sheet]
                
        return per_transect, indx_dict
    
    def TransectPolygon(self, transect_start, transect_end, transect_num, path, dist = 5):
        """
        Creates a 10x50m polygon around the transect line
        """
        
        #check if a fuel load folder exists
        if os.path.exists(path+'/fuel_load'):
            pass
        else:
            os.mkdir(path+'/fuel_load')
            
        #lons/lats
        lon_start, lat_start = transect_start
        lon_end, lat_end = transect_end
        
        #create a polgon object
        poly, coords = Polygon.CreatePolygon(lon_start, lon_end, lat_start, lat_end, dist)
        
        #write to shapefile
        Polygon.WriteSHP(poly, path+'/fuel_load/transect_bound_'+str(transect_num)+'.shp')
        
        return path+'/fuel_load/transect_bound_'+str(transect_num)+'_updated.shp', coords
        
    
    def crop_stack(self, path_to_img, transect_shapefile, outfilename, overwrite = True):
        """
        Extends the transect shapefile to the extents of the image and then crops the stack to the transect
        """
        
        if overwrite == True:
            ovr = '-overwrite'
        else:
            ovr = ''
            
        cmd = 'gdalwarp -of GTiff -cutline '+transect_shapefile+' -crop_to_cutline '+path_to_img+' '+\
        '/'.join(transect_shapefile.split('/')[:-1])+'/'+outfilename+\
                          ' '+ovr+' --config GDALWARP_IGNORE_BAD_CUTLINE YES -dstnodata -999'
     
        subprocess.call(cmd, shell = True)
        
        return '/'.join(transect_shapefile.split('/')[:-1])+'/'+outfilename
    
class fuel():
    
    def read_pickle(self, path_to_pickles, subfolder = 'fuel_load'):
        """
        Parameters
        ----------
        path_to_pickles : path to directory containing pickle files

        Returns
        -------
        A dictionary object containing entries for each file
        """
        #find all pickle filepaths in search directory
        pkl_files = []
        for root, dirs, files in os.walk(path_to_pickles):
            if subfolder in dirs:
                pkl_files.append(glob.glob(root+'/'+subfolder+'/*.pickle'))
        
        pkl_files = list(itertools.chain.from_iterable(pkl_files))
        
        #read in pickle files to a dict
        print('Reading in transect data...\n')
        fl_data = {}
        for pkl_path in pkl_files:
        
            filename = pkl_path.split('/')[-1].split('.')[0]
            location = pkl_path.split('/')[-3]
            # print(filename+'_'+location)
            with open(pkl_path, 'rb') as F:
                
                fl_data[filename+'_'+location] = pickle.load(F)
                
        return fl_data
    
    def process_fuel_data(self,
                          fl_data, 
                          time_since_fire_path = '/Volumes/LaCie/Data/Sat_data/MODIS/MCD64A1.006/Time since fire',
                          met_data_path = '/Volumes/LaCie/Data/Met_data/ECMWF',
                          year = None, 
                          tsf_region = 's_a',
                          region = 'Africa', 
                          Read_NCDF = Read_NCDF):
        """
        Processes the fuel load data dictionary returned by read_pickle function

        Returns
        -------
        DataFrames of basic features and the corresponding target values for each transect

        """
        
        #target arrays
        features_av = []
        pf_features_av = []
        ba_features_av = []
        time_since_fire = []
        grass = []
        litter = []
        coarse = []
        heavy = []
        shrubs = []
        trees = []
        post_fire_litter = []
        post_fire_coarse = []
        post_fire_heavy = []
        tree_count = []
        shrub_count = []
        class_proportions = []
        pf_class_proportions = []
        transect_names = []
        precipitation = []
        mean_precipitation = []
        mean_temp = []
        mean_ssr = []
        mean_soil_moisture = []
        map_tsf = []
        mean_wind_speed = []
        mean_LAI_hv = []
        mean_LAI_lv = []
        mean_evaporation = []
        mean_transpiration = []
        sum_evaporation = []
        sum_transpiration = []
        mean_LAI_HV = []
        mean_LAI_LV = []
        patchiness = []
        
        
        inst_soil_moisture = []
        inst_temp = []
        inst_wind = []
        inst_LAI_HV = []
        inst_LAI_LV = []
        
        
        
        
        # classes = ['bare_soil','foliage','grass','woody_material']
        fl_classes = ['grass', 'litter', 'coarse', 'heavy','shrubs','trees']
        
        print('Reading in meteo data...')
        #read in meteo data
        met_data = Read_NCDF()
        print('   ...done!\n')
        
        #read in time since fire map
        tsf = gdal.Open(time_since_fire_path+'/'+tsf_region+'_time_since_fire_reprj.tif').ReadAsArray()
        with open(time_since_fire_path+'/'+tsf_region+'_lons.pickle', 'rb') as F:
            tsf_lons = pickle.load(F)
            
        with open(time_since_fire_path+'/'+tsf_region+'_lats.pickle', 'rb') as F:
            tsf_lats = pickle.load(F)
        #organise data for training and testing
        print('\nCreating training data...\n')
        for key in fl_data.keys():
            
            print(key)
            transect_names.append(key)
            
            try:
                #post fire burned proportion
                pf_classifier = np.ma.masked_equal(np.ma.masked_equal(fl_data[key]['Post-fire Classes'], -999), 0)
                ba_mask = np.ma.masked_equal(np.ma.masked_equal(np.ma.masked_equal(fl_data[key]['Post-fire Classes'], -999), 0), 2).mask
                pf_vals, pf_counts = np.unique(pf_classifier, return_counts = True)
                
                pf_counts = np.ma.masked_array(pf_counts, mask = pf_vals.mask)
                
                pf_sum_pixels = np.ma.sum(pf_counts)
                pf_props = [x/pf_sum_pixels for x in pf_counts.compressed()]
                
                pf_class_proportions.append(pf_props)
            except KeyError:
                pf_class_proportions.append([-999,-999])
                ba_mask = -999
            
            #average the features spatially, taking care to mask any areas with no data. Allowed masks are 0 across all bands, or -999
            try:
                mask = np.tile(np.ma.masked_equal(np.sum(fl_data[key]['Pre-fire Features'], axis = 0), 0).mask, 
                               (len(fl_data[key]['Pre-fire Features']),1,1))
                
                features_av.append(np.ma.mean(np.ma.mean(np.ma.masked_array(fl_data[key]['Pre-fire Features'],mask = mask), axis = 1), axis = 1)) 
                try:
                    pf_mask = np.tile(np.ma.masked_equal(np.sum(fl_data[key]['Post-fire Features'], axis = 0), 0).mask, 
                                    (len(fl_data[key]['Post-fire Features']),1,1))
                  
                    pf_features_av.append(np.ma.mean(np.ma.mean(np.ma.masked_array(fl_data[key]['Post-fire Features'], mask = pf_mask), axis = 1), axis = 1)) 
                except KeyError:
                    pf_features_av.append([-999,-999,-999,-999,-999,-999,-999,-999])
                    
                try:
                    pf_mask_2 = np.tile(ba_mask, 
                                   (len(fl_data[key]['Post-fire Features']),1,1))
                    # if type(ba_mask) != int:
                    #     if pf_mask_2.shape[1] != ba_mask.shape[0]:
                    #         ba_mask = np.vstack((ba_mask, np.repeat(True, len(ba_mask[1]))))
                    #     if pf_mask_2.shape[2] != ba_mask.shape[1]:
                    #         ba_mask = np.append(ba_mask, np.repeat(True, len(ba_mask)).reshape(-1,len(ba_mask)), axis = 1)
                    ba_features_av.append(np.ma.mean(np.ma.mean(np.ma.masked_array(fl_data[key]['Post-fire Features'], mask = pf_mask_2), axis = 1), axis = 1))
                except KeyError:
                    ba_features_av.append([-999,-999,-999,-999,-999,-999,-999,-999])
            except np.ma.MaskError:
                mask = np.tile(np.ma.masked_equal(fl_data[key]['Pre-fire Features'][0], -999).mask,
                               (len(fl_data[key]['Pre-fire Features']),1,1))
               
                features_av.append(np.ma.mean(np.ma.mean(np.ma.masked_array(fl_data[key]['Pre-fire Features'],mask = mask), axis = 1), axis = 1))
                try:
                    pf_mask = np.tile(np.ma.masked_equal(fl_data[key]['Post-fire Features'][0], -999).mask,
                                    (len(fl_data[key]['Post-fire Features']),1,1))
                    pf_features_av.append(np.ma.mean(np.ma.mean(np.ma.masked_array(fl_data[key]['Post-fire Features'],mask = pf_mask), axis = 1), axis = 1))
                except KeyError:
                    pf_features_av.append([-999,-999,-999,-999,-999,-999,-999,-999])
                try:
                    pf_mask_2 = np.tile(ba_mask, 
                                   (len(fl_data[key]['Post-fire Features']),1,1))
                    if type(ba_mask) != int:
                        if pf_mask_2.shape[1] < pf_mask.shape[1]:
                            # pf_mask_2 = np.append(pf_mask_2, np.repeat(np.repeat(True, len(pf_mask_2[0,0])), 8).reshape(8, -1, len(pf_mask_2[0,0])), axis = 1)
                            pf_mask_2 = np.append(pf_mask_2, np.repeat(np.repeat(True, len(pf_mask_2[0,0])), 8).reshape(8, -1, len(pf_mask_2[0,0])), axis = 1)
                        elif pf_mask_2.shape[1] > pf_mask.shape[1]:
                            pf_mask_2 =  np.delete(pf_mask_2, [8,-1], axis = 1)
                        if pf_mask_2.shape[2] < pf_mask.shape[2]:
                            pf_mask_2 = np.append(pf_mask_2, np.repeat(np.repeat(True, len(pf_mask_2[0])), 8).reshape(8, len(pf_mask_2[0]), -1), axis = 2)
                        elif pf_mask_2.shape[2] > pf_mask.shape[2]:
                            pf_mask_2 = np.delete(pf_mask_2, [8,-1], axis = 2)
                    ba_features_av.append(np.ma.mean(np.ma.mean(np.ma.masked_equal(np.ma.masked_array(fl_data[key]['Post-fire Features'], mask = pf_mask_2), -999), axis = 1), axis = 1))
                except KeyError:
                    ba_features_av.append([-999,-999,-999,-999,-999,-999,-999,-999])
            #tree and shrub count
            tree_count.append(fl_data[key]['Classifier Tree Count'])
            shrub_count.append(fl_data[key]['Classifier Shrub Count'])
            
            #time since fire
            time_since_fire.append(fl_data[key]['Time since fire'])
             
            #average values for fine fuels
            grass.append(np.mean(fl_data[key]['Fine Fuels'].Grass.values))
            litter.append(np.mean(fl_data[key]['Fine Fuels'].Litter.values))
            coarse.append(np.ma.mean(np.ma.masked_invalid(fl_data[key]['Fine Fuels'].Coarse.values)))
            
            try:
                # post_fire_litter.append(np.average(fl_data[key]['Post Fire'].PFL.values+fl_data[key]['Post Fire'].Ash.values, weights = fl_data[key]['Post Fire']['%1x1'].values))
                post_fire_litter.append(np.mean(fl_data[key]['Post Fire'].PFL.values+fl_data[key]['Post Fire'].Ash.values[np.where(fl_data[key]['Post Fire']['%1x1'].values == 100)][0]))
                try:
                    post_fire_coarse.append(np.average(np.array(fl_data[key]['Post Fire'].Coarse.values, dtype = int), weights = fl_data[key]['Post Fire']['Patchiness'].values)/100)
                except AttributeError:
                    post_fire_coarse.append(np.average(np.array(fl_data[key]['Post Fire']['Coarse fuels burnt'].values, dtype = int), weights = fl_data[key]['Post Fire']['Patchiness'].values)/100)
            except (KeyError, ZeroDivisionError, IndexError) as e:
                try:
                    # post_fire_litter.append(np.average(fl_data[key]['Post Fire']['Grass & litter'].values, weights = fl_data[key]['Post Fire']['Proportion of quadrant burnt'].values))
                    post_fire_litter.append(np.average(fl_data[key]['Post Fire']['Grass & litter'].values[np.where(fl_data[key]['Post Fire']['Proportion of quadrant burnt'].values == 100)][0]))
                    post_fire_coarse.append(np.average(fl_data[key]['Post Fire']['Coarse fuels burnt'].values, weights = fl_data[key]['Post Fire']['Patchiness'].values)/100)
                except (KeyError, ZeroDivisionError, IndexError) as e:
                    post_fire_coarse.append(-999)
                    post_fire_litter.append(-999)
                    
            try:        
                patchiness.append(np.mean(fl_data[key]['Post Fire'].Patchiness.values)/100)
            except KeyError:
                patchiness.append(-999)
            #total count of heavy fuels, shrubs and trees
            try:
                heavy.append(np.sum(fl_data[key]['Heavy']['Volume(m3)'].values))
            except KeyError:
                try:
                    heavy.append(np.sum(fl_data[key]['Heavy']['Volume'].values))
                except KeyError:
                    heavy.append(0)
            try:
                shrubs.append(np.sum(fl_data[key]['Shrub Count'].Count.values))
            except KeyError:
                shrubs.append(0)
            try:
                trees.append(len(fl_data[key]['Trees']))
            except KeyError:
                trees.append(0)
            
            try:
                post_fire_heavy.append(np.mean([x for x in fl_data[key]['Post Fire']['Heavy'].values if type(x) == int])/100)
            except KeyError:
                try:
                    post_fire_heavy.append(np.mean([x for x in fl_data[key]['Post Fire']['Heavy fuels'].values if type(x) == int])/100)
                except KeyError:
                    post_fire_heavy.append(-999)
            #proportion of each class in the transect
            #mask the no-data values in the classifier, additionally excluding shadows
            classifier = np.ma.masked_equal(np.ma.masked_equal(np.ma.masked_equal(fl_data[key]['Pre-fire Classes'], -999), 5),0)
            
            #get the number of occurences of each class
            vals, counts = np.unique(classifier, return_counts = True)
            
            #mask the counts array to exclude shadows & no_data
            counts = np.ma.masked_array(counts, mask = vals.mask)
            
            #calculate the proportions of each class
            sum_pixels = np.ma.sum(counts)
            props = [x/sum_pixels for x in counts.compressed()]
            if len(props) < 4:
                props.append(0)
            class_proportions.append(props)
            
            
            
            
            #find the closest coordinates in the meteo dataset
            closest_lat_ind = np.where(abs(met_data['Latitude']-fl_data[key]['coords'][0][1]) == 
                                    min(abs(met_data['Latitude']-fl_data[key]['coords'][0][1])))[0][0]
            
            closest_lon_ind = np.where(abs((met_data['Longitude'])-fl_data[key]['coords'][0][0]) == 
                                    min(abs((met_data['Longitude'])-fl_data[key]['coords'][0][0])))[0][0]
            
            #and the same for the time_since_fire set
            closest_lat_ind_tsf = np.where(abs(tsf_lats-fl_data[key]['coords'][0][1]) == 
                                    min(abs(tsf_lats-fl_data[key]['coords'][0][1])))[0][0]
            
            closest_lon_ind_tsf = np.where(abs((tsf_lons)-fl_data[key]['coords'][0][0]) == 
                                    min(abs((tsf_lons)-fl_data[key]['coords'][0][0])))[0][0]            
            #how much precipitation to total up depends on the time since fire
            date_of_fire = dtm.datetime(pd.DatetimeIndex([fl_data[key]['Date'][0]]).year[0],
                                        month = pd.DatetimeIndex([fl_data[key]['Date'][0]]).month[0],
                                        day = pd.DatetimeIndex([fl_data[key]['Date'][0]]).day[0])
            
            tsf_point = tsf[:,closest_lat_ind_tsf, closest_lon_ind_tsf]
            tsf_date = dtm.datetime(year = tsf_point[0], month = 1, day = 1) + dtm.timedelta(days = float(tsf_point[1]))
            tsf_delta = (date_of_fire - tsf_date)
            
            map_tsf.append(tsf_delta.days)
            
            closest_previous_date_ind = np.where(np.array([(x - date_of_fire).days for x in met_data['Time']]) < 0)[0][-1]
            start_date_ind = int(closest_previous_date_ind - math.ceil(12*(tsf_delta.days/365)))
            if start_date_ind <0:
                start_date_ind=0
            
            #sort pre-fire meteorological features (sums/averages)
            precipitation.append(np.ma.sum(met_data['Total precipitation'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_precipitation.append(np.ma.mean(met_data['Total precipitation'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            
            mean_ssr.append(np.ma.mean(met_data['Solar radiation'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_temp.append(np.mean(met_data['2m Temperature'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_soil_moisture.append(np.ma.mean(met_data['Soil water 1'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_wind_speed.append(np.ma.mean(met_data['Wind speed'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_LAI_hv.append(np.ma.mean(met_data['LAI HV'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_LAI_lv.append(np.ma.mean(met_data['LAI LV'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            
            sum_evaporation.append(np.ma.sum(met_data['Evaporation'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_evaporation.append(np.ma.mean(met_data['Evaporation'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            
            sum_transpiration.append(np.ma.sum(met_data['Evaporation from vegetation'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_transpiration.append(np.ma.mean(met_data['Evaporation from vegetation'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            
            mean_LAI_HV.append(np.ma.mean(met_data['LAI HV'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            mean_LAI_LV.append(np.ma.mean(met_data['LAI LV'][start_date_ind:closest_previous_date_ind,closest_lat_ind, closest_lon_ind]))
            
            #instantaneous meteorlogical variables
            inst_temp.append(met_data['2m Temperature'][closest_previous_date_ind,closest_lat_ind, closest_lon_ind])
            inst_wind.append(met_data['Wind speed'][closest_previous_date_ind,closest_lat_ind, closest_lon_ind])
            inst_soil_moisture.append(met_data['Soil water 1'][closest_previous_date_ind,closest_lat_ind, closest_lon_ind])
            inst_LAI_HV.append(met_data['LAI HV'][closest_previous_date_ind,closest_lat_ind, closest_lon_ind])
            inst_LAI_LV.append(met_data['LAI LV'][closest_previous_date_ind,closest_lat_ind, closest_lon_ind])
            
            
        #correct coarse
        coarse = [x/5 for x in coarse]
        
        #columns = ['NDVI','BAI','DEM','RedEdge','NIR','Red','Green','Blue','Time since fire','Prop_bare_soil','Prop_grass','Prop_foliage','Prop_wood']
        columns = ['NDVI',
                   'BAI',
                   'DEM',
                   'RedEdge',
                   'NIR',
                   'Red',
                   'Green',
                   'Blue',
                   'Time_since_fire',
                   'Prop_bare_soil',
                   'Prop_foliage',
                   'Prop_grass',
                   'Prop_wood', 
                   'num_trees', 
                   'num_shrubs']
        
        pf_columns = ['NDVI',
                   'BAI',
                   'DEM',
                   'RedEdge',
                   'NIR',
                   'Red',
                   'Green',
                   'Blue',
                   'NDVI_BA',
                   'BAI_BA',
                   'DEM_BA',
                   'RedEdge_BA',
                   'NIR_BA',
                   'Red_BA',
                   'Green_BA',
                   'Blue_BA',
                   'Prop_unburned',
                   'Prop_burned']
            
        
        features = []
        target = []
        pf_features = []
        for i in range(len(features_av)):
            features.append(np.append(np.append(np.append(np.append(features_av[i].compressed(), \
                                                                    time_since_fire[i]), class_proportions[i]), tree_count[i]), shrub_count[i]))
            target.append([grass[i], litter[i], coarse[i], heavy[i], shrubs[i], trees[i]])
        
        
            pf_features.append(np.append(np.append(pf_features_av[i], ba_features_av[i]), pf_class_proportions[i]))
            
            
        features = pd.DataFrame(features, columns = columns)#, index = transect_names)
        target = pd.DataFrame(target, columns = fl_classes)
        
        
        
        pf_features = pd.DataFrame(pf_features, columns = pf_columns)#, index = transect_names)
        pf_target = pd.DataFrame()
        
        target['Transects'] = transect_names
        features['Transects'] = transect_names
        
        features = features.set_index('Transects')
        target = target.set_index('Transects')
        
        #add additional feature columns
        features['foliage_detections'] = features['num_shrubs']+features['num_trees']
        features['sum_precipitation'] = precipitation
        features['adj_time_since_fire'] = np.array(map_tsf) * np.array(precipitation)
        features['mean_annual_precipitation'] = mean_precipitation
        features['mean_ssr'] = mean_ssr
        features['mean_temp'] = mean_temp
        features['mean_soil_moisture'] = mean_soil_moisture
        features['map_tsf'] = map_tsf
        features['mean_wind'] = mean_wind_speed
        features['LAI_hv'] = mean_LAI_hv
        features['LAI_lv'] = mean_LAI_lv
        features['mean_evaporation'] = mean_evaporation
        features['mean_transpiration'] = mean_transpiration
        features['sum_evaporation'] = sum_evaporation
        features['sum_transpiration'] = sum_transpiration
        features['LAI_HV'] = mean_LAI_HV
        features['LAI_LV'] = mean_LAI_LV
        
        pf_target['PFL'] = post_fire_litter
        pf_target['CC_litter'] = 1 - ((pf_target['PFL'].values)/(target['grass']+target['litter']).values)
        pf_target['CC_litter'][pf_target['PFL'].values == -999] = -999
        pf_target['CC_coarse'] = post_fire_coarse
        pf_target['Patchiness'] = patchiness
        pf_target['CC_heavy'] = post_fire_heavy
        
        pf_features['wind_speed'] = inst_wind
        pf_features['temp'] = inst_temp
        pf_features['soil_moisture'] = inst_soil_moisture
        pf_features['LAI_HV'] = inst_LAI_HV
        pf_features['LAI_LV'] = inst_LAI_LV
                
        return features, target, pf_features, pf_target
    
    
    def feature_comparison_plot(veg_class, feature_name, target, features, savepath = '/Volumes/LaCie/Data_Analysis/Fuel_Load'):
    
        plt.figure(figsize = (12,12))
        plt.ylabel(veg_class, fontsize = 20)
        plt.xlabel(feature_name, fontsize = 20)
        plt.title('Relationship between '+veg_class+' and '+feature_name, fontsize=24)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        
        plt.scatter(features[feature_name], target[veg_class])
        
        if not os.path.exists(savepath+'/class_sensitivity/'+veg_class):
            os.mkdir(savepath+'/class_sensitivity/'+veg_class)
        
        plt.savefig(savepath+'/class_sensitivity/'+veg_class+'/Sensitivity_'+feature_name+'.png')
        
        
    def split_train_test(features, target, split_val, random_sample = True, PF = False):
        """
        Split the features and target dataframes into train and test data
        """
        
        if random_sample:
            #split parameters
            test_ind = np.random.choice(len(features),split_val, replace=False)
            print('Test plots: ')
            print(features['Transects'].values[test_ind])
    
        #use a fixed sub-sample of plots for testing
        else:
            #split parameters
            if not PF:
                test_ind = np.array([26, 32, 35, 21, 11, 29, 41])
            else:
                test_ind = np.array([1, 29, 26, 16, 20, 6, 23])
            print('Test plots: ')
            print(features['Transects'].values[test_ind])
            
        train_ind = list(np.arange(0,len(features)))
        for x in test_ind:
            train_ind.remove(x)
            
        train_ind = np.array(train_ind)
        
        #split into training and testing
        target_test = target.iloc[test_ind]
        features_test = features.iloc[test_ind]
        
        target = target.copy().iloc[train_ind]
        features = features.copy().iloc[train_ind]
        
        return features, features_test, target, target_test
    
    
    def VarPart(fuel_class, drone_features, met_features, features, target, model):
        """
        Parameters
        ----------
        fuel_class (string) : fuel class in the model
        drone_features (list): list of features extracted from a drone
        met_features (list) : list of features extracted from meteorological data
        features (DF) : DataFrame of features.
        target (DF): DataFrame of target values
        model : OLS model from statsmodels api

        Returns
        -------
        VP : DataFrame of R2s and SS values

        """

        #dataframes with either drone or met features
        drone_df = features[drone_features]
        met_df = features[met_features]
        
        #regress against drone/met
        drone_model = sm.OLS(target, drone_df).fit()
        met_model = sm.OLS(target, met_df).fit()
        
        #correlation coefficients squares
        drone_corr2 = np.corrcoef(target, drone_model.predict(drone_df))[0][1]**2
        met_corr2 = np.corrcoef(target, met_model.predict(met_df))[0][1]**2
        corr2 = np.corrcoef(target, model.predict(features))[0][1]**2
        
        #sum of squares
        
        SS = np.var(target)*(len(target)-1)
        
        #calculate individual contributions
        
        #rsquareds for differnet parts of the venn diagram
        a_b_c = abs(np.array([model.rsquared, model.rsquared_adj, corr2]))
        
        a_b = abs(np.array([drone_model.rsquared, drone_model.rsquared_adj, drone_corr2]))
        b_c = abs(np.array([met_model.rsquared, met_model.rsquared_adj, met_corr2]))
        
        #residuals
        d = (1 - a_b_c)
        SS_d = SS*d[0]
        
        #drone contribution
        a = (a_b_c - b_c)
        SS_a = SS*a[0]
        
        #met contribution
        c = (a_b_c - a_b)
        SS_c = SS*c[0]
        
        #overlap
        b = (a_b + b_c - a_b_c)
        SS_b = SS*b[0]
        
        #oranise data into output dataframe
        data = np.array([a,b,c,d, a+b+d+c])
        VP = pd.DataFrame(data, columns = ['R^2', 'R^2_adj', 'corr^2'], index = ['Drone','Both','Met','residual','total'])
        VP['SS'] = np.array([SS_a, SS_b, SS_c, SS_d, SS])
        
        
        return VP
        
    def FeatVar(features, target):
        #list to append models to
        l = []
        for f in features.columns:
            feat = features[f].values
            # flist.append(feat)
            target_vals = target
            
            # slope, intercept, r_value, p_value, std_err = stats.linregress(target_vals, feat)
            
            mdl = sm.OLS(target_vals, feat).fit()
            l.append([mdl.rsquared, mdl.rsquared_adj])

        exp_var = np.array([x[1] for x in l])
        
        return exp_var
    
    
    def Train_veg_class(fuel_class, features_to_use, features, target, 
                        features_test = None, target_test = None, num_runs = 1000, 
                        testsize = 0.1, VP = True, VarPart = VarPart, FeatVar = FeatVar, PF = False):
        """
        Trains a model based to calculate the ground loading of a given fuel class

        """
        #temporary storage lists
        loss = []
        pred_vals = []
        vals = []
        params = []
        test_feat= []
        rsquared = []
        m_rsquared = []
        
        drone_r2_adj = []
        met_r2_adj = []
        both_r2_adj = []
        exp_v = []
        
        model_params = pd.DataFrame(columns = ['R2', 'R2_adj', 'RMSE_total', 'condition_number'], index = range(num_runs))
        
        #loop over dataset, using different subsamples to artificially expand dataset
        for i in tqdm(range(num_runs)):
            #use the optimum features
            feat = features[features_to_use]
            
            #split off one test sample
            X_train, X_test, Y_train, Y_test = train_test_split(feat,target[fuel_class], 
                                                                test_size=testsize, 
                                                                random_state=None)
            
            #multivariate linear fit
            model = sm.OLS(Y_train, X_train).fit()
            predictions = model.predict(X_test)
            predictions[predictions<0]=0
            
            #model parameters
            model_params['R2'].iloc[i] = model.rsquared
            model_params['R2_adj'].iloc[i] = model.rsquared_adj
            model_params['RMSE_total'].iloc[i] = np.sqrt(model.mse_total)
            model_params['condition_number'].iloc[i] = model.condition_number
            
            if VP == True:
                if not PF:
                    drone_features = ['Prop_foliage', 'Prop_grass', 'Prop_bare_soil', 'RCNN_count']
                    # drone_features = ['NDVI','BAI','DEM','RedEdge','NIR','Red','Green','Blue','Prop_unburned','Prop_burned']
                    met_features = ['map_tsf', 'sum_precipitation', 'mean_ssr', 
                                      'mean_soil_moisture', 'sum_evaporation', 'mean_temp']
                else:
                    drone_features = ['NDVI','BAI','DEM','RedEdge','NIR','Red','Green','Blue','Prop_unburned','Prop_burned']
                    # drone_features = ['NDVI','BAI','DEM','RedEdge', 'NIR','Green','Prop_unburned','Prop_burned']
                    met_features = ['wind_speed','temp','soil_moisture']
                    # met_features = []
                var_part = VarPart(fuel_class, drone_features, met_features, X_train, Y_train, model)
            exp_var = FeatVar(X_train, Y_train)
            
            #trees & shrubs only allow integer values
            if fuel_class == 'trees' or fuel_class == 'shrubs':
                predictions = round(predictions)
            
            loss.append((predictions.values - Y_test.values)**2)
            pred_vals.append(predictions.values)
            vals.append(Y_test.values)
            params.append(model.params)
            test_feat.append(X_test.values)
            
            #variance partitioning
            if VP:
                # drone_r2_adj.append(var_part['R^2_adj'].loc['Drone'])
                # met_r2_adj.append(var_part['R^2_adj'].loc['Met'])
                # both_r2_adj.append(var_part['R^2_adj'].loc['Drone'])
                drone_r2_adj.append(var_part['corr^2'].loc['Drone'])
                met_r2_adj.append(var_part['corr^2'].loc['Met'])
                both_r2_adj.append(var_part['corr^2'].loc['Drone'])
            exp_v.append(exp_var)
            
            #calculate rsquared value of predictions vs actual for this model, for use in weighting
            slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test.values,predictions.values)
            rsquared.append(r_value)
            m_rsquared.append(model.rsquared)
        
        #set threshold differences for different classes
        if fuel_class == 'grass':
            threshold = 100**2
        elif fuel_class == 'litter':
            threshold = 100**2
        elif fuel_class == 'coarse':
            threshold = 80**2
        elif fuel_class == 'heavy':
            threshold = 0.4**2  
        elif fuel_class == 'shrubs':
            threshold = 8**2    
        elif fuel_class == 'trees':
            threshold = 4**2
        elif fuel_class == 'CC_litter':
            threshold = 0.05**2
        else:
            threshold = 150**2
            
        #indices of those model runs where all losses are under the threshold
        indx = np.array([i for i,x in list(enumerate(loss)) if len(x[x<threshold]) == len(loss[0])])
        
        good_params = np.array(params)[indx]
        good_vals = np.array(vals)[indx]
        good_preds = np.array(pred_vals)[indx]
        good_test_feat = np.array(test_feat)[indx]
        good_rsquareds = np.array(rsquared)[indx]
        
        model_params = model_params.iloc[indx].mean()
           
            
            
        #add the coefficients to a dataframe and use the weighted mean to make predictions
        if not len(good_rsquareds[good_rsquareds==0]) == len(good_rsquareds):
            weighting = np.array([np.repeat(abs(x),len(good_params[0])) for x in good_rsquareds/max(good_rsquareds)])
        
            coeffs = pd.DataFrame(good_params, columns = features_to_use)
            av_pred_vals = np.array([np.sum(x*np.average(coeffs.values, axis = 0, weights = weighting), axis = 1) for x in good_test_feat])
    
            if fuel_class == 'shrubs' or fuel_class == 'trees':
                av_pred_vals = np.round([np.sum(x*np.average(coeffs.values, axis = 0, weights = weighting), axis = 1) for x in good_test_feat])
        
            if VP == True:
                drone_contrib = np.average(np.array(drone_r2_adj)[indx], weights = weighting[:,0])
                met_contrib = np.average(np.array(met_r2_adj)[indx], weights = weighting[:,0])
                joint_contrib = np.average(np.array(both_r2_adj)[indx], weights = weighting[:,0])
            
            r2_feat = np.average(np.array(exp_v)[indx], weights = weighting, axis = 0)
            
            # r2 = np.average(np.array(m_rsquared)[indx], weights = weighting[:,0])
            r2 = np.average(good_rsquareds**2, weights = weighting[:,0])
        
        else:
            coeffs = pd.DataFrame(good_params, columns = features_to_use)
            av_pred_vals = np.array([np.sum(x*coeffs.mean().values) for x in good_test_feat])
            
            weighting = np.ones(good_params.shape)
    
            if fuel_class == 'shrubs' or fuel_class == 'trees':
                av_pred_vals = np.round([np.sum(x*coeffs.mean().values) for x in good_test_feat])
            
            if VP == True:
                drone_contrib = np.mean(np.array(drone_r2_adj)[indx])
                met_contrib = np.mean(np.array(met_r2_adj)[indx])
                joint_contrib = np.mean(np.array(both_r2_adj)[indx])
                
            r2_feat = np.mean(np.array(exp_v)[indx])
            
            r2 = np.mean(np.array(m_rsquared)[indx])

        #eclude any negative predictions
        av_pred_vals[av_pred_vals<0] = 0
        
        #dataframe of components of each prediction, to be summed
        components = pd.DataFrame(columns = features_to_use)
        for i in range(len(good_test_feat)):
            
            R = pd.DataFrame([x*np.average(coeffs.values, axis = 0, weights = weighting) for x in good_test_feat[i]],columns = features_to_use)
            components = components.append(R)
        components = components.set_index(np.linspace(0,len(components)-1,num=len(components), dtype=int))
        
        #weighted average of coefficient
        mean_coeff = pd.DataFrame(np.average(coeffs.values, axis = 0, weights = weighting).reshape(-1, len(features_to_use)), columns = [x+' coefficient' for x in features_to_use])
      
        #results dataframe
        results = pd.DataFrame({'Target values': good_vals.ravel(), 
                                       'Predicted values': good_preds.ravel(),
                                       'Model prediction': av_pred_vals.ravel(),
                                       'Prediction difference': (good_preds.ravel() - av_pred_vals.ravel()).ravel(),
                                       'MSE Loss': ((good_vals.ravel() - av_pred_vals.ravel())**2).ravel()
                                       })
        
        #total model output in one dataframe, to write to an excel file 
        newres = pd.concat([results, components, mean_coeff], axis = 1)
        #make a prediction of all the features
        relevant_features = features[features_to_use]
        
        total_preds = np.array([np.sum(x*coeffs.mean().values) for x in np.array(relevant_features)])
        total_preds[total_preds<0] = 0
        if fuel_class == 'shrubs' or fuel_class == 'trees':
            total_preds = np.round(total_preds)
        
        if 'Transects' not in features.columns:
            features['Transects'] = features.index
            # if features_test != None:
            if len(features_test)>0:
                features_test['Transects'] = features_test.index
    
        bias = pd.DataFrame({'Target': target[fuel_class],
                             'Predictions': total_preds,
                             'bias': total_preds - target[fuel_class],
                             'Transects': features['Transects']})
        # newres['Transects'] = features['Transects']
        # newres['Predictions'] = total_preds
        # newres['bias'] = bias
        
        if features_test is not None and target_test is not None:
        
            predictions = np.sum(np.array([x*mean_coeff.values[0] for x in features_test[features_to_use].values]), axis = 1)
            if fuel_class == 'shrubs' or fuel_class == 'trees':
                predictions = np.round(np.sum(np.array([x*mean_coeff.values[0] for x in features_test[features_to_use].values]), axis = 1))
            
            predictions[predictions<0]=0
            if fuel_class == 'shrubs' or fuel_class == 'trees':
                predictions = np.round(predictions)
                
            
            
            test_res = pd.DataFrame({'Target values': target_test[fuel_class].values,
                                       'Model prediction': predictions.ravel(),
                                       'Prediction difference': (target_test[fuel_class].values - predictions.ravel()),
                                       'MSE Loss': (target_test[fuel_class].values - predictions.ravel())**2
                                       })
        
            test_components = pd.DataFrame(np.array([(x*mean_coeff.values)[0] for x in features_test[features_to_use].values]),columns = features_to_use)
            
            test_newres = pd.concat([test_res, test_components, mean_coeff], axis = 1)
            test_bias = pd.DataFrame({'Target': target_test[fuel_class],
                                        'Predictions': predictions,
                                      'bias': predictions - target_test[fuel_class],
                                      'Transects': features_test['Transects']})
            if VP == True:
                
                return model_params, newres, test_newres, bias, test_bias, [drone_contrib, met_contrib, joint_contrib], r2_feat, r2
            else:
                return model_params, newres, test_newres, bias, test_bias, r2_feat, r2
        
        else:
            if VP == True:
                return model_params, newres, bias, [drone_contrib, met_contrib, joint_contrib], r2_feat, r2, model_params
            else:
                return model_params, newres, bias, r2_feat, r2, model_params
            
            
    def LOOCV_model(X,y):
        
        
        X_data = X.copy()
        y_data = y.copy()
        
        cv = LeaveOneOut()
        # enumerate splits
        y_true, y_pred = list(), list()
        i = 0
        for train_ix, test_ix in cv.split(X):
            # print('{} of {}'.format(i+1, len(X)))
            i+=1
            # split data
            X_train, X_test = X_data[train_ix, :], X_data[test_ix, :]
            y_train, y_test = y_data[train_ix], y_data[test_ix]
            # fit model
            model = sm.OLS(y_train, X_train).fit()
            # model.fit(X_train, y_train)
           	# evaluate model
            yhat = model.predict(X_test)
            # store   
            y_true.append(y_test[0])
            y_pred.append(yhat[0])
            
        return np.array(y_true), np.array(y_pred)
        
        
        
    def comparison_plots(fuel_class, model_output, bias, target, features_to_use,
                         test_output = None, target_test = None, test_bias = None, savepath = None,
                         num_runs = 1000, testsize = 0.1, RMSE = None):
        """
        Plots a compaison between actual and predicted values in a given fuel class
        """
        #set letter to plot with
        if fuel_class == 'grass':
            letter = '(a)'
            title = 'Grass'
        elif fuel_class == 'litter':
            letter = '(b)'
            title = 'Litter'
        elif fuel_class == 'coarse':
            letter = '(d)'
            title = 'Coarse'
        elif fuel_class == 'Total_fine':
            letter = '(c)'
            title = 'Total fine'
        elif fuel_class == 'shrubs':
            letter = '(e)'
            title = 'Shrubs'
        elif fuel_class == 'trees':
            letter = '(f)'
            title = 'Trees'
        elif fuel_class == 'Surface_fuels':
            letter = '(h)'
            title = 'Surface fuels'
        elif fuel_class == 'heavy':
            letter = '(g)'
            title = 'Heavy'
        elif fuel_class == 'PFL':
            letter = ''
            title = 'Post-fire Litter'
        else:
            letter = ''
            title = fuel_class
        
        #colour scheme for plots
        colours = list(np.zeros(len(bias)))
        greens = [i for i,x in enumerate(bias['Transects']) if 'Mozambique' in x]
        reds = [i for i,x in enumerate(bias['Transects']) if 'Botswana' in x]
        
        # matplotlib.rc('axes',edgecolor='white')
        matplotlib.rc('axes',edgecolor='black')
        
        def func(x,a):
            return a*x
        
        for i in greens:
            colours[i] = 'blue'
        for j in reds:
            colours[j] = 'red'
            
        if fuel_class == 'trees' or fuel_class == 'shrubs':
            units = '(count)'
        elif fuel_class == 'heavy':
            units = '(t m$^{-2}$)'
            
        else:
            units = '(g m$^{-2}$)'
        
        txtcol = 'black'
        #info for text bounding box
        # text_bbox = dict(facecolor='#002060', alpha=1, edgecolor='#002060')
        text_bbox = dict(facecolor='white', alpha=1, edgecolor='white')
        # plt.rcParams['figure.facecolor'] = '#002060'
        plt.rcParams['figure.facecolor'] = 'white'
        #plot the model prediction for the full dataset
        fig, ax = plt.subplots(figsize = (10,10))
        # ax.set_facecolor('#002060')
        plt.ylabel('Predicted '+units, fontsize = 24, color = txtcol)
        plt.xlabel('Actual '+units, fontsize = 24, color = txtcol)
        plt.title(title, fontsize = 26, color = txtcol)
        plt.xticks(fontsize = 20, color = txtcol)
        plt.yticks(fontsize = 20, color = txtcol)
        # plt.title('Predicted vs Actual biomass for '+fuel_class, fontsize = 20)
        plt.grid(True)
        
        #fit info for graph
        slope, intercept, r_value, p_value, std_err = stats.linregress(target[fuel_class], bias['Predictions'])
        
        #constrained curve_fit
        popt, pcov = curve_fit(func, bias['Predictions'].values, target[fuel_class].values)
        
        maxval = max(np.ma.max(np.ma.masked_invalid(target[fuel_class])), 
                     np.ma.max(np.ma.masked_invalid(bias['Predictions'])))
        plt.text(maxval*0.05,maxval*1.15, 'r$^2$: {:.3f}'.format(r_value**2), fontsize = 20, bbox = text_bbox, color = txtcol)
        plt.text(maxval*0.05, maxval*1.075, 'y = {:.3f}x + {:.3f}'.format(slope, intercept), fontsize = 20, bbox = text_bbox, color = txtcol)
        plt.text(maxval*0.05,maxval*1.3, letter, fontsize=20, bbox = text_bbox, color = txtcol)    
        plt.xlim(-maxval*0.05, maxval*1.4)
        plt.ylim(-maxval*0.05, maxval*1.4)

        
        x = np.linspace(0,maxval*1.4,num = 50)
        x_new = np.linspace(np.min(target[fuel_class]), maxval, 50)
    
        plt.scatter(target[fuel_class], bias['Predictions'], s = 200, color = colours)
        plt.plot()
        plt.plot(x,x,'g--')
        plt.plot(x_new, slope*x_new+intercept, txtcol)
        
        train_r2 = r_value**2
        
        if savepath != None:
            plt.savefig(savepath+'/graphs/'+fuel_class+'_'+str(num_runs)+'_runs_'+str(testsize)+'_total_points.png')
        
        if test_output is not None and target_test is not None and test_bias is not None:
            
            #colour scheme for plots
            colours = list(np.zeros(len(test_bias)))
            greens = [i for i,x in enumerate(test_bias['Transects']) if 'Mozambique' in x]
            reds = [i for i,x in enumerate(test_bias['Transects']) if 'Botswana' in x]
            
            for i in greens:
                colours[i] = 'blue'
            for j in reds:
                colours[j] = 'red'
            #plot the model prediction
            fig, ax = plt.subplots(figsize = (10,10))
            # ax.set_facecolor('#002060')
            plt.ylabel('Predicted '+units, fontsize = 18, color = txtcol)
            plt.xlabel('Measured '+units, fontsize = 18, color = txtcol)
            plt.xticks(fontsize = 14, color = txtcol)
            plt.yticks(fontsize = 14, color = txtcol)
            # plt.title('Predicted vs Actual biomass for '+fuel_class, fontsize = 20, color = txtcol)
            plt.grid(True)
            
            #fit info for graph
            slope, intercept, r_value, p_value, std_err = stats.linregress(test_output['Target values'],test_output['Model prediction'])
            
            maxval = max(np.ma.max(np.ma.masked_invalid(target[fuel_class])), 
                         np.ma.max(np.ma.masked_invalid(bias['Predictions'])))
            plt.text(maxval*0.05,maxval*1.15, 'r$^2$: {:.3f}'.format(r_value**2), fontsize = 20, color = txtcol)
            plt.text(maxval*0.05, maxval*1.1, 'y = {:.3f}x + {:.3f}'.format(slope, intercept), fontsize = 20, color = txtcol)
            plt.text(maxval*0.05,maxval*1.3, letter, fontsize=20, bbox = text_bbox, color = txtcol)
            # plt.text(maxval*0.05,maxval*1.2, 'Features used: '+', '.join(features_to_use), fontsize=14)    
            plt.xlim(-maxval*0.05, maxval*1.4)
            plt.ylim(-maxval*0.05, maxval*1.4)

            
            x = np.linspace(0,maxval*1.4,num = 50)
            x_new = np.linspace(np.min(test_output['Target values']), maxval, 50)
            print(RMSE)
        
            
            plt.errorbar(test_output['Target values'], test_output['Model prediction'],fmt='.', capsize = 0.5,
                         color = 'orange', yerr = np.repeat(RMSE, len(test_output)))
            if fuel_class in ['grass', 'litter', 'Total_fine']:
                plt.hlines(test_output['Model prediction'], target_test[fuel_class+'_lower'], target_test[fuel_class+'_upper'], color = 'purple')
            plt.scatter(test_output['Target values'], test_output['Model prediction'], s = 200, color = colours)
            # plt.plot()
            plt.plot(x,x,'g--')
            plt.plot(x_new, slope*x_new+intercept, txtcol)
           
            
            if savepath != None:
                plt.savefig(savepath+'/graphs/'+fuel_class+'_'+str(num_runs)+'_runs_'+str(testsize)+'_predictions.png')
                
        return train_r2, p_value, std_err
        
        
        
    def feature_comparison_plot(veg_class, feature_name, target, features):
        
        plt.figure(figsize = (12,12))
        plt.ylabel(veg_class, fontsize = 20)
        plt.xlabel(feature_name, fontsize = 20)
        plt.title('Relationship between '+veg_class+' and '+feature_name, fontsize=24)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        
        plt.scatter(features[feature_name], target[veg_class])
        
        # if not os.path.exists(path+'/class_sensitivity/'+veg_class):
        #     os.mkdir(path+'/class_sensitivity/'+veg_class)
        
        # plt.savefig(path+'/class_sensitivity/'+veg_class+'/Sensitivity_'+feature_name+'.png')
        
        
        
        
        
        
        
        
        
        
        
                    
                    
                