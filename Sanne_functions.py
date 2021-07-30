#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:29:01 2021

@author: tes520
"""

#import geospatial libraries - gdal for raster files and osr/ogr for vector files
from osgeo import gdal, osr

#numpy is almost always essential!
import numpy as np

#python coordinate projection library
from pyproj import Proj, transform 


def GetGeoTransform(raster_path):
    """
    Returns a dictionary object with the upper left (ul) and lower right (lr) coordinates of a raster from it's path,
    as well as the pixel size (res) and any relevant skewing.
    """
    
    #open a GDAL object containig the raster
    gdal_img = gdal.Open(raster_path)
    
    #extract basic geospatial data
    ulx, xres, xskew, uly, yskew, yres  = gdal_img.GetGeoTransform()
    
    #calculate lower right coordinates from upper left coordinates and raster size
    lrx = ulx + (gdal_img.RasterXSize * xres)
    lry = uly + (gdal_img.RasterYSize * yres)
    
    geoinfo = {'ulx': ulx,
         'lrx': lrx,
         'uly': uly,
         'lry': lry,
         'xres': xres,
         'xskew': xskew,
         'yres': yres,
         'yskew': yskew
         }
    
    return geoinfo

def GetGridFromRaster(raster_path):
    """
    Similar to the GetGeoTransform function, but goes one step further: it returns the full latitude/longitude grid of the raster
    """
    
    #geotransform dictionary from previous function
    gt = GetGeoTransform(raster_path)
    
    #latitude and longitude arrays using numpy
    x = np.arange(gt['ulx'], gt['lrx'], gt['xres'])
    y = np.arange(gt['uly'], gt['lry'], gt['yres'])
    
    return x,y

def UTMZone(x,y):
    """
    This function tells you the EPSG code for the UTM zone in which the coordinates are found.
    Parameters
    ----------
    x : Longitude in decimal degrees
    y : Latitude in decimal degrees

    Returns
    -------
    epsg : EPSG code for UTM zone containing coordinate (x,y)

    """

    #take longitudinal coordinate and add 180, then divide by 6 and round up
    lon = int(np.ceil((x + 180)/6))
    
    #determine whether y is in the Northern or Southern Hemisphere
    if y > 0:
        code = 326
    else:
        code = 327
    
    #return epsg of the utm zone
    epsg = int(str(code)+str(lon))
    return epsg

def Reproject(x, y, in_grid = 4326, out_grid = 32737):
    """
    Transforms a pair of input coordinates to another reference system

    Parameters
    ----------
    x : longitude coordinate
    y : latitude coordinate
    in_grid : EPSG number of the input coordinates. The default is 4326 (WGS 84).
    out_grid : EPSG number of the output coordinates. The default is 32737 (UTM 37S).

    Returns
    -------
    Transformed x,y coordinates

    """
    
    inProj = Proj(init='epsg:'+str(in_grid))
    outProj = Proj(init='epsg:'+str(out_grid))
    
    
    x2,y2 = transform(inProj,outProj,x,y)
    
    return x2, y2

def UpdateGT(out_file, data, src_file, epsg = 4326, drv = 'GTiff', datatype = gdal.GDT_Float32, NoData = -999):
    """
    This function takes a numpy array (data) with either 2 or 3 dimensions and writes it to a raster, using geospatial data from an existing file.
    This is useful for example if you want to write a raster for the same location, but with other information (e.g. calculating NDVI from reflectance values)
    """
    
    #assign which driver to use for the file format - default is .tif
    driver = gdal.GetDriverByName(drv)
    
    #source raster from which to use the geospatial metadata
    src_gt = gdal.Open(src_file)
    
    #data shape - detects whether to produce multiple bands or not
    if len(data.shape) == 2:
        [cols, rows] = data.shape
        n = 1
    else:
        [n, cols, rows] = data.shape

    #create the destination file
    dst_gt = driver.Create(out_file, rows, cols, n, datatype)
    
    #get the geotransform from the source file
    gt = src_gt.GetGeoTransform()
    
    #coordinate system in which to create the raster, in wkt form
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()
    
    #set the projection in the desired reference system
    dst_gt.SetGeoTransform(gt)
    dst_gt.SetProjection(dest_wkt)
    
    #set the data
    if n==1:
        dst_gt.GetRasterBand(1).WriteArray(data)
        dst_gt.GetRasterBand(1).SetNoDataValue(NoData)
        #close and write to file
        dst_gt.FlushCache()
    #used if there are multiple bands
    else:
        for i in range(n):
            dst_gt.GetRasterBand(i+1).WriteArray(data[i])
            dst_gt.GetRasterBand(i+1).SetNoDataValue(NoData)
            dst_gt.FlushCache()
            
            
