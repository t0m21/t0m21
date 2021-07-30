#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:50:58 2020

@author: tes520
"""
import ESMF

import numpy as np 
import subprocess

from osgeo import gdal
from tqdm import tqdm

import sys
# sys.path.append('/media/tes520/LaCie1/Mapping/modules')
# from __FL import Read_NCDF
# from __CLASS import GetGeoTransform, UpdateGT
from __GEO import GetGeoTransform, UpdateGT


def regrid_to_sat_tile(raster_path, varname, tileshape_path,
                       sat_lats, sat_lons):
    
    #crop to tile shapefile
    crop_cmd = 'gdalwarp -cutline '+tileshape_path+' -crop_to_cutline '+\
                   raster_path+' '+\
                   tileshape_path.replace('tile.shp', varname+'_tile.tif')
    subprocess.call(crop_cmd, shell=True)
    cropped_path = tileshape_path.replace('tile.shp', varname+'_tile.tif')

    
    #read in raster and get geotransform data
    raster = gdal.Open(cropped_path)
    raster_gt = GetGeoTransform(cropped_path)
    raster_arr = raster.ReadAsArray()
    
    lons_in = np.arange(raster_gt['ulx'], raster_gt['lrx'], raster_gt['xres'])
    lats_in = np.arange(raster_gt['uly'], raster_gt['lry'], raster_gt['yres'])
    
    #lat/lon info to feed into regridder
    x_dist = raster_gt['xres']/2
    y_dist = raster_gt['yres']/2
    
    if len(raster_arr.shape) > 2:
        n = len(raster_arr)
        raster_regridded = np.zeros((n, sat_lats.shape[0], sat_lons.shape[0]))
    else:
        n = 1
        raster_regridded = np.zeros((n, sat_lats.shape[0], sat_lons.shape[0]))
        raster_arr = raster_arr.reshape(1, raster_arr.shape[0], raster_arr.shape[1])
     
    #regrid by selecting a large MODIS pixel, and finding the overlapping pixel
    for i in tqdm(range(len(lons_in))):
        x = lons_in[i]
    
        xinds = np.where(np.logical_and(x - x_dist <= sat_lons, sat_lons <= x + x_dist))[0]
        #exclude areas outside the raster
        if len(xinds) == 0:
            continue
        
        for j in range(len(lats_in)):
            y = lats_in[j]
        
            yinds = np.where(np.logical_and(y + y_dist <= sat_lats, sat_lats <= y - y_dist))[0]
            if len(yinds) == 0:
                continue
            for k in range(n):
                raster_regridded[k,yinds[0]:yinds[-1]+1,xinds[0]:xinds[-1]+1] = raster_arr[k,j,i]
    #write to file
    UpdateGT(tileshape_path.replace('tile.shp',varname+'_regridded.tif'), raster_regridded, 
             tileshape_path.replace('tile.shp', 'tile.tif'))
    
    return raster_regridded
 

def regrid(dataset,src_lon,src_lat,dst_lon,dst_lat,FORTRAN_CONTIGUOUS = True):
    
    ESMF.Manager(debug=True)


    #use fortran engine when reshaping
    if FORTRAN_CONTIGUOUS:
        dst_shape = dst_lat.T.shape
        src_shape = src_lat.T.shape
    else:
        dst_shape = dst_lat.shape
        src_shape = src_lat.shape
        
        
    #source met grid
    sourcegrid = ESMF.Grid(np.array(src_shape), staggerloc=ESMF.StaggerLoc.CENTER, coord_sys=ESMF.CoordSys.SPH_DEG)
    #destination grid based on LDS date data
    destgrid = ESMF.Grid(np.array(dst_shape), staggerloc=ESMF.StaggerLoc.CENTER, coord_sys=ESMF.CoordSys.SPH_DEG)
    
    source_lon = sourcegrid.get_coords(0)
    source_lat = sourcegrid.get_coords(1)
    
    dest_lon = destgrid.get_coords(0)
    dest_lat = destgrid.get_coords(1)
    
    #add data to pointers
    if FORTRAN_CONTIGUOUS:
        source_lon[...] = src_lon.T
        source_lat[...] = src_lat.T
    
        dest_lon[...] = dst_lon.T
        dest_lat[...] = dst_lon.T
    else:
        source_lon[...] = src_lon
        source_lat[...] = src_lat
    
        dest_lon[...] = dst_lon
        dest_lat[...] = dst_lat
        
    sourcefield = ESMF.Field(sourcegrid, name='ECMWF 0.1x0.1')
    
    destfield = ESMF.Field(destgrid, name='LDS 50kmx50km')
    
    if FORTRAN_CONTIGUOUS:
        sourcefield.data[...] = dataset.T
    else:
        sourcefield.data[...] = dataset
        
    regrid = ESMF.Regrid(sourcefield, destfield, regrid_method=ESMF.RegridMethod.BILINEAR,  
                         unmapped_action=ESMF.UnmappedAction.IGNORE)
    
    destfield = regrid(sourcefield, destfield)
    
    return destfield.data.T



def write_to_raster(outfile, data, lats, lons, epsg = 4326):
    
    xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
    
    if np.shape(data) > 2:
        n,nrows,ncols = np.shape(data)
    else:
        nrows,ncols = np.shape(data)
        data.reshape(1, data.shape[0], data.shape[1])
        n = 1
        
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)   
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up), 
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    
    output_raster = gdal.GetDriverByName('GTiff').Create(outfile ,ncols, nrows, n, gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(epsg)                     # This one specifies WGS84 lat long.
                                                 # Anyone know how to specify the 
                                                 # IAU2000:49900 Mars encoding?
    output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
    
    for i in range(n):                                                  # to the file
        output_raster.GetRasterBand(i+1).WriteArray(data[i])   # Writes my array to the raster
    
    output_raster.FlushCache()