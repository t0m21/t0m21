#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:37:42 2020

@author: tes520
"""

from glob import glob
from osgeo import gdal, osr, ogr
from pyproj import Proj, transform 
from shapely.geometry import Point, Polygon

import geopandas as gpd
import pandas as pd
import numpy as np

import os
import subprocess
import re
import pickle
from xml.etree import cElementTree as ET


# from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
# from datetime import date
import datetime as dtm

def UpdateGT(out_file, data, src_file, epsg = 4326, drv = 'GTiff', datatype = gdal.GDT_Float32, NoData = -999):
    """
    This function takes an aligned raster and updates the GeoLocation metadata
    to match that of the unaligned band.
    This is different from the function within the Classifier class only in that there is no default source file.
    """
    
    driver = gdal.GetDriverByName(drv)
    
    #source raster
    src_gt = gdal.Open(src_file)
    
    #array details
    if len(data.shape) == 2:
        [cols, rows] = data.shape
        n = 1
        # data.reshape(-1, cols, rows)
        # print(data.shape)
    else:
        [n, cols, rows] = data.shape

    #create the destination file
    dst_gt = driver.Create(out_file, rows, cols, n, datatype)
    
    #get the geotransform from the source file
    gt = src_gt.GetGeoTransform()
    
    #create wkt to write to the file metadata
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()
    
    #set the projection
    dst_gt.SetGeoTransform(gt)
    dst_gt.SetProjection(dest_wkt)
    
    #set the data
    if n==1:
        dst_gt.GetRasterBand(1).WriteArray(data)
        dst_gt.GetRasterBand(1).SetNoDataValue(NoData)
        dst_gt.FlushCache()
    else:
        for i in range(n):
            dst_gt.GetRasterBand(i+1).WriteArray(data[i])
            dst_gt.GetRasterBand(i+1).SetNoDataValue(NoData)
            dst_gt.FlushCache()
    
    

def GetGeoTransform(path_to_tif):
    """
    Returns a dictionary object of the geotransform info given by GDAL
    """
    gdal_img = gdal.Open(path_to_tif)
    ulx, xres, xskew, uly, yskew, yres  = gdal_img.GetGeoTransform()
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

def Reproject(x, y, in_grid = 4326, out_grid = 32737):
    """
    Transforms a pair of coordinates to another reference system

    Parameters
    ----------
    x : longitude coordinate
    y : latitude coordinate
    in_grid : EPSG number of the input coordinates. The default is 4326.
    out_grid : EPSG number of the output coordinates. The default is 32727.

    Returns
    -------
    Transformed x,y coordinates

    """
    
    inProj = Proj(init='epsg:'+str(in_grid))
    outProj = Proj(init='epsg:'+str(out_grid))
    
    
    x2,y2 = transform(inProj,outProj,x,y)
    
    return x2, y2

def GPD_Shape(coordinates, epsg = 4326):
    """
    Returns a GeoPandas DataFrame of a shape from a given coordinate set,
    which can then be written to a shapefile.
    Requires coordinate in the form of a list of tuples, e.g. [(x1,y1), (x2,y2),.....]
    """
    

    #create a polygon from input coordinates
    if len(coordinates) == 1:
        x,y = coordinates[0][0], coordinates[0][1]
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x], [y]),crs = {'init' :'epsg:'+str(epsg)})
        
    else:
        gdf = gpd.GeoDataFrame(crs = {'init' :'epsg:'+str(epsg)})
        polygon = Polygon(coordinates)
        gdf.loc[0,'geometry'] = polygon
        
    gdf = gdf.to_crs('EPSG:'+str(epsg))
        
    return gdf


def TransectDataFinder(coordinates, raster_path, epsg):
    
    #find the mid point of the given plot
    # x_mid = np.mean([x[0] for x in coordinates])
    # y_mid = np.mean([x[1] for x in coordinates])
    
    #get coordinates of the raster
    # raster_gt = GetGeoTransform(raster_path)
    # x_sat = np.arange(raster_gt['ulx'], raster_gt['lrx'], raster_gt['xres'])
    # y_sat = np.arange(raster_gt['uly'], raster_gt['lry'], raster_gt['yres'])
    # 
    #plot polygon object & midpoint
    shape = GPD_Shape(coordinates, epsg=epsg)
    # midpoint = GPD_Shape([(x_mid,y_mid)], epsg = epsg)

    #bounding box polygon
    # surrounding_shape = GPD_Shape(pixel_coords, epsg = epsg)
    
    
    #add a random element to the tiff to avoid double pixels
    pixel_arr = gdal.Open(raster_path).ReadAsArray()
    pixel_arr_rand = pixel_arr + np.random.rand(pixel_arr.shape[-2], pixel_arr.shape[-1])*10000
    UpdateGT(raster_path.replace('.jp2', '_rand.jp2'),
         pixel_arr_rand,
         raster_path, epsg = epsg)
    
    #vectorize the surrounding pixels
    vectorize_cmd = 'gdal_polygonize.py '+\
                    raster_path.replace('.jp2', '_rand.jp2')+' '+\
                    raster_path.replace('.jp2', '_pixels.shp')
    subprocess.call(vectorize_cmd, shell=True)
    
     #read shapes back into a new geopandas object
    pixel_df = gpd.read_file(raster_path.replace('.jp2', '_pixels.shp'))
    pixel_df['DN'] = gdal.Open(raster_path).ReadAsArray().ravel()
    pixel_df['area_m'] = pixel_df.geometry.area
    
    #find overlap of each pixel object with transect shapefile
    overlay = gpd.overlay(shape, pixel_df)
    overlay['proportion'] = overlay.area/overlay['area_m']
    
    #return weighted average value, ignoring nodata regions
    return np.ma.average(np.ma.masked_equal(overlay['DN'].values, -999), weights = overlay['proportion'])
    
    
    
    
    
    
    
    
def PlotFinder_Sentinel(coordinates, 
                        tile_path, 
                        epsg, 
                        plot_name, 
                        savepath = '/Volumes/LaCie/Data_Analysis/Upscaling/sat_plots/', 
                        distance = 60,
                        AGB = False,
                        cc = None):
    """
    

    Parameters
    ----------
    coordinates : List of coordinate tuples (x,y) bounding a given plot.
    tile_path : Path to directory containing band tiles
    epsg : ID number of tile coordinate reference system
    plot_name : Name/ID of plot
    savepath : Directory in which to save created shapefiles and rasters. The default is '/Volumes/LaCie/Data_Analysis/Upscaling/sat_plots/'.
    distance : Distance in m from which the satellite bounding box should extend from the plot midpoint. The default is 60m.

    Returns
    -------
    band_vals : Weighted mean value for each band within the plot area
    bnames : Name of each band

    """
    
    #list of available bands in the tile
    if AGB:
        bands_list = glob(tile_path+'/../AGB/*direct_v2.jp2')
    else:
        bands_list = glob(tile_path+'/*_B*.jp2')
    
    #cloud & SCL
    scl_path = glob(tile_path+'/*SCL*.jp2')[0]
    try:
        cld_path = glob(tile_path+'/../../QI_DATA/*CLD*20m.jp2')[0]
        CLD = True
    except IndexError:
        CLD = False
        
    
    #find the mid point of the given plot
    x_mid = np.mean([x[0] for x in coordinates])
    y_mid = np.mean([x[1] for x in coordinates])
    
    #get coordinates of satellite tile
    tile_gt = GetGeoTransform(bands_list[0])
    x_sat = np.arange(tile_gt['ulx'], tile_gt['lrx'], tile_gt['xres'])
    y_sat = np.arange(tile_gt['uly'], tile_gt['lry'], tile_gt['yres'])
    
    
    #indexing surrounding pixels
    x_inds = np.where(np.logical_and(x_mid-distance < x_sat, x_sat < x_mid+distance))[0]
    y_inds = np.where(np.logical_and(y_mid-distance < y_sat, y_sat < y_mid+distance))[0]
    
    segment_x = x_sat[x_inds]
    segment_y = y_sat[y_inds]
    
    #plot bounding box in the satellite tile
    pixel_coords = [(max(segment_x), max(segment_y)),
                    (max(segment_x), min(segment_y)),
                    (min(segment_x), min(segment_y)),
                    (min(segment_x), max(segment_y))]
    
    #plot polygon object & midpoint
    shape = GPD_Shape(coordinates, epsg=epsg)
    midpoint = GPD_Shape([(x_mid,y_mid)], epsg = epsg)

    #bounding box polygon
    surrounding_shape = GPD_Shape(pixel_coords, epsg = epsg)
    
    #create new folder for plot shapefiles
    if not os.path.exists(savepath+plot_name):
        os.mkdir(savepath+plot_name)
    
    #scl & cloudmask
    scl = gdal.Open(scl_path).ReadAsArray()[y_inds[0]:y_inds[-1]+1,x_inds[0]:x_inds[-1]+1]
    if CLD:
        cld = gdal.Open(cld_path).ReadAsArray()[y_inds[0]:y_inds[-1]+1,x_inds[0]:x_inds[-1]+1]
        
        if np.any(cld):
            cld_val = 1
        else:
            cld_val = 0
    else:
        cld_val = 0
    
    if len(np.unique(scl)) == 1:
        if np.unique(scl)[0] == 5 or np.unique(scl)[0] == 4:
            scl_val = 0
        else:
            scl_val = 1
    elif len(np.unique(scl)) == 2:
        if np.all(np.unique(scl) == np.array([4,5])):
            scl_val = 0
        else:
            scl_val = 1
    else:
        scl_val = 1
        
        
    #save shapefiles in plot folder
    shape.to_file(savepath+plot_name+'/'+plot_name+'.shp')
    midpoint.to_file(savepath+plot_name+'/'+plot_name+'_midpoint.shp')
    surrounding_shape.to_file(savepath+plot_name+'/'+plot_name+'_bounds.shp')
    
    # #crop SCL & cloud mask
    # crop_cmd = 'gdalwarp -overwrite -of GTiff -cutline '+\
    #             savepath+plot_name+'/'+plot_name+'_scl.shp'+\
    #             ' -crop_to_cutline -dstnodata -999.0 '+\
    #             scl_path+' '+\
    #             savepath+plot_name+'/'+plot_name+'_SCL.tif'
    # subprocess.call(crop_cmd, shell=True)
    
    
    
    band_vals = []
    bnames = []
    for band in bands_list:
        print(band)
        if AGB:
            bname = band.split('/')[-1].split('_')[0]
        else:
            bname = band.split('/')[-1].split('_')[-2]
        bnames.append(bname)
        
        #crop band tiles to bounding region
        crop_cmd = 'gdalwarp -overwrite -of GTiff -cutline '+\
                    savepath+plot_name+'/'+plot_name+'_bounds.shp'+\
                    ' -crop_to_cutline -dstnodata -999.0 '+\
                    band+' '+\
                    savepath+plot_name+'/'+plot_name+'_'+bname+'.tif'
        subprocess.call(crop_cmd, shell=True)
        
        #add a random element to the tiff to avoid double pixels
        pixel_arr = gdal.Open(savepath+plot_name+'/'+plot_name+'_'+bname+'.tif').ReadAsArray()
        pixel_arr_rand = pixel_arr + np.random.rand(pixel_arr.shape[0], pixel_arr.shape[1])*10000
        UpdateGT(savepath+plot_name+'/'+plot_name+'_'+bname+'_rand.tif',
             pixel_arr_rand,
             savepath+plot_name+'/'+plot_name+'_'+bname+'.tif', epsg = epsg)
        
        #vectorize the surrounding pixels
        vectorize_cmd = 'gdal_polygonize.py '+\
                        savepath+plot_name+'/'+plot_name+'_'+bname+'_rand.tif'+' '+\
                        savepath+plot_name+'/'+plot_name+'_'+bname+'_pixels.shp'
        subprocess.call(vectorize_cmd, shell=True)
        
        #read shapes back into a new geopandas object
        pixel_df = gpd.read_file(savepath+plot_name+'/'+plot_name+'_'+bname+'_pixels.shp')
        pixel_df['DN'] = gdal.Open(savepath+plot_name+'/'+plot_name+'_'+bname+'.tif').ReadAsArray().ravel()
        pixel_df['area_m'] = pixel_df.geometry.area
        
        #find overlap of each pixel object with transect shapefile
        overlay = gpd.overlay(shape, pixel_df)
        overlay['proportion'] = overlay.area/overlay['area_m']
        
        #add the weighted mean value to output list
        band_vals.append(np.average(overlay['DN'], weights = overlay['proportion']))
        
        
    
        
        
    #output weighted average band values, along with respective names
    return band_vals, bnames, cld_val, scl_val
        
        
def UTMZone(x,y):
    """
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
    
    #determine whether y is in NH or SH
    if y > 0:
        code = 326
    else:
        code = 327
    
    #return epsg of the utm zone
    epsg = int(str(code)+str(lon))
    return epsg


def Sentinel_GridConverter(gridpath = '/Volumes/LaCie/Data/Sat_data/Sentinel/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml'):
    
    
    #open the sentinel tile grid
    grid = ogr.Open(gridpath)
    
    #set up dataframe
    grid_df = gpd.GeoDataFrame(columns = ['TileName', 'EPSG', 'UTM', 'LatLon' ])
        
    layer = grid.GetLayer()    
    # for kml_lyr in layer:
    i = 0
    for feat in layer:
        # print(feat)
        
        #name of the tile
        tname = feat.GetField(0)
        
        #coordinates in local and global CRS
        details = feat.GetField(1).split('MULTIPOLYGON')
        utmcoords = [int(x) for x in re.findall(r'[0-9]+', details[1])[:10]]
        wgs84 = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", details[2])[:10]]
        
        #create polygon objects from coordinates
        utm = Polygon([(utmcoords[0], utmcoords[1]), 
                       (utmcoords[2], utmcoords[3]), 
                       (utmcoords[4], utmcoords[5]), 
                       (utmcoords[6], utmcoords[7]), 
                       (utmcoords[8], utmcoords[9])])
        latlon = Polygon([(wgs84[0], wgs84[1]), 
                          (wgs84[2], wgs84[3]), 
                          (wgs84[4], wgs84[5]), 
                          (wgs84[6], wgs84[7]),
                          (wgs84[8], wgs84[9])])
        #EPSG
        epsg_list = details[0].split('font')[7]
        epsg = [int(x) for x in re.findall(r'[0-9]+',epsg_list)][1]
        
        
        new_df = gpd.GeoDataFrame({'TileName': 'T'+tname,
                               'EPSG': epsg,
                               'UTM': utm,
                               'LatLon': latlon}, index = [i])
        
        grid_df = grid_df.append(new_df)
        i+=1
        
    grid_df.to_pickle('/Volumes/LaCie/Data/Sat_data/Sentinel/GRID.pickle')
    
    
def Sentinel_TileName(x,y,epsg = None, gridpath = '/Volumes/LaCie/Data/Sat_data/Sentinel/GRID.pickle'):
    
    pt = Point(x,y)
    #read in grid
    with open(gridpath, 'rb') as df:
        grid = pickle.load(df)
    
    #if epsg is given, only explore relevant options
    if epsg is not None:
        inds = np.where(grid['EPSG'] == epsg)[0]
        utmgrid = grid.iloc[inds]
        utmgrid = utmgrid.drop('LatLon', axis=1)                                                           #drop irrelevant WGS84 CRS
        utmgrid = gpd.GeoDataFrame(utmgrid.rename(columns={'UTM': 'geometry'}).set_geometry('geometry'))   #set geometry column
        
        for i in utmgrid.index:
            
            tile, epsg, bounds = utmgrid.loc[i]
            
            if bounds.contains(pt):
                break
            
    else:
        wgsgrid = grid.drop('UTM', axis=1)
        wgsgrid = gpd.GeoDataFrame(utmgrid.rename(columns={'LatLon': 'geometry'}).set_geometry('geometry'))
        
        for i in wgsgrid.index:
            
            tile, epsg, bounds = wgsgrid.loc[i]
        
            if bounds.contains(pt):
                break
            
    return tile, epsg
        
def Sentinel_ProductFinder(coords, date, from_date = None,
                           usr = 't0m44', pwd = 'Candyrat8568!',
                           platformname = 'Sentinel-2',
                           producttype = 'S2MSI1C',
                           cc = (0,50),
                           download_path = None):
    
    
    
    #connect to the API
    api = SentinelAPI(usr, pwd, 'https://scihub.copernicus.eu/dhus')
    
    
    #establish a footprint to search
    if len(coords) == 2:
        x,y = coords
        footprint = Point(x,y).to_wkt()
    else:
        footprint = Polygon(coords).to_wkt()
        
    #find a date range to search in
    if type(date) is str:
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        date = dtm.date(year, month, day)
    else:
        year = date.year
        month = date.month
        day = date.day
        date = dtm.date(year,month,day)
    
    
    #if no date range is given, search up to two weeks before the given date
    if from_date is None:
        from_date = date - dtm.timedelta(days=14)
        
    #find products fulfilling given conditions
    products = api.query(footprint,
                         platformname = platformname, 
                         date = (from_date, date),
                         cloudcoverpercentage = cc,
                         producttype = producttype)
    
    if download_path is not None:
        api.download_all(products)
    return products, api

def PySen2Cor(path,
              output_dir = '/Volumes/LaCie/Data/Sat_data/Sentinel/Validation/L2A',
              xmlfile = '/Users/tes520/sen2cor/2.8/cfg/L2A_GIPP.xml',
              LOG_TYPE='INFO',
              NR_THREADS='AUTO',
              DEM_OUTPUT='TRUE',
              TCI_OUTPUT='TRUE',
              DDV_OUTPUT='TRUE',
              DOWNSAMPLE_20_TO_60='TRUE',
              SEASON='SUMMER',
              OZONE_CONTENT=0,
              AEROSOL_TYPE='RURAL',
              DEM_DIRECTORY = '/Volumes/LaCie/Data/Sat_data/Sentinel/DEM',
              LEGACY=False
              ):
    
    #open xml as a text file
    with open(xmlfile, 'r') as F:
        config = F.read().split('\n')
        
    if '2.8' in xmlfile:
        #update configuration options
        config[3] = '    <Log_Level>'+LOG_TYPE+'</Log_Level>'
        config[5] = '    <Nr_Threads>'+NR_THREADS+'</Nr_Threads>'
        config[18] = '    <Generate_DEM_Output>'+DEM_OUTPUT+'</Generate_DEM_Output>'
        config[20] = '    <Generate_TCI_Output>'+TCI_OUTPUT+'</Generate_TCI_Output>'
        config[22] = '    <Generate_DDV_Output>'+DDV_OUTPUT+'</Generate_DDV_Output>'
        config[24] = '    <Downsample_20_to_60>'+DOWNSAMPLE_20_TO_60+'</Downsample_20_to_60>'
        config[54] = '      <Aerosol_Type>'+AEROSOL_TYPE+'</Aerosol_Type>'
        config[56] = '      <Mid_Latitude>'+SEASON+'</Mid_Latitude>'
        config[58] = '      <Ozone_Content>'+str(OZONE_CONTENT)+'</Ozone_Content>'
    elif '2.5' in xmlfile:
        config[3] = '    <Log_Level>'+LOG_TYPE+'</Log_Level>'
        config[7] = '    <Nr_Processes>'+NR_THREADS+'</Nr_Processes>'
        config[9] = '    <Target_Directory>'+output_dir+'</Target_Directory>'
        # config[11] = '    <DEM_Directory>'+DEM_DIRECTORY+'</DEM_Directory>'
        # config[15] = '    <Generate_DEM_Output>'+DEM_OUTPUT+'</Generate_DEM_Output>'
        # config[17] = '    <Generate_TCI_Output>'+TCI_OUTPUT+'</Generate_TCI_Output>'
        # config[19] = '    <Generate_DDV_Output>'+DDV_OUTPUT+'</Generate_DDV_Output>'
        # config[24] = '    <Downsample_20_to_60>'+DOWNSAMPLE_20_TO_60+'</Downsample_20_to_60>'
        config[65] = '      <Aerosol_Type>'+AEROSOL_TYPE+'</Aerosol_Type>'
        config[67] = '      <Mid_Latitude>'+SEASON+'</Mid_Latitude>'
        config[69] = '      <Ozone_Content>'+str(OZONE_CONTENT)+'</Ozone_Content>'
    
    xml = '\n'.join(config)
    with open(xmlfile, 'w') as G:
        G.write(xml)
        
    #conversion command here
    if not LEGACY:
        cmd = 'L2A_Process --output_dir '+output_dir+' '+path
        # subprocess.call(cmd, shell=True)
    else:
        cmd = '/Users/tes520/Sen2Cor-02.05.05-Darwin64/bin/L2A_Process '+path
    
    subprocess.call(cmd, shell=True)
    
    
    
    # tree = ET.parse(xmlfile)
    # root = tree.getroot()
    # xmldict = XmlDictConfig(root)
    
    
def GetGridFromRaster(raster_path):
    
    gt = GetGeoTransform(raster_path)
    xsize = gdal.Open(raster_path).RasterXSize
    ysize = gdal.Open(raster_path).RasterYSize
    # x = np.arange(gt['ulx'], gt['lrx'], gt['xres'])
    # y = np.arange(gt['uly'], gt['lry'], gt['yres'])
    
    x = np.linspace(gt['ulx'], gt['lrx'], xsize)
    y = np.linspace(gt['uly'], gt['lry'], ysize)
    
    return x,y
        
        
def TransectBoundingBox(start_coords, end_coords, d = 5, in_epsg = None, out_epsg = None): 
    
    
    x1, y1 = start_coords
    x2, y2 = end_coords

    if in_epsg is not None and out_epsg is not None:
     
        #transform coordinates into local grid system
        x1, y1 = Reproject(x1, y1, in_grid = in_epsg, out_grid = out_epsg)
        x2, y2 = Reproject(x2, y2, in_grid = in_epsg, out_grid = out_epsg)
    else:
        print('Warning! Coordinate system is not set, no projection will be carried out. Please adjust d accordingly.')
    
    #equation of the line passing through both points
    m = (y1-y2)/(x1-x2)
    m_perp = -1/m
    #intercept
    b_end = y2-x2 * m_perp
    b_start = y1-x1 * m_perp
    
    #find the coordinates
    x_bounds = np.array([x1 + d/np.sqrt(1+m_perp**2),
                         x1 - d/np.sqrt(1+m_perp**2),
                         x2 - d/np.sqrt(1+m_perp**2),
                         x2 + d/np.sqrt(1+m_perp**2)])
        
    y_bounds = np.array([x_bounds[0]*m_perp + b_start,
                         x_bounds[1]*m_perp + b_start,
                         x_bounds[2]*m_perp + b_end,
                         x_bounds[3]*m_perp + b_end])
    
    coordinates = [(x_bounds[0],y_bounds[0]),
                   (x_bounds[1],y_bounds[1]),
                   (x_bounds[2],y_bounds[2]),
                   (x_bounds[3],y_bounds[3])]      
    
    return coordinates


def resample_raster(raster_path, imgref, filename):
    """Resample angle bands.
    Parameters:
       rasterpath (str): raster to be resampled.
       imgref (str): path to image that will be used as reference.
       filename (str): filename of the resampled angle band.
    """
    src_ds = gdal.Open(imgref)
    src_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    geotrans = src_ds.GetGeoTransform() #get GeoTranform from existed 'data0'
    proj = src_ds.GetProjection() #you can get from a exsited tif or import 

    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize

    rasterOrigin = (geotrans[0],geotrans[3])
    
    #raster details
    raster = gdal.Open(raster_path)
    raster_gt = raster.GetGeoTransform()
    matrix = raster.ReadAsArray()

    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')
    tmp_ds = mem_drv.Create('', len(matrix[0]), len(matrix), 1, gdal.GDT_Float32)

    # Set the geotransform
    tmp_ds.SetGeoTransform((rasterOrigin[0], raster_gt[1], 0, rasterOrigin[1], 0, raster_gt[-1]))
    tmp_ds.SetProjection(proj)
    tmp_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    tmp_ds.GetRasterBand(1).WriteArray(matrix)

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotrans)
    dst_ds.SetProjection(proj)

    resampling = gdal.GRA_Bilinear
    gdal.ReprojectImage( tmp_ds, dst_ds, tmp_ds.GetProjection(), dst_ds.GetProjection(), resampling)

    del src_ds
    del tmp_ds
    del dst_ds

    return


def get_tileid(xml):
    """Get tile id from MTD_TL.xml file.
    Parameters:
       xml (str): path to MTD_TL.xml.
    Returns:
       str: .SAFE tile id.
    """
    tile_id = ""
    # Parse the XML file 
    tree = ET.parse(xml)
    root = tree.getroot()

    # Find the angles
    for child in root:
        if child.tag[-12:] == 'General_Info':
            geninfo = child

    for segment in geninfo:
        if segment.tag == 'TILE_ID':
            tile_id = segment.text.strip()

    return(tile_id)

def get_sun_angles(xml):
    """Extract Sentinel-2 solar angle bands values from MTD_TL.xml.
    Parameters:
       xml (str): path to MTD_TL.xml.
    Returns:
       str, str: sz_path, sa_path: path to solar zenith image, path to solar azimuth image, respectively.
    """
    solar_zenith_values = np.empty((23,23,)) * np.nan #initiates matrix
    solar_azimuth_values = np.empty((23,23,)) * np.nan

    # Parse the XML file 
    tree = ET.parse(xml)
    root = tree.getroot()

    # Find the angles
    for child in root:
        if child.tag[-14:] == 'Geometric_Info':
            geoinfo = child

    for segment in geoinfo:
        if segment.tag == 'Tile_Angles':
            angles = segment

    for angle in angles:
        if angle.tag == 'Sun_Angles_Grid':
            for bset in angle:
                if bset.tag == 'Zenith':
                    zenith = bset
                if bset.tag == 'Azimuth':
                    azimuth = bset
            for field in zenith:
                if field.tag == 'Values_List':
                    zvallist = field
            for field in azimuth:
                if field.tag == 'Values_List':
                    avallist = field
            for rindex in range(len(zvallist)):
                zvalrow = zvallist[rindex]
                avalrow = avallist[rindex]
                zvalues = zvalrow.text.split(' ')
                avalues = avalrow.text.split(' ')
                values = list(zip( zvalues, avalues )) #row of values
                for cindex in range(len(values)):
                    if ( values[cindex][0] != 'NaN' and values[cindex][1] != 'NaN' ):
                        zen = float(values[cindex][0] )
                        az = float(values[cindex][1] )
                        solar_zenith_values[rindex,cindex] = zen
                        solar_azimuth_values[rindex,cindex] = az
    return (solar_zenith_values, solar_azimuth_values)


def get_sensor_angles(xml):
    """Extract Sentinel-2 view (sensor) angle bands values from MTD_TL.xml.
    Parameters:
       xml (str): path to MTD_TL.xml.
    Returns:
       str, str: path to view (sensor) zenith image and path to view (sensor) azimuth image, respectively.
    """
    numband = 13
    sensor_zenith_values = np.empty((numband,23,23)) * np.nan #initiates matrix
    sensor_azimuth_values = np.empty((numband,23,23)) * np.nan

    # Parse the XML file 
    tree = ET.parse(xml)
    root = tree.getroot()

    # Find the angles
    for child in root:
        if child.tag[-14:] == 'Geometric_Info':
            geoinfo = child

    for segment in geoinfo:
        if segment.tag == 'Tile_Angles':
            angles = segment

    for angle in angles:
        if angle.tag == 'Viewing_Incidence_Angles_Grids':
            bandId = int(angle.attrib['bandId'])
            for bset in angle:
                if bset.tag == 'Zenith':
                    zenith = bset
                if bset.tag == 'Azimuth':
                    azimuth = bset
            for field in zenith:
                if field.tag == 'Values_List':
                    zvallist = field
            for field in azimuth:
                if field.tag == 'Values_List':
                    avallist = field
            for rindex in range(len(zvallist)):
                zvalrow = zvallist[rindex]
                avalrow = avallist[rindex]
                zvalues = zvalrow.text.split(' ')
                avalues = avalrow.text.split(' ')
                values = list(zip( zvalues, avalues )) #row of values
                for cindex in range(len(values)):
                    if ( values[cindex][0] != 'NaN' and values[cindex][1] != 'NaN' ):
                        zen = float( values[cindex][0] )
                        az = float( values[cindex][1] )
                        sensor_zenith_values[bandId, rindex,cindex] = zen
                        sensor_azimuth_values[bandId, rindex,cindex] = az
    return(sensor_zenith_values, sensor_azimuth_values)


def write_intermediary(newRasterfn, rasterOrigin, proj, array):
    """Writes intermediary angle bands (not resampled, as 23x23 5000m spatial resolution).
    Parameters:
       newRasterfn (str): output raster file name.
       rasterOrigin (tuple): gdal geotransform origin tuple (geotrans[0],geotrans[3]).
       proj (str): gdal projection.
       array (array): angle values array.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, 5000, 0, originY, 0, -5000))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRaster.SetProjection(proj)
    outband.FlushCache()

    return
    
def generate_anglebands(xml):
    """Generate angle bands.
    Parameters:
       xml (str): path to MTD_TL.xml.
    """
    path = os.path.split(xml)[0]
    imgFolder = path + "/IMG_DATA/R20m"
    angFolder = path + "/ANG_DATA/"
    os.makedirs(angFolder, exist_ok=True)

    #use band 7 as reference due to 20m spatial resolution
    imgref = [f for f in glob(imgFolder + "/*B07*.jp2", recursive=True)][0]

    tmp_ds = gdal.Open(imgref)
    tmp_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    geotrans = tmp_ds.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj = tmp_ds.GetProjection() #you can get from a exsited tif or import 

    scenename = get_tileid(xml)
    solar_zenith, solar_azimuth = get_sun_angles(xml)
    sensor_zenith, sensor_azimuth = get_sensor_angles(xml)

    rasterOrigin = (geotrans[0],geotrans[3])

    write_intermediary((angFolder + scenename + "solar_zenith"),rasterOrigin, proj, solar_zenith)
    write_intermediary((angFolder + scenename + "solar_azimuth"),rasterOrigin, proj, solar_azimuth)
    for num_band in (range(len(sensor_azimuth))):
        write_intermediary((angFolder + scenename + "sensor_zenith_b" + str(num_band)), rasterOrigin, proj, sensor_zenith[num_band])
        write_intermediary((angFolder + scenename + "sensor_azimuth_b" + str(num_band)), rasterOrigin, proj, sensor_azimuth[num_band])

    del tmp_ds

    return


def resample_anglebands(ang_matrix, imgref, filename):
    """Resample angle bands.
    Parameters:
       ang_matrix (arr): matrix of angle values.
       imgref (str): path to image that will be used as reference.
       filename (str): filename of the resampled angle band.
    """
    src_ds = gdal.Open(imgref)
    src_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    geotrans = src_ds.GetGeoTransform() #get GeoTranform from existed 'data0'
    proj = src_ds.GetProjection() #you can get from a exsited tif or import 

    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize

    rasterOrigin = (geotrans[0],geotrans[3])

    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')
    tmp_ds = mem_drv.Create('', len(ang_matrix[0]), len(ang_matrix), 1, gdal.GDT_Float32)

    # Set the geotransform
    tmp_ds.SetGeoTransform((rasterOrigin[0], 5000, 0, rasterOrigin[1], 0, -5000))
    tmp_ds.SetProjection(proj)
    tmp_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    tmp_ds.GetRasterBand(1).WriteArray(ang_matrix)

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotrans)
    dst_ds.SetProjection(proj)

    resampling = gdal.GRA_Bilinear
    gdal.ReprojectImage( tmp_ds, dst_ds, tmp_ds.GetProjection(), dst_ds.GetProjection(), resampling)

    del src_ds
    del tmp_ds
    del dst_ds

    return

def generate_resampled_anglebands(xml):
    """Generates angle bands resampled to 10 meters.
    Parameters:
       xml (str): path to MTD_TL.xml.
    Returns:
       str, str, str, str: path to solar zenith image, path to solar azimuth image, path to view (sensor) zenith image and path to view (sensor) azimuth image, respectively.
    """
    path = os.path.split(xml)[0]
    imgFolder = path + "/IMG_DATA/R20m"
    angFolder = path + "/ANG_DATA/"
    os.makedirs(angFolder, exist_ok=True)

    #use band 7 as reference due to 20m spatial resolution
    imgref = [f for f in glob(imgFolder + "/*B07*.jp2", recursive=True)][0]

    scenename = get_tileid(xml)
    solar_zenith, solar_azimuth = get_sun_angles(xml)
    sensor_zenith, sensor_azimuth = get_sensor_angles(xml)

    sensor_zenith_mean = sensor_zenith[0]
    sensor_azimuth_mean = sensor_azimuth[0]
    for num_band in (range(1,len(sensor_azimuth))):
        sensor_zenith_mean += sensor_zenith[num_band]
        sensor_azimuth_mean += sensor_azimuth[num_band]
    sensor_zenith_mean /= len(sensor_azimuth)
    sensor_azimuth_mean /= len(sensor_azimuth)

    sz_path = angFolder + scenename + '_solar_zenith_resampled.tif'
    sa_path = angFolder + scenename + '_solar_azimuth_resampled.tif'
    vz_path = angFolder + scenename + '_sensor_zenith_mean_resampled.tif'
    va_path = angFolder + scenename + '_sensor_azimuth_mean_resampled.tif'

    resample_anglebands(solar_zenith, imgref, sz_path)
    resample_anglebands(solar_azimuth, imgref, sa_path)
    resample_anglebands(sensor_zenith_mean, imgref, vz_path)
    resample_anglebands(sensor_azimuth_mean, imgref, va_path)

    return sz_path, sa_path, vz_path, va_path

    