#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:41:04 2020

@author: tes520
"""


import subprocess
import os





class ImageProcessing():

    
    def ODM(path_to_images, 
            verbose = True, 
            ignore_gsd = True, 
            dtm = True, 
            texturing_data_term = 'area',
            texturing_nadir_weight = 5,
            smrf_scalar = 3,
            smrf_slope = 0.15,
            smrf_threshold = 0.2,
            resolution = 5,
            cutline = True,
            log=True,
            output_path = None):
        """
        Python skin for ODM orthophoto creation. 
        Descriptions and more detail are found on ODM documentation pages at 
        https://docs.opendronemap.org/arguments.html

        Parameters
        ----------
        path_to_images : Path to ODM folder containing drone images.
        verbose : bool, optional
            Print additional messages to the console. The default is True.
        ignore_gsd : bool, optional
            Ignore Ground Sampling Distance (GSD). GSD caps the
            maximum resolution of image outputs and resizes images
            when necessary, resulting in faster processing and
            lower memory usage. Since GSD is an estimate,
            sometimes ignoring it can result in slightly better
            image output quality.
            The default is True.
        dtm : bool, optional
            Use this tag to build a DTM (Digital Terrain Model,
            ground only) using a simple morphological filter.
            The default is True.
        texturing_data_term : string, optional
            Data term: [area, gmi]. The default is 'area'.
        texturing_nadir_weight : int, optional
            Integer between 0 and 32.
            Affects orthophotos only. Higher values result in
            sharper corners, but can affect color distribution and
            blurriness. Use lower values for planar areas and
            higher values for urban areas. The value 16 works
            well for most scenarios. The default is 5.
        smrf_scalar : float, optional
            Simple Morphological Filter elevation scalar
            parameter. The default is 3.
        smrf_slope : float, optional
            Simple Morphological Filter slope parameter (rise over
            run). The default is 0.15.
        smrf_threshold : float, optional
            Simple Morphological Filter elevation threshold
            parameter (meters). The default is 0.5.
        resolution : float, optional
            Orthophoto resolution in cm / pixel. The default is 5.
        cutline : bool, optional
            Generates a polygon around the cropping area that cuts
            the orthophoto around the edges of features. This
            polygon can be useful for stitching seamless mosaics
            with multiple overlapping orthophotos. The default is True.
        log : bool, optional
            Save the ODM output to a lof file. The default is True.
        output_path : string, optional
            Path to the log file. If no path is given, 
            defaults to the ODM folder containing the images. 
            The default is None.

        Returns
        -------
        Runs the ODM process in docker with the given parameters.

        """
    
        
        
        if verbose:
            v = '-v'
        else:
            v = ''
            
        if ignore_gsd:
            gsd = '--ignore-gsd'
        else:
            gsd = ''
        
        if dtm:
            dem = '--dtm --dsm'
        else:
            dem = ''
        
        if cutline:
            cl = ' --orthophoto-cutline '
        else:
            cl = ''
            
        if log:
            if output_path == None:
                logpath = ' > '+path_to_images+'/odm_output.log'
            else:
                logpath = output_path
        else:
            logpath = ''
            
        
        ODM_cmd = 'docker run -t --rm '+v+' '+path_to_images+\
                ':/datasets/code opendronemap/odm --project-path /datasets --time '+\
                gsd+' --texturing-nadir-weight '+str(texturing_nadir_weight)+' --texturing-data-term '+texturing_data_term+\
                ' '+dem+' --dem-resolution '+str(resolution)+' --orthophoto-resolution '+str(resolution)+\
                ' --smrf-scalar '+str(smrf_scalar)+' --smrf-slope '+str(smrf_slope)+cl+\
                ' --smrf-threshold '+str(smrf_threshold)+logpath
                
        print(ODM_cmd)
        subprocess.call(ODM_cmd, shell=True)
                
        
        
        
    def DEM_image(path):
        """
        

        Parameters
        ----------
        path : Path to pre-fire flight folder.

        Returns
        -------
        Creates a 4-band standardised image with band 1 as the DEM, and bands 2-4 as R, G, B respectively.

        """
        
        
        #find the aligned orthophoto
        ortho_path = path+'/ODM/odm_orthophoto/odm_orthophoto.tif'
        
        #check a DEM file exists
        if not os.path.exists(path+'/ODM/odm_dem/dem.tif'):
            # dem_cmd = 'gdal_calc.py -A '+\
            #         path+'/ODM/odm_dem/dsm.tif -B '+\
            #         path+'/ODM/odm_dem/dtm.tif --outfile='+\
            #         path+'/ODM/odm_dem/dem.tif --calc="A-B" --NoDataValue=-999 --overwrite'
            dem_cmd = 'gdal_calc.py -A '+\
                        path+'/ODM/odm_dem/dsm.tif -B '+\
                        path+'/ODM/odm_dem/dtm.tif --calc="A-B" --type="Float32" --outfile='+\
                        path+'/ODM/odm_dem/dem.tif --overwrite --NoDataValue=-999'
            subprocess.call(dem_cmd, shell=True)
        
        #make a target folder if it doesn't already exist
        if not os.path.exists(path+'/ortho'):
            os.mkdir(path+'/ortho')
        
        #split the RGB bands off into seperate tifs and standardise their outputs
        for band in [1,2,3,4]:
            split_cmd = 'gdal_translate '+\
                        ortho_path+' '+\
                        path+'/ortho/band_'+str(band)+'.tif '+\
                        '-b '+str(band)
            subprocess.call(split_cmd, shell=True)
            
            calc_cmd = 'gdal_calc.py -A '+\
                        path+'/ortho/band_'+str(band)+'.tif '+\
                        '--outfile='+path+'/ortho/band_'+str(band)+'_std.tif '+\
                        '--calc="A/255" --type="Float32" --NoDataValue=-999 --overwrite'
            subprocess.call(calc_cmd, shell=True)
            
        
        #reproject the DEM
        # DEM = gdal.Open(path+'/ODM/odm_dem/odm_orthophoto.tif')
            
        #merge the bands with their relevant DEM file
        merge_cmd = 'gdal_merge.py -o '+path+'/ortho/dRGB.tif '+\
                    '-seperate '+\
                    path+'/ODM/odm_dem/dem.tif '+\
                    path+'/ortho/band_1_std.tif '+\
                    path+'/ortho/band_2_std.tif '+\
                    path+'/ortho/band_3_std.tif '
        subprocess.call(merge_cmd, shell=True)
                
    
    