
import datetime as dtm
import sys, re, glob, os, piexif, subprocess, fiona, itertools
import geopandas as gpd

from osgeo import gdal


##home directory
#HOME = os.path.expanduser('~')
#
#activate KML support for geopandas
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

##set the path to get coordinates from
#path = HOME+'/Documents/Data/Fieldwork/EDS_2019/Mozambique/Matondovela/12_13/rededge/'
## path = HOME+'/Documents/Data/Fieldwork/EDS_2019/botswana/Western_Fence/WF23_24/rededge/'
##path = sys.argv[1]
#
##set the output path for LANDSAT/Sentinel tiles
#sat_sets = glob.glob(HOME+'/Documents/Data/Sat_data/*')
#sat_names = [x.split('/')[-1] for x in sat_sets]
#
##read in overlap shapefile
#ovrlap = gpd.read_file(path+'Overlap.shp')


class SAT():
    
    def __init__(self, sat_paths):
        
        #activate KML support for geopandas
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        
        self.path = sat_paths
        self.sat_paths = glob.glob(sat_paths+'/*')
        self.names = [x.split('/')[-1] for x in self.sat_paths]
        self.LANDSAT_path = glob.glob(sat_paths+'/*LANDSAT*')
        self.Sentinel_path = glob.glob(sat_paths+'/*Sentinel*')
        self.MODIS_path = glob.glob(sat_paths+'/*MODIS*')
    
    
    def img_dates(img_path):
        """
        Finds the raw images taken in a given path and returns the date that these images were taken
        """
        
        #get time and date from exif data
        for root, dirs, files in os.walk(img_path):
            	if 'IMG_0020_1.tif' in files or 'IMG_0015_1.tif' in files:
        	
                    IMG_root = root
        
        IMG_path = glob.glob(IMG_root+'/IMG_00*')[0]
        
        #read exifdata from image
        tags = piexif.load(IMG_path)
        date, time = [''.join(x.split(':')) for x in str(tags['0th'][306]).split("'")[1].split()]
        
        #create a date range		
        dt = dtm.datetime(int(date[:4]), int(date[4:6]), int(date[-2:]))
    
        
        return dt
    
    def convert_to_shp(area_path, allowed_ext = None):
        """
        Converts a .tif file to a .shp polygon file
        """
        
        if allowed_ext == None:
            allowed_ext = ['shp','kml']
        
        #detect if given path is in a permitted format
        ext = area_path.split('.')[-1]
        if ext in allowed_ext:
            pass
        else:
            cmd = 'gdal_polygonize '+area_path+' -b 1 '+'.'.join(area_path.split('.')[:-1])+'.shp'
            subprocess.call(cmd, shell = True)
    
    
    
    def LANDSAT_filename(self, area_path, img_path, img_dates = img_dates, convert_to_shp = convert_to_shp, delta_t = 16):
        """
        Returns the directory of LANDSAT images just before and just after a given date over a certain area.
        """
        #make sure the input area is in the correct format
        convert_to_shp(area_path)
        
        #read in area file
        area = gpd.read_file(area_path)
        
        #read in WRS-2 shapefile
        wrs = gpd.read_file(self.LANDSAT_path+'/WRS2_descending_0/WRS2_descending.shp')
        
        #find the WRS-2 coordinate of the LANDSAT tile
        wrs_intersection = wrs[wrs.intersects(area.geometry[0])]
        paths, rows = wrs_intersection['PATH'].values, wrs_intersection['ROW'].values
        
        scene = '{:03}'.format(paths[0])+'{:03}'.format(rows[0])
        
        #create a date range		
        dt = img_dates(img_path)
        date_diff = dtm.timedelta(delta_t)
        start_dt = dt - date_diff
        end_dt = dt + date_diff
        
        start_date = str(start_dt.year)+'{:02}'.format(start_dt.month)+'{:02}'.format(start_dt.day)
        end_date = str(end_dt.year)+'{:02}'.format(end_dt.month)+'{:02}'.format(end_dt.day)
        
        #initiate loop variables
        LANDSAT_set = glob.glob(self.LANDSAT_path+'/*')
        filename_prefire = 'LC08'+scene+start_date+'01T1'
        filename_postfire = 'LC08'+scene+end_date+'01T1'
        
        i=0
        while filename_prefire not in [x.split('/')[-1].split('-')[0] for x in LANDSAT_set]:
            
            #if no file found for this date, step one day forwards
            start_dt = start_dt + dtm.timedelta(1)
            start_date = str(start_dt.year)+'{:02}'.format(start_dt.month)+'{:02}'.format(start_dt.day)
            
            #new filename
            filename_prefire = 'LC08'+scene+start_date+'01T1'
            
            
            i+=1
            
            if i > 16:
                
                raise FileNotFoundError("Too many loops elapsed - no tile found for given date range. Is the correct scene number present?")
                sys.exit(1)
                
            if (start_dt - dt).days >0:
                
                raise FileNotFoundError("Date range exhausted - no tile found for given date range. Is the correct scene number present?")
                sys.exit(1)
                
        i=0
        while filename_postfire not in [x.split('/')[-1].split('-')[0] for x in LANDSAT_set]:
            
            #if no file found for this date, step one day forwards
            end_dt = end_dt - dtm.timedelta(1)
            end_date = str(end_dt.year)+'{:02}'.format(end_dt.month)+'{:02}'.format(end_dt.day)
            
            #new filename
            filename_postfire = 'LC08'+scene+end_date+'01T1'
            
            
            i+=1
            
            if i > 16:
                
                raise FileNotFoundError("Too many loops elapsed - no tile found for given date range. Is the correct scene number present?")
                sys.exit(1)
                
            if (end_dt - dt).days <=0:
                
                raise FileNotFoundError("Date range exhausted - no tile found for given date range. Is the correct scene number present?")
                sys.exit(1)
        
        
        pre_idx = [x.split('/')[-1].split('-')[0] for x in LANDSAT_set].index(filename_prefire)
        pre_filename = LANDSAT_set[pre_idx]
        
        post_idx = [x.split('/')[-1].split('-')[0] for x in LANDSAT_set].index(filename_postfire)
        post_filename = LANDSAT_set[post_idx]
        
        return pre_filename, post_filename
    
    
    def Sentinel_filename(self, area_path, img_path, img_dates = img_dates, convert_to_shp = convert_to_shp, delta_t = 10):
        """
        Returns the directory of Sentinel-2 images just before and just after a given date over a certain area.
        """        
        #make sure the input area is in the correct format
        convert_to_shp(area_path)
        
        #read in area file
        area = gpd.read_file(area_path)
        
        #get tile info for the scene
        tile_gpd = gpd.read_file(self.Sentinel_path+'/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml', driver = 'KML')
        tile_intersect = tile_gpd[tile_gpd.intersects(area.geometry[0])]
        Tile = tile_intersect['Name'].values[0]
        
        
        #create a date range		
        dt = img_dates(img_path)
        date_diff = dtm.timedelta(delta_t)
        start_dt = dt - date_diff
        end_dt = dt + date_diff
        
        start_date = str(start_dt.year)+'{:02}'.format(start_dt.month)+'{:02}'.format(start_dt.day)
        end_date = str(end_dt.year)+'{:02}'.format(end_dt.month)+'{:02}'.format(end_dt.day)
        
        S_set = glob.glob(self.Sentinel_path+'/*')
        filename_prefire = 'S2A_MSIL2A_'+start_date
        filename_postfire = 'S2A_MSIL2A_'+end_date
        
        i=0
        while filename_prefire not in ['_'.join(x.split('/')[-1].split('_')[:3]).split('T')[0] for x in S_set if Tile in x]:
        
            #if no file found for this date, step one day forwards
            start_dt = start_dt + dtm.timedelta(1)
            start_date = str(start_dt.year)+'{:02}'.format(start_dt.month)+'{:02}'.format(start_dt.day)
    
            #new filename
            filename_prefire = 'S2A_MSIL2A_'+start_date
        
            i+=1
            
            if i > 10:
                
                raise FileNotFoundError("Too many loops elapsed - no tile found for given date range. Is the correct scene number present?")
                sys.exit(1)
                
            if (start_dt - dt).days > 0:
                
                raise FileNotFoundError("Date range exhausted - no tile found for given date range. Is the correct scene number present?")
                sys.exit(1)

        while filename_postfire not in ['_'.join(x.split('/')[-1].split('_')[:3]).split('T')[0] for x in S_set if Tile in x]:
        
            #if no file found for this date, step one day forwards
            end_dt = end_dt - dtm.timedelta(1)
            end_date = str(end_dt.year)+'{:02}'.format(end_dt.month)+'{:02}'.format(end_dt.day)
            
            #new filename
            filename_postfire = 'S2A_MSIL2A_'+end_date   
            
            i+=1
            
            if i > 10:
                
                raise FileNotFoundError("Too many loops elapsed - no tile found for given date range. Is the correct tile number present?")
                sys.exit(1)
                
            if (end_dt - dt).days <= 0:
                
                raise FileNotFoundError("Date range exhausted - no tile found for given date range. Is the correct tile number present?")
                sys.exit(1)
                
        pre_idx = ['_'.join(x.split('/')[-1].split('_')[:3]).split('T')[0] for x in S_set].index(filename_prefire)
        pre_filename = S_set[pre_idx]
        
        post_idx = ['_'.join(x.split('/')[-1].split('_')[:3]).split('T')[0] for x in S_set].index(filename_postfire)
        post_filename = S_set[post_idx]
        
        return pre_filename, post_filename
            
    
    def MODIS_filename(self, area_path, img_path, img_dates = img_dates, convert_to_shp = convert_to_shp):
        """
        Returns the directory of Sentinel-2 images just before and just after a given date over a certain area.
        """ 
        #make sure the input area is in the correct format
        convert_to_shp(area_path)
        
        #read in area file
        area = gpd.read_file(area_path)
        
        #get tile info for the scene
        tile_gpd = gpd.read_file(self.MODIS_path+'/MODIS_tiles.kml', driver = 'KML')
        tile_intersect = tile_gpd[tile_gpd.intersects(area.geometry[0])]
        print(tile_intersect)
        Tile = tile_intersect['Name'].values[0]
        
        
        #create a date range		
        dt = img_dates(img_path)
        date_diff = dtm.timedelta(1)
        start_dt = dt - date_diff
        end_dt = dt + date_diff
        
        start_date = str(start_dt.year)+'{:03}'.format(start_dt.timetuple().tm_yday)
        end_date = str(end_dt.year)+'{:03}'.format(start_dt.timetuple().tm_yday)
        
        M_set = glob.glob(self.MODIS_path+'/MCD64/*')
        filename_prefire = 'MCD64A1.A'+start_date+'.'+Tile+'.006'
        filename_postfire = 'MCD64A1.A'+start_date+'.'+Tile+'.006'
        
        i=0
        while filename_prefire not in ['.'.join(x.split('/')[-1].split('.')[:-2])for x in M_set]:
        
            #if no file found for this date, step one day forwards
            start_dt = start_dt + dtm.timedelta(1)
            start_date = str(start_dt.year)+'{:03}'.format(start_dt.timetuple().tm_yday)
            
            #new filename
            filename_prefire = 'MCD64A1.A'+start_date+'.'+Tile+'.006'
            
            i+=1
            
            if i > 1:
                
                raise FileNotFoundError("Too many loops elapsed - no tile found for given date range. Is the correct scene number present?")
                sys.exit(1)
                
            if (start_dt - dt).days > 0:
                
                raise FileNotFoundError("Date range exhausted - no tile found for given date range. Is the correct scene number present?")
                sys.exit(1)
                
                
        while filename_postfire not in ['.'.join(x.split('/')[-1].split('.')[:-2])for x in M_set]:
        
            #if no file found for this date, step one day forwards
            end_dt = end_dt - dtm.timedelta(1)
            end_date = str(end_dt.year)+'{:03}'.format(start_dt.timetuple().tm_yday)
            
            #new filename
            filename_postfire = 'MCD64A1.A'+end_date+'.'+Tile+'.006'

            i+=1
            
            if i > 1:
                
                raise FileNotFoundError("Too many loops elapsed - no tile found for given date range. Is the correct tile number present?")
                sys.exit(1)
                
            if (end_dt - dt).days <= 0:
                
                raise FileNotFoundError("Date range exhausted - no tile found for given date range. Is the correct tile number present?")
                sys.exit(1)
                
        pre_idx = ['.'.join(x.split('/')[-1].split('.')[:-2])for x in M_set].index(filename_prefire)
        pre_filename = M_set[pre_idx]
        
        post_idx = ['.'.join(x.split('/')[-1].split('.')[:-2])for x in M_set].index(filename_postfire)
        post_filename = M_set[post_idx]
        
        return pre_filename, post_filename
    
    def raster_paths(self, filename, ext = ['jp2','tif', 'hdf'], search = ''):
        """
        Returns a list of all raster files with pre-defined extensions found in a given directory.
        Search term is optional
        """        
        #find the raster files available, including an optional search term
        raster_paths = []
        for root, dirs, files in os.walk(filename):
            rasters = [root+'/'+x for x in files if x.split('.')[-1] in ext and search in root+'/'+x]
            raster_paths.append(rasters)


        return list(itertools.chain.from_iterable(raster_paths))    

    def rasters_as_dict(self,filename, ext = ['jp2','tif','hdf'], search = '', raster_paths = raster_paths):
        
        #list of raster paths available 
        rasterpaths = raster_paths(self,filename, ext = ext, search = search)
        
        #set they keys from the rasters
        keys = ['.'.join(x.split('/')[-1].split('.')[:-1]) for x in rasterpaths]
        
        #set arrays into dict object
        raster_dict = {}
        for key in keys:
            raster_dict[key] = gdal.Open(rasterpaths[keys.index(key)]).ReadAsArray()
            
        return raster_dict
            
            
        
                         
        
        

