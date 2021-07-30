#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:56:53 2020

@author: tes520
"""


from osgeo import gdal, osr
from glob import glob
import os
import random
# import torch

from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from rsgislib.segmentation import segutils
from rsgislib import rastergis, classification
from rsgislib.classification import classratutils
from rsgislib.rastergis import ratutils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import __TRAINING as TR
from tqdm import tqdm


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
    
def PrePatch(src_file, out_file):
    """
    Creates a new classifier file to account for some stretching of the file
    in later processes
    """
    
    #read in data from source file
    data = gdal.Open(src_file).ReadAsArray()
    
    #create a new dataset
    newdata = np.zeros(data.shape)
    
    #determine whether pre- or post-fire, and update accordingly
    if len(np.unique(data)) == 3:
        newdata[data == 2] = 200
    else:
        newdata[data==2] = 200
        newdata[data==3] = 200
        newdata[data==4] = 200
    
    UpdateGT(out_file, newdata, src_file)
    
    

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

#create a neural network, inheriting from nn.Module
class Net(torch.nn.Module):#, width = full_width, height = full_height, num_classes = len(shp_paths)):
    
    #initialise the neural network
    def __init__(self, bands, num_classes):
        
        #initiate the inherited module
        super().__init__()
        
        #define values for neuron layers
        self.fc1 = torch.nn.Linear(int(bands), 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, int(num_classes))
        
        
    #define how data passes through the NN
    def forward(self, x):
        """
        Set how the data passes through the neural network.
        In this instance, the first 3 layers use the rectified linear activation function.
        It outputs a softmax function, which is essentially a confidence score, adding up to 1.
        Extra steps can be added here for added complexity.
        """
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        
        return torch.nn.functional.log_softmax(x, dim = 1)


class Classifier():
    
    def __init__(self,path):
        
        self.path = path
        self.gdal_img = gdal.Open(path)
        self.asarray = gdal.Open(path).ReadAsArray()
        self.shape = self.asarray.shape
    
    
    def ReadImageArray(self):
        """
        Returns a numpy array of the image
        """
        
        return self.asarray
    
    def RandColour(n):
        """
        Generates a random colour set in RGB format
        """
        #initialise the colour list with blue
        colors = [np.array([0,0,204])]
    
        for i in range(n):
            #generate a new random colour
            newcolor = np.random.choice(256,3)
            
            #check that it is not too similar to any already existing colour
            check = [0,0]
            while len(check) !=0:
                colordiff = colors - newcolor
                diffsum = [np.sum(abs(x)) for x in colordiff]
                check = [x for x in diffsum if x < 150]
                
                newcolor = np.random.choice(256,3)

            
            colors.append(newcolor)
        del colors[0]
            
        return colors

    def GetGeoTransform(self):
        """
        Returns a dictionary object of the geotransform info given by GDAL
        """
        
        ulx, xres, xskew, uly, yskew, yres  = self.gdal_img.GetGeoTransform()
        lrx = ulx + (self.gdal_img.RasterXSize * xres)
        lry = uly + (self.gdal_img.RasterYSize * yres)
        
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
    
    
    
    def UpdateGT(self, out_file, data, src_file = None, epsg = 4326, drv = 'GTiff', datatype = gdal.GDT_Float32, NoData = -999):
        """
        This function takes an aligned raster and updates the GeoLocation metadata
        to match that of the unaligned band
        """
        
        driver = gdal.GetDriverByName(drv)
        if src_file == None:
            src_file = self.path
        
        #source raster
        src_gt = gdal.Open(src_file)
        
        #array details
        [cols, rows] = data.shape
        
        #create the destination file
        dst_gt = driver.Create(out_file, rows, cols, 1, datatype)
        
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
        dst_gt.GetRasterBand(1).WriteArray(data)
        dst_gt.GetRasterBand(1).SetNoDataValue(NoData)
        dst_gt.FlushCache()
        
        

    def Seperability_graphic(self, X_data, Y_data, n_components = 2, labels = None):
        """
        Graphical representation of the seperability of different categories via PCA
        """
        
        #initialise the principal component analysis
        pca = PCA(n_components = n_components)
        
        #set the column headers
        clmn_headers = []
        for i in range(n_components):
            clmn_headers.append('principal component '+str(i+1))
        
        #transform the X_data into its prin
        principalComponents = pca.fit_transform(X_data)
        principalDf = pd.DataFrame(data = principalComponents
             , columns = clmn_headers)
        
        #list of colours to choose from
        colors = ['brown','orange','green','magenta','black', 'red', 'blue', 'purple']
        
        #create a colourmap by assigning a random colour to a given label
        chosen_colors = random.sample(colors, k = max(Y_data)+1)
        cmap = [chosen_colors[x] for x in Y_data]
        
        #create legend if labels are given
        if labels is not None:
            legend = [labels[x] for x in Y_data]
        
        #create figure and add axes
        fig = plt.figure(figsize = (11,11))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principle Component 1', fontsize=15)
        ax.set_ylabel('Principle Component 2', fontsize=15)
        
        #add the explained varience as text to the graphic
        ax.text(-6, -7.5, 'Explained variance of PC1: {:02.3f} %'.format(pca.explained_variance_ratio_[0]*100), fontsize = 12)
        ax.text(-6, -8, 'Explained variance of PC2: {:02.3f} %'.format(pca.explained_variance_ratio_[1]*100), fontsize = 12)
        
        ax.set_title('Seperability of different categories', fontsize = 20)
        
        #ax.set_zlim(-1,1)
        for l in range(len(colors)):
            
            ndx_arr = np.array([i for i, x in enumerate(cmap) if x == colors[l]])
            
            try:
                lbl = np.unique(np.array(legend)[ndx_arr])[0]
                scatter = ax.scatter(principalDf['principal component 1'][ndx_arr], principalDf['principal component 2'][ndx_arr],\
                                 c = np.array(cmap)[ndx_arr], s = 4, label = lbl, alpha = 0.2)
                
            except IndexError:
                pass
        
        legend1 = ax.legend(loc="lower right", title="Classes", fontsize=12, markerscale = 5)
        ax.add_artist(legend1)
        plt.show()
        
        
        
    def PCA_classifier(self, X_data, Y_data, test_size = 1./7., var = .95):
        """
        Train a classifier using a PCA analysis of input training data and returns a predictor for the full image
        """

        #split the data into training and testing data
        train_img, test_img, train_lbl, test_lbl = train_test_split(X_data,Y_data, test_size=test_size, random_state=0)
        
        #let sklearn choose the number of components such that a given percentage of the variance is kept
        pca_train = PCA(var)
        
        pca_train.fit(train_img)
        train_img_pca = pca_train.transform(train_img)
        test_img_pca = pca_train.transform(test_img)
        
        
        #fit to training data
        logisticRegr = LogisticRegression(solver = 'lbfgs')
        logisticRegr.fit(train_img_pca, train_lbl)
        
        #test the predictor
        predictions = []
        for i in test_img_pca:
            predictions.append(logisticRegr.predict(i.reshape(1,-1)))
        
        predictions = np.array([x[0] for x in predictions])
        
        #check the accuracy on the test data set
        correct = 0
        total = 0
        for i in range(len(predictions)):
            
            if predictions[i] == test_lbl[i]:
                
                correct +=1
                
            total += 1
            
        print("Accuracy from PCA fitting on test data is: {:02.3f} %".format(100*correct/total))
        
        return logisticRegr
        
        
    
    def PCA_fit(self, X_data, Y_data, test_size = 1./7., img = None, NoData = -999, var = .95, PCA_classifier = PCA_classifier):
        """
        Classifies an image based on PCA analysis of training data
        """
        
        #let sklearn choose the number of components such that a given percentage of the variance is kept
        pca_train = PCA(var)
        
        #flatten the input array
        if img is not None:
            img_flattened = img.reshape(-1,len(img))

        else:
            img = self.asarray
            img_flattened = img.reshape(-1, len(img))
            
        
        #find the nans & replace them with NoData value
        img_flattened[np.isnan(img_flattened)] = NoData
        
        #PCA transform of full image
        img_flattened = pca_train.transform(img_flattened)
        
        #create the regression classifier
        logisticRegr = PCA_classifier(X_data, Y_data, test_size, var)
        
        #make predictions across each individual pixel
        classifier = []
        for i in tqdm(range(len(img_flattened))):
            
            classifier.append(logisticRegr.predict(img_flattened[i,:].reshape(1,-1)))
            
        #reshape the classifier to match the input image
        classifier = np.array(classifier).reshape(img[0].shape)
        rearranged_classifier = np.zeros(classifier.shape)
        
        #reshape the output image to the original format
        indx_count = []
        for k in (range(len(classifier))):
            x_indx = (k*8)%len(classifier)+1
            
            if x_indx in indx_count:
                
                x_indx -=1
            
            if x_indx >= len(classifier):
                
                x_indx = 0
            
            indx_count.append((k*8)%len(classifier))
            
            for j in range(len(classifier[0])):
            
                
                rearranged_classifier[x_indx,(j*8)%len(classifier[0])] = classifier[k,j]
        
        
        return rearranged_classifier
    
    
    def OBIA(self, min_pixels = 75, nClusters = 12, band_names = None, RandColour = RandColour, UpdateGT = UpdateGT, training_path = None):
        """
        Performs Object-based classfication using given training data
        """
        
        #use the KEA format for this process
        frmt = 'KEA'
        driver = gdal.GetDriverByName(frmt)
        
        #gdal object for use in translate functions
        img = self.gdal_img
        
        #make a directory to store all the temporary files created
        if not os.path.exists('/'.join(self.path.split('/')[:-2])+'/Classification/OBIA'):
            os.mkdir('/'.join(self.path.split('/')[:-2])+'/Classification/OBIA')
            
        #path to temporary KEA files
        outputImg_path = '/'.join(self.path.split('/')[:-2])+'/Classification/OBIA/img.kea'
        clumpsImg_path = '/'.join(self.path.split('/')[:-2])+'/Classification/OBIA/clumps.kea'
        
    
        #convert the input image to KEA format
        img_kea = gdal.Translate(outputImg_path, img)
        
        #run the segmentation
        segs = segutils.runShepherdSegmentation(outputImg_path,
                                         clumpsImg_path,
                                         numClusters=nClusters, minPxls=min_pixels, distThres=100,
                                         sampling=100, kmMaxIter=200, 
                                         gdalformat = 'KEA')
        
        #find names of the bands
        if band_names is None:
            if len(self.asarray) == 8:
                band_names = ['NDVI','BAI','DEM','RedEdge','NIR','Red','Green','Blue']
            elif len(self.asarray) == 7:
                band_names = ['SAM','dBAI','RedEdge','NIR','Red','Green','Blue']
            elif len(self.asarray) == 6:
                band_names = ['SAM','dBAI','RedEdge','NIR','Red','Green']
            elif len(self.asarray) == 4:
                band_names = ['DEM', 'Red', 'Green', 'Blue']
            else:
                raise IndexError("Please provide a list of the band names")
                
        #add statistics from the image to segmentation
        stats = []
        for i in range(len(band_names)):
            stats.append(rastergis.BandAttStats(band=i+1, meanField=band_names[i]))
            
        rastergis.populateRATWithStats(self.path, clumpsImg_path, stats)
        
        #Training data location
        if training_path == None:
            if 'post-fire' in self.path:
                if 'Matrice' in self.path:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('post-fire')+1])+'/Matrice'
                elif 'Mavic' in self.path:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('post-fire')+1])+'/Mavic'
                else:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('post-fire')+1])
            elif 'pre-fire' in self.path:
                if 'Matrice' in self.path:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('pre-fire')+1])+'/Matrice'
                elif 'Mavic' in self.path:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('pre-fire')+1])+'/Mavic'
                else:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('pre-fire')+1])
            else:
                rootpath = '/'.join(self.path.split('/')[:-2])
            for root, dirs, files in os.walk(rootpath):
                if 'Training' in dirs:
                    rt = root
            training_paths = glob(rt+'/Training/*.shp')
        else:
            training_paths = glob(training_path+'/*.shp')
        
        #re-arrange into training dictionary object
        classDict = {}
        for i in range(len(training_paths)):
            classDict[training_paths[i].split('/')[-1]] = [i+1, training_paths[i]]
            
        #populate the RAT with training data
        tmpPath = '/'.join(self.path.split('/')[:-2])+'/Classification/OBIA'
        classesIntColIn = 'ClassInt'
        classesNameCol = 'ClassStr'
        ratutils.populateClumpsWithClassTraining(clumpsImg_path, classDict, tmpPath,
                                                  classesIntColIn, classesNameCol)
        
        #balance training samples across classes
        classesIntCol = 'ClassIntSamp' 
        classratutils.balanceSampleTrainingRandom(clumpsImg_path, classesIntColIn,
                                               classesIntCol, 10, 500)
        
        #set bands as variables to be used for the classifier
        classParameters = {'n_estimators':[10,100,500], 'max_features':[2,3,4]}
        gSearch = GridSearchCV(ExtraTreesClassifier(), classParameters)
        preProcessing = None
        classifier = classratutils.findClassifierParameters(clumpsImg_path, classesIntCol, band_names, preProcessor=None, gridSearch=gSearch)
        
        #set random colours for the classes
        colors = RandColour(len(training_paths))
        classColours = {}
        for i in range(len(training_paths)):
            key = list(classDict.keys())[i]
            classColours[key] = colors[i]
        
        #classify within the RAT
        outClassIntCol = 'OutClass'
        outClassStrCol = 'OutClassName'
        classratutils.classifyWithinRAT(clumpsImg_path, classesIntCol, classesNameCol, band_names,
                                    classifier=classifier, outColInt=outClassIntCol,
                                    outColStr=outClassStrCol, classColours=classColours,
                                    preProcessor=preProcessing)
        
        # Export to a 'classification image' rather than a RAT...
        outClassImg = '/'.join(self.path.split('/')[:-2])+'/Classification/OBIA/OBIA_classifier.kea'
        classification.collapseClasses(clumpsImg_path, outClassImg, 'KEA', 'OutClassName', 'OutClass')
        
        #create an image to use as a georeference
        gdal.Translate('/'.join(self.path.split('/')[:-2])+'/Classification/OBIA/OBIA_georeference.tif', outClassImg, outputType = gdal.GDT_Byte)
        
        #re-read in the created raster
        kea_raster = gdal.Open(outClassImg).ReadAsArray()
        
        #create a georeferenced .tif version
        UpdateGT(self,'/'.join(self.path.split('/')[:-2])+'/Classification/OBIA/OBIA_classifier.tif',\
                 kea_raster,\
                 src_file = '/'.join(self.path.split('/')[:-2])+'/Classification/OBIA/OBIA_georeference.tif')
        
    
    
    
    def NN_classifier(self, training_path, Net = Net, BATCH_SIZE = 100, Epochs = 10, learning_rate = 0.01, test_size = 1./7., overwrite = False, balance = True):
        """
        Classify an image using a neural network
        """
        
        print('\nPreparing training data\n')
        #initiate training data
        Training = TR.Training(training_path)
        Training.Match_Extents(self.path, overwrite = overwrite)
        X,Y = Training.split_train_test(balance = balance) 
        
        #split data into training and testing
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
        
        #initialise the NN
        bands, width, height = self.shape
        net = Net(bands = bands, num_classes = len(Training.classes))
        
        #set optimiser & loss function
        optimiser= torch.optim.Adam(net.parameters(), lr = learning_rate)
        loss_fn = torch.nn.MSELoss()
        
        print('\nTraining Neural Network...\n')
        for epoch in range(Epochs):
            
            print('\nRunning epoch', epoch)
            
            for i in (range(0, len(X_train), BATCH_SIZE)):
                
                batch_X = torch.Tensor(X_train[i:i+BATCH_SIZE])
                batch_Y = torch.Tensor(Y_train[i:i+BATCH_SIZE])
                
                net.zero_grad()
                
                outputs = net(batch_X)
                
                loss = loss_fn(outputs, batch_Y)
                loss.backward()
                optimiser.step()
                
        print('\n',loss)
        
        correct = 0
        total = 0
        
        Y_test = torch.Tensor(Y_test)
        X_test = torch.Tensor(X_test)
        
        with torch.no_grad():
            for i in (range(len(X_test))):
                
                real_class = torch.argmax(Y_test[i])
                net_out = net(X_test[i].view(-1,bands))[0]
                pred_class = torch.argmax(net_out)
                
                if pred_class == real_class:
                    correct +=1
                    
                total +=1
                
        print('\nAccuracy: ',round(correct/total,3))
        
        full_img_tensor = torch.Tensor(self.asarray)
        NN_classifier = np.zeros(np.array(full_img_tensor[0].size()))
        with torch.no_grad():
        
            for i in tqdm(range(len(full_img_tensor[0]))):
                for j in range(len(full_img_tensor[0,0])):
                
                    net_out = net(full_img_tensor[:,i,j].view(-1,bands))[0]
                    pred_class = torch.argmax(net_out).item()
                    
                    NN_classifier[i,j] = pred_class
                    
        return NN_classifier
    
    
    def RandomForest(self, training_path = None, n_estimators = 100, overwrite = False, test_size = 1./7.):
        
        
        print('\nPreparing training data\n')
        #Training data location
        if training_path == None:
            if 'post-fire' in self.path:
                if 'Matrice' in self.path:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('post-fire')+1])+'/Matrice'
                elif 'Mavic' in self.path:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('post-fire')+1])+'/Mavic'
                else:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('post-fire')+1])
            elif 'pre-fire' in self.path:
                if 'Matrice' in self.path:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('pre-fire')+1])+'/Matrice'
                elif 'Mavic' in self.path:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('pre-fire')+1])+'/Mavic'
                else:
                    rootpath = '/'.join(self.path.split('/')[:self.path.split('/').index('pre-fire')+1])
            else:
                rootpath = '/'.join(self.path.split('/')[:-2])
            for root, dirs, files in os.walk(rootpath):
                if 'Training' in dirs:
                    rt = root
            training_paths = rt+'/Training/'
        else:
            training_paths = training_path
        #initiate training data split
        Training = TR.Training(training_paths)
        Training.Match_Extents(self.path, overwrite = overwrite)
        X,Y = Training.split_train_test()
        
        #split data into training and testing
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
        
        print(Y_train)
        
        # un one-hot the labels
        # Y_train = np.array([np.where(x==1)[0] for x in Y_train])
        # Y_test = np.array([np.where(x==1)[0] for x in Y_test])
        
        print(Y_train)
        
        #initialise random forest classifier
        clf = RandomForestClassifier(n_estimators = n_estimators, random_state=42)
        
        #train the model
        clf.fit(X_train, Y_train)
        
        #make predictions
        y_pred = clf.predict(X_test)
        
        #accuracy
        print('Accuracy of Random Forest is: ',metrics.accuracy_score(Y_test, y_pred))
        
        #apply the model to the whole image
        img_orig = self.asarray
        # img_orig = gdal.Open(path).ReadAsArray()
        img = img_orig.reshape(len(img_orig), -1)
        img = np.swapaxes(img, 0,1)
        
        RF_classifier = clf.predict(img)
        
        # #un-onehot the array
        # RF = np.zeros(RF_classifier.shape[0])
        # for i in tqdm(range(len(RF))):
            
        #     if 1 in RF_classifier[i]:
                
        #         RF[i] = np.where(RF_classifier[i] == 1)[0][0]
        #     else:
        #         continue
            
        # RF = np.array([np.where(x==1)[0]+1 for x in RF_classifier])

        RF = RF_classifier.reshape(img_orig[0].shape)
        
        if not os.path.exists('/'.join(self.path.split('/')[:-2])+'/Classification/RF'):
            os.mkdir('/'.join(self.path.split('/')[:-2])+'/Classification/RF')
        
        UpdateGT('/'.join(self.path.split('/')[:-2])+'/Classification/RF/RF_classifier_2.tif',\
                 RF,\
                 src_file = self.path)
        
        # return RF_classifier
                
                
        
        

        
        
        
            
    
    
    

        
    
            
        
        
        
        
        

    
    