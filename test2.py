import numpy as np
import dicom
import os
from random import shuffle
import matplotlib.pyplot as plt
from glob import glob
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from numpy.linalg import inv

output_path="./"
id=0

def load_scan(patient):
    path=glob("./MED/"+patient+"/*/*/*.dcm")
    slices = [dicom.read_file(s) for s in path]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image

patient=load_scan("MED_LYMPH_028")

imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print("Shape after resampling\t", imgs_after_resamp.shape)

def create_train_data(positive_indices, negative_indices, imgs_to_process, slices):
    training_data=[]
    for row in positive_indices:
        x=row[1]
        y=row[2]
        z=row[0]
        VOI=resample(imgs_to_process[z-30:z+30, x-30:x+30, y-30:y+30], slices, [1, 1, 1])
        print(VOI.shape)
        x_center=VOI.shape[1]//2
        y_center=VOI.shape[2]//2
        z_center=VOI.shape[0]//2
        img=[]
        img.append(VOI[0, x_center-16:x_center+16, y_center-16:y_center+16])
        img.append(VOI[z_center-16:z_center+16, 0, y_center-16:y_center+16])
        img.append(VOI[z_center-16:z_center+16, x_center-16:x_center+16, 0])
        img=np.array(img)
        training_data.append([np.array(img),[1,0]])

    for row in negative_indices:
        x=row[1]
        y=row[2]
        z=row[0]
        VOI=resample(imgs_to_process[z-30:z+30, x-30:x+30, y-30:y+30], slices, [1, 1, 1])
        print(VOI.shape)
        x_center=VOI.shape[1]//2
        y_center=VOI.shape[2]//2
        z_center=VOI.shape[0]//2
        img=[]
        img.append(VOI[0, x_center-16:x_center+16, y_center-16:y_center+16])
        img.append(VOI[z_center-16:z_center+16, 0, y_center-16:y_center+16])
        img.append(VOI[z_center-16:z_center+16, x_center-16:x_center+16, 0])
        img=np.array(img)
        training_data.append([np.array(img),[0,1]])
    shuffle(training_data)
    np.save(output_path + "training_%d.npy" % (id), training_data)
    return training_data


