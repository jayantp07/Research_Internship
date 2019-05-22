import numpy as np
import dicom
import os
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
from random import shuffle

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

def file_sum(patient):
	filename1="./MED_ABD_LYMPH_ANNOTATIONS/"+patient+"/"+patient+"_lymphnodes_physicalPoints.txt"
	filename2="./MED_ABD_LYMPH_CANDIDATES/"+patient+"/"+patient+"_negCADe_physicalPoints.txt"
	positive=[]
	negative=[]
	for line in open(filename1):
		allnumber=list(map(float,line.strip('\n').split()))
		positive.append(allnumber)
	positive=np.array(positive)
	for line in open(filename2):
		allnumber=list(map(float,line.strip('\n').split()))
		negative.append(allnumber)
	negative=np.array(negative)
	positive_indices=[]
	negative_indices=[]
	for row in positive:
		positive_indices.append(indices(slices, row)[0:3].transpose())
	positive_indices=np.array(np.ceil(positive_indices), dtype=int)

	for row in negative:
		negative_indices.append(indices(slices, row)[0:3].transpose())
	negative_indices=np.array(np.ceil(negative_indices), dtype=int)
	positive_indices=np.reshape(positive_indices, (positive_indices.shape[0], 3))
	negative_indices=np.reshape(negative_indices, (negative_indices.shape[0], 3))
	return positive_indices, negative_indices

def indices(slices, points):
	A=np.zeros((4, 4))
	T1=np.array(slices[0].ImagePositionPatient)
	TN=np.array(slices[len(slices)-1].ImagePositionPatient)
	TN=(T1-TN)/(1-len(slices))
	A[1, 0]=(slices[0].PixelSpacing)[0]
	A[0, 1]=(slices[0].PixelSpacing)[1]
	A[0, 2]=TN[0]
	A[1, 2]=TN[1]
	A[2, 2]=TN[2]
	A[0, 3]=T1[0]
	A[1, 3]=T1[1]
	A[2, 3]=T1[2]
	A[3, 3]=1
	A_inverse=inv(A)
	P=np.array([[points[0]], [points[1]], [points[2]], [1]])
	hope=np.dot(A_inverse, P)
	return hope

output_path="./"
id=0
slices=load_scan("MED_LYMPH_028")
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


imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
positive_indices, negative_indices=file_sum("MED_LYMPH_028")

#TEST CODE
'''for row in positive_indices:   
	print(row[0], row[1], row[2])
x=positive_indices[0, 0]
y=positive_indices[0, 1]
z=positive_indices[0, 2]
print(x, y, z)
img=imgs_to_process[z-30:z+30, x-30:x+30, y-30:y+30]
print(img.shape)
VOI=resample(img, slices)
print(VOI.shape)
x_center=VOI.shape[1]//2
y_center=VOI.shape[2]//2
z_center=VOI.shape[0]//2
c1=VOI[0, x_center-16:x_center+16, y_center-16:y_center+16]
c1=c1[:, :, np.newaxis]
c2=VOI[z_center-16:z_center+16, 0, y_center-16:y_center+16]
c2=c2[:, :, np.newaxis]
c3=VOI[z_center-16:z_center+16, x_center-16:x_center+16, 0]
c3=c3[:, :, np.newaxis]
img=np.concatenate((c1, c2, c3), axis=2)
print(img.shape)'''


def create_train_data(positive_indices, negative_indices, imgs_to_process, slices):
    training_data=[]
    for row in positive_indices:
        x=row[0]
        y=row[1]
        z=row[2]
        VOI=resample(imgs_to_process[z-30:z+30, x-30:x+30, y-30:y+30], slices)
        x_center=VOI.shape[1]//2
        y_center=VOI.shape[2]//2
        z_center=VOI.shape[0]//2
        c1=VOI[0, x_center-16:x_center+16, y_center-16:y_center+16]
        c1=c1[:, :, np.newaxis]
        c2=VOI[z_center-16:z_center+16, 0, y_center-16:y_center+16]
        c2=c2[:, :, np.newaxis]
        c3=VOI[z_center-16:z_center+16, x_center-16:x_center+16, 0]
        c3=c3[:, :, np.newaxis]
        img=np.concatenate((c1, c2, c3), axis=2)
        training_data.append([img, np.array([1, 0])])

    for row in negative_indices:
        x=row[0]
        y=row[1]
        z=row[2]
        VOI=resample(imgs_to_process[z-30:z+30, x-30:x+30, y-30:y+30], slices, [1, 1, 1])
        x_center=VOI.shape[1]//2
        y_center=VOI.shape[2]//2
        z_center=VOI.shape[0]//2
        c1=VOI[0, x_center-16:x_center+16, y_center-16:y_center+16]
        c1=c1[:, :, np.newaxis]
        c2=VOI[z_center-16:z_center+16, 0, y_center-16:y_center+16]
        c2=c2[:, :, np.newaxis]
        c3=VOI[z_center-16:z_center+16, x_center-16:x_center+16, 0]
        c3=c3[:, :, np.newaxis]
        img=np.concatenate((c1, c2, c3), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])
	
	
    np.save(output_path + "training_%d.npy" % (id), training_data)
    shuffle(training_data)
    return training_data

train=create_train_data(positive_indices, negative_indices, imgs_to_process, slices)
X = np.array([i[0] for i in train]).reshape(-1,32,32,3)
Y = np.array([i[1] for i in train])
print(X.shape, Y.shape)
