import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
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
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
from dltk.io.augmentation import *
from dltk.io.preprocessing import *



training_data_output_path="./DATA_FINAL/2.5D-I/train/"
testing_data_output_path="./DATA_FINAL/2.5D-I/test/"
ids=os.listdir("./MED")
ids.sort()

def flip1(imagelist, axis=1):
    """Randomly flip spatial dimensions

    Args:
        imagelist (np.ndarray or list or tuple): image(s) to be flipped
        axis (int): axis along which to flip the images

    Returns:
        np.ndarray or list or tuple: same as imagelist but randomly flipped
            along axis
    """

    # Check if a single image or a list of images has been passed
    was_singular = False
    if isinstance(imagelist, np.ndarray):
        imagelist = [imagelist]
        was_singular = True

    # With a probility of 0.8 flip the image(s) across `axis`
    do_flip = np.random.random(1)
    if do_flip > 0.2:
        for i in range(len(imagelist)):
            imagelist[i] = np.flip(imagelist[i], axis=axis)
    if was_singular:
        return imagelist[0]
    return imagelist

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, 
origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


'''def load_scan(patient):
    path=glob("./MED/"+patient+"/*/*/*.dcm")
    slices = [dicom.read_file(s) for s in path]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices'''



'''def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)'''



'''def file_sum(patient):
	filename1="./MED_ABD_LYMPH_CANDIDATES/"+patient+"/"+patient+"_posCADe_physicalPoints.txt"
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
	return positive_indices, negative_indices'''



'''def indices(slices, points):
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
	return hope'''



'''def resample(image, slices, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([slices[0].SliceThickness] + slices[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image'''


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

def generator(patient):
    filename1="./MED_ABD_LYMPH_ANNOTATIONS/"+patient+"/"+patient+"_lymphnodes_physicalPoints.txt"
    filename2="./MED_ABD_LYMPH_CANDIDATES/"+patient+"/"+patient+"_negCADe_physicalPoints.txt"
    world_pos=[]
    world_neg=[]
    for line in open(filename1):
    	allnumber=list(map(float,line.strip('\n').split()))
    	world_pos.append(allnumber)
    world_pos=np.array(world_pos)
    for line in open(filename2):
    	allnumber=list(map(float,line.strip('\n').split()))
    	world_neg.append(allnumber)
    world_neg=np.array(world_neg)
    return world_pos, world_neg



def create_train_data(patient):
    training_data=[]
    img, origin, spacing=load_itk("./MHA/"+patient+".mha")
    RESIZE_SPACING=[1, 1, 1]
    resize_factor=spacing/RESIZE_SPACING
    new_real_shape=img.shape*resize_factor
    new_shape=np.round(new_real_shape)
    real_resize=new_shape/img.shape
    new_spacing=spacing/real_resize
    lung_img=np.load("./images1/"+patient+".npy")
    world_pos, world_neg=generator(patient)
    for i in range(world_pos.shape[0]):
        voxel=world_2_voxel(world_pos[i, :][::-1], origin, spacing)
        voxel=np.round(voxel).astype('int')
        voxel=np.round(voxel*resize_factor).astype('int')
        z=voxel[0]
        y=voxel[1]
        x=voxel[2]
        t1_slice = lung_img[z-16:z+16, y-16:y+16, x-16:x+16].copy()
        t1_slice = resize_image_with_crop_or_pad(t1_slice, [32, 32, 32], mode='symmetric')
        print("Processing Positive ", patient)
        print(t1_slice.shape, z, y, x)

        # Add a feature dimension and normalise
        t1_norm = np.expand_dims(normalise_one_one(t1_slice), axis=-1)

        # Randomly flip the image along axis 1
        t1_flipped1 = flip1(t1_norm.copy(), axis=2)

        # Add a Gaussian offset (independently for each channel)
        t1_offset1 = add_gaussian_offset(t1_norm.copy(), sigma=0.1)
        t1_offset2 = add_gaussian_offset(t1_norm.copy(), sigma=0.5)

        # Add Gaussian noise
        t1_noise1 = add_gaussian_noise(t1_norm.copy(), sigma=0.02)
        t1_noise2 = add_gaussian_noise(t1_norm.copy(), sigma=0.05)

        # Elastic transforms according to:
        # [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        #     Neural Networks applied to Visual Document Analysis", in Proc. of the
        #     International Conference on Document Analysis and Recognition, 2003.
        t1_trans_low_s = elastic_transform(t1_norm.copy(), alpha=[1, 1e4, 1e4], sigma=[1, 8, 8])
        t1_trans_high_s = elastic_transform(t1_norm.copy(), alpha=[1, 6e4, 6e4], sigma=[1, 16, 16])


        c1_1=t1_norm[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_norm[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_norm[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_flipped1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_flipped1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_flipped1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_offset1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_offset1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_offset1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_offset2[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_offset2[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_offset2[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_noise1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_noise1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_noise1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_noise2[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_noise2[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_noise2[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_trans_low_s[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_trans_low_s[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_trans_low_s[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_trans_high_s[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_trans_high_s[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_trans_high_s[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([0, 1])])

    util=world_neg.shape[0]
    if(world_neg.shape[0]>(2*world_pos.shape[0])):
        util=2*world_pos.shape[0]

    for i in range(util):
        voxel=world_2_voxel(world_neg[i, :][::-1], origin, spacing)
        voxel=np.round(voxel).astype('int')
        voxel=np.round(voxel*resize_factor).astype('int')
        z=voxel[0]
        y=voxel[1]
        x=voxel[2]
        t1_slice = lung_img[z-16:z+16, y-16:y+16, x-16:x+16].copy()
        t1_slice = resize_image_with_crop_or_pad(t1_slice, [32, 32, 32], mode='symmetric')
        print("Processing Negative", patient)
        print(t1_slice.shape, z, y, x)

        # Add a feature dimension and normalise
        t1_norm = np.expand_dims(normalise_one_one(t1_slice), axis=-1)

        # Randomly flip the image along axis 1
        t1_flipped1 = flip1(t1_norm.copy(), axis=2)

        # Add a Gaussian offset (independently for each channel)
        t1_offset1 = add_gaussian_offset(t1_norm.copy(), sigma=0.1)

        # Add Gaussian noise
        t1_noise1 = add_gaussian_noise(t1_norm.copy(), sigma=0.02)

        # Elastic transforms according to:
        # [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        #     Neural Networks applied to Visual Document Analysis", in Proc. of the
        #     International Conference on Document Analysis and Recognition, 2003.
        t1_trans_low_s = elastic_transform(t1_norm.copy(), alpha=[1, 1e4, 1e4], sigma=[1, 8, 8])

        c1_1=t1_norm[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_norm[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_norm[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([1, 0])])
        c1_1=t1_flipped1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_flipped1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_flipped1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([1, 0])])
        c1_1=t1_offset1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_offset1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_offset1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([1, 0])])
        c1_1=t1_noise1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_noise1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_noise1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([1, 0])])
        c1_1=t1_trans_low_s[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_trans_low_s[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_trans_low_s[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        training_data.append([np.array(img), np.array([1, 0])])

        print("#############################")
        print()
        
    shuffle(training_data)
    np.save(training_data_output_path+patient+".npy", training_data)
    X = np.array([i[0] for i in training_data]).reshape(-1,32,32,3)
    Y = np.array([i[1] for i in training_data])
    print(X.shape, Y.shape)

       
  

def create_test_data(patient):
    testing_data=[]
    img, origin, spacing=load_itk("./MHA/"+patient+".mha")
    RESIZE_SPACING=[1, 1, 1]
    resize_factor=spacing/RESIZE_SPACING
    new_real_shape=img.shape*resize_factor
    new_shape=np.round(new_real_shape)
    real_resize=new_shape/img.shape
    new_spacing=spacing/real_resize
    lung_img=np.load("./images1/"+patient+".npy")
    world_pos, world_neg=generator(patient)
    for i in range(world_pos.shape[0]):
        voxel=world_2_voxel(world_pos[i, :][::-1], origin, spacing)
        voxel=np.round(voxel).astype('int')
        voxel=np.round(voxel*resize_factor).astype('int')
        z=voxel[0]
        y=voxel[1]
        x=voxel[2]
        t1_slice = lung_img[z-16:z+16, y-16:y+16, x-16:x+16].copy()
        t1_slice = resize_image_with_crop_or_pad(t1_slice, [32, 32, 32], mode='symmetric')
        print("Processing Positive ", patient)
        print(t1_slice.shape, z, y, x)

        # Add a feature dimension and normalise
        t1_norm = np.expand_dims(normalise_one_one(t1_slice), axis=-1)

        # Randomly flip the image along axis 1
        t1_flipped1 = flip1(t1_norm.copy(), axis=2)

        # Add a Gaussian offset (independently for each channel)
        t1_offset1 = add_gaussian_offset(t1_norm.copy(), sigma=0.1)
        t1_offset2 = add_gaussian_offset(t1_norm.copy(), sigma=0.5)

        # Add Gaussian noise
        t1_noise1 = add_gaussian_noise(t1_norm.copy(), sigma=0.02)
        t1_noise2 = add_gaussian_noise(t1_norm.copy(), sigma=0.05)

        # Elastic transforms according to:
        # [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        #     Neural Networks applied to Visual Document Analysis", in Proc. of the
        #     International Conference on Document Analysis and Recognition, 2003.
        t1_trans_low_s = elastic_transform(t1_norm.copy(), alpha=[1, 1e4, 1e4], sigma=[1, 8, 8])
        t1_trans_high_s = elastic_transform(t1_norm.copy(), alpha=[1, 6e4, 6e4], sigma=[1, 16, 16])

        c1_1=t1_norm[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_norm[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_norm[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_flipped1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_flipped1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_flipped1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_offset1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_offset1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_offset1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_offset2[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_offset2[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_offset2[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_noise1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_noise1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_noise1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_noise2[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_noise2[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_noise2[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_trans_low_s[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_trans_low_s[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_trans_low_s[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([0, 1])])
        c1_1=t1_trans_high_s[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_trans_high_s[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_trans_high_s[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([0, 1])])

    util=world_neg.shape[0]
    if(world_neg.shape[0]>(2*world_pos.shape[0])):
        util=2*world_pos.shape[0]

    for i in range(util):
        voxel=world_2_voxel(world_neg[i, :][::-1], origin, spacing)
        voxel=np.round(voxel).astype('int')
        voxel=np.round(voxel*resize_factor).astype('int')
        z=voxel[0]
        y=voxel[1]
        x=voxel[2]
        t1_slice = lung_img[z-16:z+16, y-16:y+16, x-16:x+16].copy()
        t1_slice = resize_image_with_crop_or_pad(t1_slice, [32, 32, 32], mode='symmetric')
        print("Processing Negative", patient)
        print(t1_slice.shape, z, y, x)

        # Add a feature dimension and normalise
        t1_norm = np.expand_dims(normalise_one_one(t1_slice), axis=-1)

        # Randomly flip the image along axis 1
        t1_flipped1 = flip1(t1_norm.copy(), axis=2)

        # Add a Gaussian offset (independently for each channel)
        t1_offset1 = add_gaussian_offset(t1_norm.copy(), sigma=0.1)

        # Add Gaussian noise
        t1_noise1 = add_gaussian_noise(t1_norm.copy(), sigma=0.02)

        # Elastic transforms according to:
        # [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        #     Neural Networks applied to Visual Document Analysis", in Proc. of the
        #     International Conference on Document Analysis and Recognition, 2003.
        t1_trans_low_s = elastic_transform(t1_norm.copy(), alpha=[1, 1e4, 1e4], sigma=[1, 8, 8])

        c1_1=t1_norm[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_norm[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_norm[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([1, 0])])
        c1_1=t1_flipped1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_flipped1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_flipped1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([1, 0])])
        c1_1=t1_offset1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_offset1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_offset1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([1, 0])])
        c1_1=t1_noise1[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_noise1[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_noise1[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([1, 0])])
        c1_1=t1_trans_low_s[15, :, :, :]
        print(c1_1.shape)
        c2_1=t1_trans_low_s[:, 15, :, :]
        print(c2_1.shape)
        c3_1=t1_trans_low_s[:, :, 15, :]
        print(c3_1.shape)
        img=np.concatenate((c1_1, c2_1, c3_1), axis=2)
        testing_data.append([np.array(img), np.array([1, 0])])

        print("#############################")
        print()

    shuffle(testing_data)
    np.save(testing_data_output_path+patient+".npy", testing_data)
    X = np.array([i[0] for i in testing_data]).reshape(-1,32,32,3)
    Y = np.array([i[1] for i in testing_data])
    print(X.shape, Y.shape)

       

for patient in ids:
    create_train_data(patient)



for patient in ids:
    create_test_data(patient)
