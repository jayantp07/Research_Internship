import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import os
from random import shuffle

train_ids=os.listdir("../MED")
train_ids.sort()
training_data=[]

def generator(patient):
    filename1="../MED_ABD_LYMPH_ANNOTATIONS/"+patient+"/"+patient+"_lymphnodes_physicalPoints.txt"
    filename2="../MED_ABD_LYMPH_ANNOTATIONS/"+patient+"/"+patient+"_lymphnodes_sizes.txt"
    filename3="../MED_ABD_LYMPH_ANNOTATIONS/"+patient+"/"+patient+"_lymphnodes_indices.txt"
    cands1=[]
    cands2=[]
    cands3=[]
    for line in open(filename1):
    	allnumber=list(map(float,line.strip('\n').split()))
    	cands1.append(allnumber)
    cands1=np.array(cands1)
    for line in open(filename2):
    	allnumber=list(map(float,line.strip('\n').split()))
    	cands2.append(allnumber)
    cands2=np.array(cands2)
    for line in open(filename3):
    	allnumber=list(map(float,line.strip('\n').split()))
    	cands3.append(allnumber)
    cands3=np.array(cands3)
    return cands1, cands2, cands3


def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

def load_itk(filename):
    # Reads the image using SimpleITK
    filename = "../MHA/"+filename+".mha"
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing


def seq(start, stop, step=1):
    n=int(round((stop-start)/float(step)))
    if(n>1):
        return ([start + step*i for i in range(n+1)])
    else:
        return ([])

def draw_circles(patient, image, cands1, cands2, cands3, origin, spacing, resize_factor):
    cnt=0
    zz=image.shape[0]//2
    yy=image.shape[1]//2
    xx=image.shape[2]//2
    RESIZE_SPACING=[1, 1, 1]
    image_mask=np.zeros(image.shape)

    for i in range(len(cands1)):
        radius=np.ceil(cands2[i][0])/2
        coord_x=cands1[i][0]
        coord_y=cands1[i][1]
        coord_z=cands1[i][2]

        image_coord=np.array((coord_z, coord_y, coord_x))
        image_coord=world_2_voxel(image_coord, origin, spacing)*resize_factor
        noduleRange=seq(-radius, radius, RESIZE_SPACING[0])
        
        for z in noduleRange:
            for y in noduleRange:
                for x in noduleRange:
                    coords=world_2_voxel(np.array((coord_z+z, coord_y+y, coord_x+x)), origin, spacing)*resize_factor
                    if((np.linalg.norm(image_coord-coords)*RESIZE_SPACING[0])<radius):
                        image_mask[int(np.round(coords[0])), int(np.round(coords[1])), int(np.round(coords[2]))]=int(1)
    
    for i in range(len(cands3)):
        coord_z=int(cands3[i][2])
        img_training=image[coord_z, :, :]
        img_training[img_training<-250]=-1000
        img_training[img_training>300]=1000
        img_training=img_training[(yy-128):(yy+128), (xx-128):(xx+128)]
        img_training=resize_image_with_crop_or_pad(img_training, [256, 256], mode='constant', constant_values=-1000)
        img_mask_training=image_mask[coord_z, :, :]
        img_mask_training=img_mask_training[(yy-128):(yy+128), (xx-128):(xx+128)]
        img_mask_training=resize_image_with_crop_or_pad(img_mask_training, [256, 256], mode='constant', constant_values=0)
        flag=0
        for i in range(256):
            for j in range(256):
                if(img_mask_training[i][j]==1):
                    flag=1
                    break
            if(flag==1):
                break

        if(flag==1):
            cnt=cnt+1
            print(cnt)
            training_data.append([np.array(img_training), np.array(img_mask_training)])      

def create_train_data():
    for patient in train_ids:
        RESIZE_SPACING=[1, 1, 1]
        ct_scan, origin, spacing=load_itk(patient)
        img=np.load("../images1/"+patient+".npy")
        resize_factor=spacing/RESIZE_SPACING
        cands1, cands2, cands3=generator(patient)
        print("PROCESSING PATIENT : "+patient)
        print("Please Wait......................")
        print()
        draw_circles(patient, img, cands1, cands2, cands3, origin, spacing, resize_factor)
    shuffle(training_data)
    np.save("./training_data1/training_data.npy", training_data)
        
    
create_train_data()