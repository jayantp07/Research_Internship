import numpy as np
import SimpleITK as sitk
from glob import glob
import os

def mha(name):
    print("Processing Patient : "+name)
    print()
    print()
    data_directory=glob("./MED/"+name+"/*/*/")[0]
    OUTPUT_DIR="./MHA2"
    reader=sitk.ImageSeriesReader()
    original_image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(data_directory))
    original_image1 = sitk.ReadImage("./MHA/"+name+".mha")

    # Write the image.
    output_file_name_3D = os.path.join(OUTPUT_DIR, name+'.mha')
    sitk.WriteImage(original_image, output_file_name_3D)

    # Read it back again.
    written_image = sitk.ReadImage(output_file_name_3D)

    # Check that the original and written image are the same.
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(original_image - written_image)

    statistics_image_filter1 = sitk.StatisticsImageFilter()
    statistics_image_filter1.Execute(original_image1 - written_image)

    # Check that the original and written files are the same
    print('Max, Min differences are : {0}, {1}'.format(statistics_image_filter.GetMaximum(), statistics_image_filter.GetMinimum()))
    print('Max, Min differences are : {0}, {1}'.format(statistics_image_filter1.GetMaximum(), statistics_image_filter1.GetMinimum()))


ids=os.listdir("./MED")
ids.sort()

for patient in ids:
    mha(patient)