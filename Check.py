import os
import SimpleITK as sitk
import numpy as np

def get_nrrd_pixel_data(folder_path):
    nrrd_files = [file for file in os.listdir(folder_path) if file.endswith(".nrrd")]

    pixel_data = []
    for file in nrrd_files:
        file_path = os.path.join(folder_path, file)
        try:
            image = sitk.ReadImage(file_path)
            data = sitk.GetArrayFromImage(image)
            pixel_data.append(data)
        except Exception as e:
            print(f"Error reading NRRD file: {file_path}\n{e}")
            continue

    return pixel_data

# Specify the folder path containing the NRRD files
folder_path = "/Users/chu/Downloads/外部验证/Label/ADC"

# Call the function to retrieve pixel data from NRRD files in the folder
nrrd_pixel_data = get_nrrd_pixel_data(folder_path)

# Access and work with the pixel data as needed
for data in nrrd_pixel_data:
    print(data.shape)  # Print the shape of each NRRD file's pixel data
    # Perform further processing or analysis on the pixel data
