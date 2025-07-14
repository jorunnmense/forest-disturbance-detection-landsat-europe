#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:47:12 2023

@author: aviana
"""

# Script to calculate spectral indices from landsat data cube
# see FORCE documentation for further details on the datacube formats and nomenclature: https://force-eo.readthedocs.io/en/latest/index.html
# to save space Int16 to Tasseled Cap and only use float32 for NDVI-NBR (but scale to 10,000 is recomended)


from osgeo import gdal
import os
import numpy as np
import time


def arrays_to_raster(arrays, cols, rows, transform, projection, output_name):
    """Converts list of NumPy arrays to a multi-band raster file, overwriting existing file if it exists."""
    bands = len(arrays)
    driver = gdal.GetDriverByName('GTiff')

    # Check if the output file already exists, and delete it if it does
    if os.path.exists(output_name):
        os.remove(output_name)

    dataset_index = driver.Create(output_name, cols, rows, bands, gdal.GDT_Float32)
    dataset_index.SetGeoTransform(transform)
    dataset_index.SetProjection(projection)
    
    for i, array in enumerate(arrays, start=1):
        band = dataset_index.GetRasterBand(i)
        band.WriteArray(array)
    dataset_index.FlushCache()

    return dataset_index

def calculate_indices(dsimage):
    """Calculates various spectral indices from raster bands."""
    # int und 6 Landsat channels
    array1 = dsimage.GetRasterBand(1).ReadAsArray().astype(float)  # Blue
    array2 = dsimage.GetRasterBand(2).ReadAsArray().astype(float)  # Green
    array3 = dsimage.GetRasterBand(3).ReadAsArray().astype(float)  # Red
    array4 = dsimage.GetRasterBand(4).ReadAsArray().astype(float)  # NIR
    array5 = dsimage.GetRasterBand(5).ReadAsArray().astype(float)  # SWIR1
    array6 = dsimage.GetRasterBand(6).ReadAsArray().astype(float)  # SWIR2
    
    # Calculate indices and tasseled cap 
    # TC coefficients from http://dx.doi.org/10.1080/2150704X.2014.915434
    ndvi = (array4 - array3) / (array4 + array3)
    nbr = (array4 - array6) / (array4 + array6)
    tcb = 0.3029 * array1 + 0.2786 * array2 + 0.4733 * array3 + 0.5599 * array4 + 0.508 * array5 + 0.1872 * array6
    tcg = -0.2941 * array1 - 0.243 * array2 - 0.5424 * array3 + 0.7276 * array4 + 0.0713 * array5 - 0.1608 * array6
    tcw = 0.1511 * array1 + 0.1973 * array2 + 0.3283 * array3 + 0.3407 * array4 - 0.7117 * array5 - 0.4559 * array6
    tc_di = tcb - (tcg + tcw) # Disturbance index: (Healey et al., 2005)
    
    # Normalized indices
    # mean und std von allen Dateien (!)
    tcb_n = (tcb - np.mean(tcb)) / np.std(tcb)
    tcg_n = (tcg - np.mean(tcg)) / np.std(tcg)
    tcw_n = (tcw - np.mean(tcw)) / np.std(tcw)
    tc_di_n = tcb_n - (tcg_n + tcw_n)               

    return {
        "NBR": nbr,
        "NDVI": ndvi,
        "TCB": tcb,
        "TCG": tcg,
        "TCW": tcw,
        #"TC_DI": tc_di,
        "TC_DI_N": tc_di_n
    }



def arrays_to_raster(arrays, cols, rows, transform, projection, output_name):
    """ Converts list of NumPy arrays to a multi-band raster file. """
    bands = len(arrays)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_name, cols, rows, bands, gdal.GDT_Float32)
    dataset.SetGeoTransform(transform)
    dataset.SetProjection(projection)
    
    for i, array in enumerate(arrays, start=1):
        band = dataset.GetRasterBand(i)
        band.WriteArray(array)
    dataset.FlushCache()

    return dataset

def process_images_in_folder(worksp):
    """ Process images in the workspace, saving results year by year, skipping the first four bands. """
    gdal.AllRegister()
    start_time = time.time()
    
    # Gather and sort file names
    files = sorted(f for f in os.listdir(worksp) if f.endswith('801_LEVEL3_LNDLG_BAP.tif'))
    
    # Organize files by year
    year_groups = {}
    for file in files:
        # Extract year from filename - assuming a specific format 'MMYYYY_otherinfo.tif'
        year = file[0:4]
        if year not in year_groups:
            year_groups[year] = []
        year_groups[year].append(file)
    
    # Process each year group
    for year, file_group in year_groups.items():
        all_stacked_data = []
        first_image = True
        for f in file_group:
            image_path = os.path.join(worksp, f)
            print(f"Processing image: {image_path}")
            dsimage = gdal.Open(image_path, gdal.GA_ReadOnly)
            if dsimage is None:
                print(f"Failed to open image: {image_path}")
                continue

            if first_image:
                cols, rows = dsimage.RasterXSize, dsimage.RasterYSize
                transform = dsimage.GetGeoTransform()
                projection = dsimage.GetProjection()
                first_image = False

            # Skip the first 4 bands, read the rest
            bands_data = [dsimage.GetRasterBand(i + 1).ReadAsArray().astype(float) 
                          for i in range(3, dsimage.RasterCount)]
            indices = calculate_indices(dsimage)  # Assuming pre-defined index calculation function

            all_stacked_data.extend(bands_data + [indices[key] for key in sorted(indices.keys())])

        output_filename = f"{year}_stacked.tif"
        workspace_saving ='/home/ubuntu/work/saved_data/landsat_disturbance_detection/1D_U_Net/tiles/stacked_per_year'
        output_path = os.path.join(workspace_saving, output_filename)
        
        arrays_to_raster(all_stacked_data, cols, rows, transform, projection, output_path)
        print(f"Saved: {output_path}")

    print(f"Process completed in {time.time() - start_time} seconds")


if __name__ == "__main__":
    workspace = '/home/ubuntu/work/saved_data/landsat_disturbance_detection/1D_U_Net/tiles/raw_data'
    process_images_in_folder(workspace)




