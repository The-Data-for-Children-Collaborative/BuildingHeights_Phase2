#The script takes a directory path containing VHM (Vegetation Height Model) GeoTIFF files as input, and analyzes the vegetation information in each file to calculate various statistics. The script then stores the results in a CSV file.

#To achieve this, the script does the following:

#1. Parses the command line arguments using the argparse module.
#2. Opens a CSV file to store the results.
#3. Loops through each file in the input directory that starts with 'vhm' and has a '.tif' extension.
#4. Opens each file using the gdal module, which is a library for working with raster data.
#5. Reads the vegetation height data from the file as a NumPy array.
#6. Calculates various statistics on the vegetation data, including mean, median, fraction of vegetation above a certain threshold, quantiles, and variance.
#7. Writes the results to the CSV file.
#8. Closes the file and moves on to the next file.
#9. At the end of the script, the CSV file containing the results is saved in the specified output location

# To call the function use;
#python analyze_vhm.py /path/to/vhm/files/ /path/to/output/output.csv --threshold 0.5


#Typical path to triples is /work/unicef/valentina/triples


#!/usr/bin/env python3

import os
import csv
import argparse
from osgeo import gdal
import numpy as np

# Define the command line arguments
parser = argparse.ArgumentParser(description='Analyzes VHM GeoTIFF files to determine vegetation information.')
parser.add_argument('path', type=str, help='The path to the directory containing the VHM files.')
parser.add_argument('output', type=str, help='The path to the output CSV file.')
parser.add_argument('--threshold', type=float, default=0.5, help='The vegetation height threshold (in meters).')
args = parser.parse_args()

# Open the CSV file to store the results
csv_file = open(args.output, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['filename', 'mean_vegetation_height', 'median_vegetation_height', 'vegetation_fraction', 'vegetation_quantiles', 'vegetation_variance'])

# Loop through all the VHM files
for file in os.scandir(args.path):
    if file.name.startswith('vhm') and file.name.endswith('.tif') and file.is_file():
        # Open the file
        dataset = gdal.Open(file.path)

        # Read the vegetation height data as a NumPy array
        vegetation_height = dataset.ReadAsArray()

        # Calculate the mean vegetation height
        mean_vegetation_height = np.mean(vegetation_height)

        # Calculate the median vegetation height
        median_vegetation_height = np.median(vegetation_height)

        # Calculate the fraction of pixels with vegetation above the threshold
        vegetation_pixels = np.count_nonzero(vegetation_height > args.threshold)
        total_pixels = vegetation_height.size
        vegetation_fraction = vegetation_pixels / total_pixels

        # Calculate the vegetation quantiles
        vegetation_quantiles = np.quantile(vegetation_height[vegetation_height > args.threshold], [0.25, 0.5, 0.75])

        # Calculate the vegetation variance
        vegetation_variance = np.var(vegetation_height[vegetation_height > args.threshold])

        # Write the results to the CSV file
        csv_writer.writerow([file.name, mean_vegetation_height, median_vegetation_height, vegetation_fraction, vegetation_quantiles, vegetation_variance])

        # Close the file
        dataset = None

# Close the CSV file
csv_file.close()
