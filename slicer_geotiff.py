#This is a Python script that slices a GeoTIFF image into smaller tiles.
#It uses the GDAL library to open and extract the image data.
#The script takes three command-line arguments: the path to the input GeoTIFF file, the size of each tile in pixels, and the size of the overlap between adjacent tiles in pixels.
#The script first opens the input GeoTIFF file and retrieves its size in pixels.
#It then calculates the number of tiles needed to cover the entire image based on the tile size and overlap specified.
#Next, the script creates an output folder and loops through all the tiles, extracting each one using the GDAL Translate function.
#Finally, the input dataset is closed.
#To run this script, you would need to have the GDAL library installed on your system.
#You can then call the script from the command line and specify the three required arguments. For example:#
#python slicer_geotiff.py --input_file input.tif --tile_size N --overlap C
#This would slice the input.tif image into NxN pixel tiles with a C pixel overlap between adjacent tiles.
#The resulting tiles would be saved in a folder called "slices_input" in the same directory as the input file.



import argparse
import os
from osgeo import gdal

def slice_image(input_file, tile_size, overlap):
    # Open the input GeoTiff file
    input_dataset = gdal.Open(input_file)

    # Get the size of the input image
    cols = input_dataset.RasterXSize
    rows = input_dataset.RasterYSize

    # Calculate the number of tiles in each dimension
    num_cols = int((cols - tile_size) / (tile_size - overlap)) + 1
    num_rows = int((rows - tile_size) / (tile_size - overlap)) + 1

    # Create the output folder if it doesn't exist
    output_folder = "slices_" + os.path.splitext(os.path.basename(input_file))[0]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all the tiles and extract them
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate the offset for this tile
            x_offset = j * (tile_size - overlap)
            y_offset = i * (tile_size - overlap)

            # Create the output filename for this tile
            output_file = os.path.join(output_folder, "{}_{}_{}.tif".format(os.path.splitext(os.path.basename(input_file))[0], x_offset, y_offset))

            # Extract the tile using GDAL
            gdal.Translate(output_file, input_dataset, srcWin=[x_offset, y_offset, tile_size, tile_size])

            print("Extracted tile {}_{}".format(x_offset, y_offset))

    # Close the input dataset
    input_dataset = None

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Slice a GeoTIFF image into tiles")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input GeoTIFF file")
    parser.add_argument("--tile_size", type=int, required=True, help="Size of each tile in pixels")
    parser.add_argument("--overlap", type=int, required=True, help="Size of overlap between adjacent tiles in pixels")
    args = parser.parse_args()

    # Call the slice_image function with the command-line arguments
    slice_image(args.input_file, args.tile_size, args.overlap)
