import os
import sys
from PIL import Image
from osgeo import gdal

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: {} <filename>'.format(sys.argv[0]))
        sys.exit(0)

    filename, file_extension = os.path.splitext(sys.argv[1])

    image = gdal.Open(sys.argv[1])
    width, height = image.RasterXSize, image.RasterYSize
    counter = 0

    x_window_size, y_window_size = 100, 100
    x_overlap, y_overlap = 40, 40

    processed = []

    for x in range(0, width-x_window_size+1, x_overlap):
        for y in range(0, height-y_window_size+1, y_overlap):

            new_filename = '{}_s{}{}'.format(filename, counter, file_extension)

            # We make sure that we are not recursively slicing a slice
            if new_filename in processed:
                continue

            counter += 1

            gdal.Translate(new_filename, image, srcWin=[x, y, x_window_size, y_window_size])

            processed.append(new_filename)
