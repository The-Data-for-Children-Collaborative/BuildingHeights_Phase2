import glob
import re
import os.path
import imageio
import numpy as np
import torch

def get_bhm_path(subsetid):
    return 'height_model/' + subsetid + '-BHM/BHM-0-merged-' + subsetid + '_reproject_resolve.tif'

def get_vhm_path(subsetid):
    return 'height_model/' + subsetid + '-VHM/VHM-0-merged-' + subsetid + '_reproject_resolve.tif'

def get_s2_b2_path(subsetid):
    return 'sentinel_data/S2_SR/B2/0-merged-' + subsetid + '_sentinel.tif'

def get_s2_b3_path(subsetid):
    return 'sentinel_data/S2_SR/B3/0-merged-' + subsetid + '_sentinel.tif'

def get_s2_b4_path(subsetid):
    return 'sentinel_data/S2_SR/B4/0-merged-' + subsetid + '_sentinel.tif'

def get_s2_b5_path(subsetid):
    return 'sentinel_data/S2_SR/B5/0-merged-' + subsetid + '_sentinel.tif'

def get_s2_b6_path(subsetid):
    return 'sentinel_data/S2_SR/B6/0-merged-' + subsetid + '_sentinel.tif'

def get_s2_b7_path(subsetid):
    return 'sentinel_data/S2_SR/B7/0-merged-' + subsetid + '_sentinel.tif'

def get_s2_b8_path(subsetid):
    return 'sentinel_data/S2_SR/B8/0-merged-' + subsetid + '_sentinel.tif'

def get_s2_b11_path(subsetid):
    return 'sentinel_data/S2_SR/B11/0-merged-' + subsetid + '_sentinel.tif'

def get_s2_b12_path(subsetid):
    return 'sentinel_data/S2_SR/B12/0-merged-' + subsetid + '_sentinel.tif'

def get_s1_vv_path(subsetid):
    return 'sentinel_data/S1_GRD/VV/0-merged-' + subsetid + '_sentinel.tif'

def get_s1_vh_path(subsetid):
    return 'sentinel_data/S1_GRD/VH/0-merged-' + subsetid + '_sentinel.tif'


nr_files = 0
current_file = 0

# This needs to be initialized to the number of features
matrix = np.empty((0,24))
nr_samples = 0

for filename in glob.glob('height_model/????-BHM/BHM-*_reproject_resolve.tif'):
    nr_files += 1

for filename in glob.glob('height_model/????-BHM/BHM-*_reproject_resolve.tif'):

    regex = re.compile(r'height_model/(\d\d\d\d)-BHM/BHM-0-merged-(.*)_reproject_resolve.tif')

    matches = regex.search(filename)
    subsetid = matches.group(1)

    print('{}/{}'.format(current_file, nr_files))
    current_file += 1

    # We skip subsets with more than 5% mismatch in the original BHM files

    bad_subsetids = ['3435', '4321', '2446', '3342']

    if subsetid in bad_subsetids:
        continue

    if not os.path.exists('sentinel_pixels/'):
        os.makedirs('sentinel_pixels')

    # Features

    b2 = get_s2_b2_path(subsetid)
    b3 = get_s2_b3_path(subsetid)
    b4 = get_s2_b4_path(subsetid)
    b5 = get_s2_b5_path(subsetid)
    b6 = get_s2_b6_path(subsetid)
    b7 = get_s2_b7_path(subsetid)
    b8 = get_s2_b8_path(subsetid)
    b11 = get_s2_b11_path(subsetid)
    b12 = get_s2_b12_path(subsetid)
    vv = get_s1_vv_path(subsetid)
    vh = get_s1_vh_path(subsetid)

    # Label

    bhm = get_bhm_path(subsetid)

    # We check if all the files exist

    s1s2_filenames = [b2, b3, b4, b5, b6, b7, b8, b11, b12, vv, vh]
    incomplete_dataset = False

    for f in [bhm] + s1s2_filenames:
        if not os.path.isfile(f):
            print('File not found: {}'.format(f))
            incomplete_dataset = True

    if incomplete_dataset == True:
        continue

    # We read the datafiles, and resize them in case there's some slight mismatch

    s1s2_data = list(map(lambda f: imageio.imread(f)[:, :, 0], s1s2_filenames))
    bhm_data = imageio.imread(bhm)

    xdim = bhm_data.shape[0]
    ydim = bhm_data.shape[1]

    for i in range(0, len(s1s2_data)):

        layer = s1s2_data[i]

        if (layer.shape[0] != xdim) or (layer.shape[1] != ydim):

            layer = torch.tensor(layer)
            layer = layer[None, None, :, :]
            layerprime = torch.nn.functional.interpolate(layer, (xdim, ydim), mode='bilinear')

            s1s2_data[i] = layerprime.detach().numpy()[0,0]

    # We finally iterate through all pixels, and save them in the matrix as needed

    for x in range(0, xdim):
        for y in range(0, ydim):

            # The shape must match the number of features

            entry = np.zeros(shape=(12+12))
            counter = 0

            # Features from Sentinel 1 and 2

            for layer_data in s1s2_data:
                entry[counter] = layer_data[x,y]
                counter += 1

            if np.linalg.norm(entry) <= 1e-5:
                continue

            # Engineered features

            for j in range(0,3):

                local = float(s1s2_data[j][x,y])

                entry[counter] = float(s1s2_data[j][x-1, y]) - local if x > 0 else 0
                counter += 1

                entry[counter] = float(s1s2_data[j][x+1, y]) - local if x < (xdim-1) else 0
                counter += 1

                entry[counter] = float(s1s2_data[j][x, y-1]) - local if y > 0 else 0
                counter += 1

                entry[counter] = float(s1s2_data[j][x, y+1]) - local if y < (ydim-1) else 0
                counter += 1

            # At last, after the features, the label

            entry[counter] = bhm_data[x, y]

            if (nr_samples % 100) == 0:
                matrix = np.vstack((matrix, entry))

            nr_samples += 1

np.random.shuffle(matrix)
np.save('pixel100s_features_shuffled.npy', matrix)
