import glob
import re
import os.path
import imageio
import numpy as np

def get_bhm_path(subsetid, sliceid):
    return 'height_model/' + subsetid + '-BHM/BHM-0-merged-' + subsetid + '_reproject_resolve_s' + sliceid + '.tif'

def get_vhm_path(subsetid, sliceid):
    return 'height_model/' + subsetid + '-VHM/VHM-0-merged-' + subsetid + '_reproject_resolve_s' + sliceid + '.tif'

def get_s2_b2_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B2/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s2_b3_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B3/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s2_b4_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B4/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s2_b5_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B5/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s2_b6_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B6/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s2_b7_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B7/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s2_b8_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B8/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s2_b11_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B11/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s2_b12_path(subsetid, sliceid):
    return 'sentinel_data/S2_SR/B12/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s1_vv_path(subsetid, sliceid):
    return 'sentinel_data/S1_GRD/VV/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'

def get_s1_vh_path(subsetid, sliceid):
    return 'sentinel_data/S1_GRD/VH/0-merged-' + subsetid + '_sentinel_s' + sliceid + '.tif'


for filename in glob.glob('height_model/????-BHM/BHM-*_reproject_resolve.tif'):

    regex = re.compile(r'height_model/(\d\d\d\d)-BHM/BHM-0-merged-(.*)_reproject_resolve.tif')

    matches = regex.search(filename)
    subsetid = matches.group(1)

    # We skip subsets with more than %5 mismatch in the original BHM files

    bad_subsetids = ['3435', '4321', '2446', '3342']

    if subsetid in bad_subsetids:
        continue

    pattern = filename.replace('.tif','_s*.tif')

    if not os.path.exists('sentinel_layers/'):
        os.makedirs('sentinel_layers')
    
    for slicefilename in glob.glob(filename.replace('.tif','_s*.tif')):
    
        regex2 = re.compile(r'height_model/(\d\d\d\d)-BHM/BHM-0-merged-(.*)_reproject_resolve_s(.*).tif')

        matches2 = regex2.search(slicefilename)
        sliceid = matches2.group(3)

        bhm = get_bhm_path(subsetid, sliceid)
        #vhm = get_vhm_path(subsetid, sliceid)
        b2 = get_s2_b2_path(subsetid, sliceid)
        b3 = get_s2_b3_path(subsetid, sliceid)
        b4 = get_s2_b4_path(subsetid, sliceid)
        b5 = get_s2_b5_path(subsetid, sliceid)
        b6 = get_s2_b6_path(subsetid, sliceid)
        b7 = get_s2_b7_path(subsetid, sliceid)
        b8 = get_s2_b8_path(subsetid, sliceid)
        b11 = get_s2_b11_path(subsetid, sliceid)
        b12 = get_s2_b12_path(subsetid, sliceid)
        vv = get_s1_vv_path(subsetid, sliceid)
        vh = get_s1_vh_path(subsetid, sliceid)

        s1s2_filenames = [b2, b3, b4, b5, b6, b7, b8, b11, b12, vv, vh]
        incomplete_dataset = False

        for f in [bhm] + s1s2_filenames:
            if not os.path.isfile(f):
                print('File not found: {}'.format(f))
                incomplete_dataset = True
        
        if incomplete_dataset == True:
            continue

        layers = np.array([])

        for f in s1s2_filenames:

            # The second layer in Sentinel data is always filled with 255
            image = imageio.imread(f)[:, :, 0]
            
            if layers.shape == (0,):
                layers = image
            else:
                layers = np.dstack((layers, image))
            
            #print('{} {}'.format(f, image.shape))

        bhm_filename = 'sentinel_layers/bhm_{}_s{}.npy'.format(subsetid, sliceid)
        layers_filename = 'sentinel_layers/layers_{}_s{}.npy'.format(subsetid, sliceid)

        print('{} {}'.format(layers.shape, layers_filename))

        np.save(bhm_filename, imageio.imread(bhm))
        np.save(layers_filename, layers)
