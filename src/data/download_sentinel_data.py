"""
A script to download the relevant data from the Sentinel sattelite, and then resized to be paired.
"""
# package imports
import ee
import numpy as np
import pandas as pd
from pyproj import Transformer
import urllib.request
import sys
import json
from datetime import datetime
from PIL import Image
import cv2
import os

import list_files
import regrid_maxar
import height_model_file_edges

# activation of GEE
# ee.Authenticate()

# initialize gee
ee.Initialize()


def load_ref_coords(path, filename):
    """Load the csv which contains the reference co-ordinates."""
    df_coords = pd.read_csv(path + filename)

    return df_coords


def save_sentinel_image(df_coords, i, image_parameters, savedir, overwrite_files):
    # split image parameters into separate variables
    image_collection = image_parameters["image_collection"]
    date_initial = image_parameters["start_date"]
    date_final = image_parameters["end_date"]
    bands = image_parameters["bands"]
    image_bounds = image_parameters["image_bounds"]
    mask_clouds = image_parameters["mask_clouds"]
    cloud_pc = image_parameters["cloud_pc"]
    resize = image_parameters["resize"]
    scale = image_parameters["scale"]

    # first retrieve the co-ordinates
    coords, coords_3857, filecode = get_latlon_coords(df_coords, i)

    print("Downloading sentinel data for file: " + filecode)
    print(coords)
    #print(coords_3857)

    # strip the relevant info from filename
    sentinel_filename = filecode + "_tmp.tif"
    sentinel_filename_out = filecode + "_sentinel.tif"

    if os.path.exists(savedir + sentinel_filename_out) and not overwrite_files:
        return

    # get the fdb image
    fdb_img = get_fdb_image(
        image_collection, coords, date_initial, date_final, mask_clouds, cloud_pc
    )

    # select the bands and scale
    url = fdb_img.select(bands).getThumbURL(
        {"min": image_bounds[0], "max": image_bounds[1]}
    )

    # save the image
    urllib.request.urlretrieve(url, savedir + sentinel_filename)

    coord_row = df_coords.loc[[i]]
    size_x = coord_row.pixel_horiz.values[[0]]
    size_y = coord_row.pixel_vert.values[[0]]

    img = Image.open(savedir + sentinel_filename)

    if resize:
        img_arr = np.array(img)
        sy, sx, _ = np.shape(img_arr)
        print(sx, sy)
        num_x = int(scale * sx)
        num_y = int(scale * sy)
        resized = cv2.resize(
            img_arr, (int(num_x), int(num_y)), interpolation=cv2.INTER_NEAREST
        )
        resized2 = Image.fromarray(resized)
        resized2.save(savedir + sentinel_filename)
    else:
        img.save(savedir + sentinel_filename)

    regrid_maxar.gdal_georef_sentinel(
        savedir + sentinel_filename, savedir + sentinel_filename_out, coords_3857
    )

    os.remove(savedir + sentinel_filename)

    return


def maskS2clouds(image):
    # Select the 'MSK_CLDPRB' band
    cld = image.select("MSK_CLDPRB")

    # Mask out cloudy pixels
    mask = cld.eq(0)

    # Apply the mask and scale the image
    return image.updateMask(mask).divide(10000)


def get_fdb_image(
    image_collection, coords, date_initial, date_final, mask_clouds=False, cloud_pc=20
):
    """Get an fdb image from a set of co-ordinates."""
    aoi = ee.Geometry.Polygon(coords)

    if not mask_clouds:
        ffa_db = ee.Image(
            ee.ImageCollection(image_collection)
            .filterBounds(aoi)
            .filterDate(ee.Date(date_initial), ee.Date(date_final))
            .first()
            .clip(aoi)
        )
    else:
        ffa_db = ee.Image(
            ee.ImageCollection(image_collection)
            .filterBounds(aoi)
            .filterDate(ee.Date(date_initial), ee.Date(date_final))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pc))
            .map(maskS2clouds)
            .first()
            .clip(aoi)
        )

    return ffa_db


def get_latlon_coords(df_coords, i):
    """Returns a set of lat/long co-ordinates from tif file."""

    # choose a given row
    coord_row = df_coords.loc[[i]]

    # retrive the co-ordinates
    x0 = coord_row.left.values[0]
    x1 = coord_row.right.values[0]
    y0 = coord_row.bottom.values[0]
    y1 = coord_row.top.values[0]
    coords_3857 = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=object)

    # change - to _ in filename to match with maxar data
    filename = coord_row.file_name.values[0][:13]

    # define the transformer
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # transform to lat lon (epsg:4326)
    coords_x, coords_y = transformer.transform(coords_3857[:, 0], coords_3857[:, 1])

    coords_latlon = np.dstack([coords_x, coords_y])[0].tolist()
    coords_3857 = [x0, y1, x1, y0]
    # print(coords_latlon)
    # print([x0],[x1],[y1],[y0])
    # print(coords_x)
    # print(coords_y)
    return coords_latlon, coords_3857, filename


def saving(df_path, image_parameters, savedir, overwrite_files=False):
    df_coords = load_ref_coords("", df_path)

    # print(df_coords.head())
    # print(df_coords.info())

    # coords_latlon, filename = get_latlon_coords(df_coords, 1)
    # print(coords_latlon, filename)

    df = pd.read_csv(df_path)
    n = df.shape[0]

    for i in range(n):
        save_sentinel_image(df_coords, i, image_parameters, savedir, overwrite_files)

    with open("parameters.json", "w") as outfile:
        json.dump(image_parameters, outfile)

    now = datetime.now()
    date_time = now.strftime("%d.%m_%H.%M")
    print("date and time:", date_time)


if __name__ == "__main__":
    ### change user variables as appropriate

    overwrite_files = False  # overwrites pre-written files
    resize = True  # whether to resize sentinel files
    scale = 2  # scaling factor for resize (if true)

    # list of sentinel inputs to loop over
    # for each set of parameters, a new directory is made
    image_params_list = [
        {
            "image_collection": "COPERNICUS/S1_GRD",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["VV"],
            "image_bounds": [-25, 5],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 2,
        }
    ]

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S1_GRD",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["VH"],
            "image_bounds": [-25, 5],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 2,
        }
    )

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B2"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 2,
        }
    )

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B3"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 2,
        }
    )

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B4"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 2,
        }
    )

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B5"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 4,
        }
    )

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B6"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 4,
        }
    )

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B7"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 4,
        }
    )

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B8"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 2,
        }
    )    

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B11"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 4,
        }
    )

    image_params_list.append(
        {
            "image_collection": "COPERNICUS/S2_SR",
            "start_date": "2020-09-01",
            "end_date": "2020-09-30",
            "bands": ["B12"],
            "image_bounds": [0, 3000],
            "mask_clouds": False,
            "cloud_pc": 20,
            "resize": True,
            "scale": 4,
        }
    )

    ### end of user variables - should be no need to change anything below

    datadir = "/work/unicef/datasets/"

    # path to write csv summary files to
    csvs_path = datadir + "summary_data/"

    # path to write lists of files to
    listfiles_path = csvs_path + "file_lists/"

    BHM_init_path = datadir + "height_model/"
    BHM_filename = "BHM_merged_list.txt"
    VHM_filename = "VHM_merged_list.txt"

    # first: write the csv file with the merged bhm images
    list_files.list_BHM_files(BHM_init_path, listfiles_path, BHM_filename, merged=True)
    list_files.list_VHM_files(BHM_init_path, listfiles_path, VHM_filename, merged=True)

    # next: convert the co-ordinate system to epsg:3857
    regrid_maxar.reproject_all_bhm(listfiles_path + BHM_filename, overwrite_files)
    regrid_maxar.reproject_all_bhm(listfiles_path + VHM_filename, overwrite_files)

    BHM_filename = "BHM_merged_reproj_list.txt"
    VHM_filename = "VHM_merged_reproj_list.txt"

    # list the reprojected files
    list_files.list_BHM_files(
        BHM_init_path,
        listfiles_path,
        BHM_filename,
        search_string="_reproject",
        merged=True,
    )
    list_files.list_VHM_files(
        BHM_init_path,
        listfiles_path,
        VHM_filename,
        search_string="_reproject",
        merged=True,
    )

    BHM_csv = "BHM_pm_merged.csv"
    VHM_csv = "BHM_pm_merged.csv"

    for image_parameters in image_params_list:
        sentinel_dir = (
            datadir
            + "sentinel_data/"
            + image_parameters["image_collection"].split("/")[1]
            + "/"
            + "_".join(image_parameters["bands"])
            + "/"
        )

        if not os.path.exists(sentinel_dir):
            os.makedirs(sentinel_dir)

        saving(
            csvs_path + BHM_csv,
            image_parameters,
            sentinel_dir,
            overwrite_files=overwrite_files,
        )

        # list the sentinel files
        sentinel_filename = (
            "sentinel_"
            + image_parameters["image_collection"].split("/")[1]
            + "_"
            + "_".join(image_parameters["bands"])
            + ".csv"
        )
        list_files.list_sentinel_files(sentinel_dir, listfiles_path, sentinel_filename)

        # write the co-ordinates
        height_model_file_edges.write_csv(
            listfiles_path + sentinel_filename,
            csvs_path + sentinel_filename,
            bhm=False,
            sentinel=True,
        )

    # downsize bhm files to same extents
    regrid_maxar.resolve_bhm_all(
        datadir + "height_model/",
        datadir + "height_model/",
        csvs_path + sentinel_filename,
        csvs_path + BHM_csv,
        sentinel=True,
        overwrite_files=overwrite_files,
    )

    # downsize bhm files to same extents
    regrid_maxar.resolve_vhm_all(
        datadir + "height_model/",
        datadir + "height_model/",
        csvs_path + sentinel_filename,
        csvs_path + VHM_csv,
        sentinel=True,
        overwrite_files=overwrite_files,
    )

    BHM_filename = "BHM_merged_subset_reproj_res_list.txt"
    BHM_csv = "BHM_merged_pm_res.csv"

    # list the files and get their co-ords
    list_files.list_BHM_files(
        datadir + "height_model/",
        listfiles_path,
        BHM_filename,
        search_string="_reproject_resolve",
        merged=True,
    )

    # write the co-ordinates to file
    height_model_file_edges.write_csv(
        listfiles_path + BHM_filename,
        csvs_path + BHM_csv,
    )

    VHM_filename = "VHM_merged_subset_reproj_res_list.txt"
    VHM_csv = "VHM_merged_pm_res.csv"

    # list the files and get their co-ords
    list_files.list_VHM_files(
        datadir + "height_model/",
        listfiles_path,
        VHM_filename,
        search_string="_reproject_resolve",
        merged=True,
    )

    # write the co-ordinates to file
    height_model_file_edges.write_csv(
        listfiles_path + VHM_filename,
        csvs_path + VHM_csv,
    )


# 3. And finally the function is called to run the script to download the data:
# saving(image_parameters=image_parameters, df_path=df_path)


### user defined inputs ###
# This is what the user has to run to download the Sentinel images:
# 1. It is necessary to specify the directory and file name of the csv containing the co-ordinates:
# df_path = "file.csv"
# The file.csv has to contain the co-ordinates as: left, right, top, bottom; and the image size as: pixel_horiz, pixel_vert
#
# 2. Then, these parameters need to be specified (the data included is an example, see the option in the point 4):
# image_parameters = {"image_collection": "COPERNICUS/S1_GRD",
#                    "start_date": "2020-09-01",
#                    "end_date": "2020-09-30",
#                    "bands": ["VV"],
#                    "image_bounds": [-25, 5]}
#
# 3. And finally the function is called to run the script to download the data:
# saving(image_parameters=image_parameters, df_path=df_path)
#
# 4. Possible parameters to download from Sentinel 1 or Sentinel 2:
# COPERNICUS/S2_SR; min: 0.0, max: 3000 bands B4, B3, B2
# COPERNICUS/S1_GRD; min: -25, max: 5 bands VV,VH (these bands can only be downloaded separatelly)
