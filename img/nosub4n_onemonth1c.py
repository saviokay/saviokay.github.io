#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 08 15:26:55 2019
@author: saviokay
"""

import dask
import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import time

def aggMODIS(M06_files, M03_files):
    print("The List of Input MYD06_L2 Files: ")
    print(M06_files)
    print("The List of Input MYD03 Files: ")
    print(M03_files)

    """ Initialized total and cloud pixel as
        a 2D numpy array of dimension 180x360. """
    total_pix = np.zeros((180, 360))
    cloud_pix = np.zeros((180, 360))

    for M03, M06 in zip (M03_files, M06_files):
        """ Aggregate one file from MYD06_L2 for 'Cloud_Mask_1km' and its corresponding file from MYD03 for 'Latitude' and 'Longitude'.
        Read 'Latitude' and 'Longitude' variables from the MYD03 file and read the 'Cloud_Mask_1km' variable from the MYD06_L2 file.
        Group Cloud_Mask_1km values based on their corresponding (latitude, longitude) grid.

        Arguments:
            M06_files: File path for M06_file.
            M03_files: File path for corresponding M03_file.

        Returns:
            Total Pixel and Cloud Pixel as a Tuple for further calculation.
            Cloud_Pixel is an 2D numpy array of dimension 180x360 for cloud pixel count of each grid,
            Total_pixel is an 2D numpy array of dimension 180x360 for total pixel count of each grid.
        """


        """ Reading the MYD06_L2 file for the variable Cloud_Mask_1km variable
            as a Xarray Dataset with no subsampling and orignal dimension of (2030x1354). """
        d06 = xr.open_mfdataset(M06[:], parallel=True)['Cloud_Mask_1km'][:,:,:].values
        d06CM = d06[:,:,0]
        ds06_decoded = (np.array(d06CM, dtype = "byte") & 0b00000110) >> 1

        """ Reading the MYD03 file for the variable Latitude and Longitude variables
            as a Xarray Dataset with no subsampling and orignal dimension of (2030x1354). """
        d03_lat = xr.open_mfdataset(M03[:], drop_variables = "Scan Type", parallel=True)['Latitude'][:,:].values
        d03_lon = xr.open_mfdataset(M03[:], drop_variables = "Scan Type", parallel=True)['Longitude'][:,:].values

        lat = d03_lat[:,:]
        lon = d03_lon[:,:]

        l_index = (lat + 89.5).astype(int).reshape(lat.shape[0]*lat.shape[1])
        lat_index = np.where(l_index > -1, l_index, 0)
        ll_index = (lon + 179.5).astype(int).reshape(lon.shape[0]*lon.shape[1])
        lon_index = np.where(ll_index > -1, ll_index, 0)
        for i, j in zip(lat_index, lon_index):
            total_pix[i,j] += 1

        indicies = np.nonzero(ds06_decoded <= 0)
        row_i = indicies[0]
        column_i = indicies[1]
        cloud_lon = [lon_index.reshape(ds06_decoded.shape[0],ds06_decoded.shape[1])[i,j] for i, j in zip(row_i, column_i)]
        cloud_lat = [lat_index.reshape(ds06_decoded.shape[0],ds06_decoded.shape[1])[i,j] for i, j in zip(row_i, column_i)]

        for x, y in zip(cloud_lat, cloud_lon):
            cloud_pix[int(x),int(y)] += 1

        return total_pix, cloud_pix


def save_output(cf):
    """ Exporting and save cloud pixel variable
        into xarray dataarray. """
    cf1 = xr.DataArray(cf)
    cf1.to_netcdf("/home/savio1/jianwu_common/MODIS_Aggregation/savioexe/test/4/nosub4n_onemonth.hdf")

    """ Plot and save graph with the calculated
        latitude, longitude and cloud pixel. """
    plt.figure(figsize=(14,7))
    plt.contourf(range(-180,180), range(-90,90), cf, 100, cmap = "jet")
    plt.xlabel("Longitude", fontsize = 14)
    plt.ylabel("Latitude", fontsize = 14)
    plt.title("Level 3 Cloud Fraction Aggregation For One Month With No Subsampling [:,:]", fontsize = 16)
    plt.colorbar()
    plt.savefig("/home/savio1/jianwu_common/MODIS_Aggregation/savioexe/test/4/nosub4n_onemonth.png")


if __name__ =='__main__':

    """ Directory path for
        MYD06_L2 and MYD03 files. """
    M03_dir = "/umbc/xfs1/cybertrn/common/Data/Satellite_Observations/MODIS/MYD03/"
    M06_dir = "/home/savio1/cybertrn_common/common/Data/Satellite_Observations/MODIS/MYD06_L2/"

    """ Sorted file path for
        MYD06_L2 and MYD03 directory. """
    M03_files = sorted(glob.glob(M03_dir + "MYD03.A2008*.hdf"))
    M06_files = sorted(glob.glob(M06_dir + "MYD06_L2.A2008*.hdf"))

    """ Initiate program start time. """
    t0 = time.time()

    """ Calculate cloud_pixel and total_pixel
        with the input list of MYD03 and MYD06_L2 files. """
    total_pix, cloud_pix = aggMODIS(M06_files, M03_files)

    """ Calculate Cloud Fraction with the aggregated
        cloud_pixel and total_pixel. """
    cf = cloud_pix/total_pix

    print("The Computed Cloud Fraction:" + cf)
    print("The Computed Cloud Fraction Shape:" + cf.shape)

    """ Saving output into DataArray and plot graphs
        with calculated cloud fraction. """
    save_output(cf)

    """ Calculate execution time. """
    t1 = time.time()
    total = t1-t0
    print("Total program execution time (in seconds):" + total)
