#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from astropy.io import fits, ascii
import astropy.table
from astropy.table import Table, join, vstack, unique #, setdiff
import numpy as np
import math
import astropy.units as u
from astropy.utils import data
from spectral_cube import SpectralCube
import turbustat
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, Kernel
import pylab as py
from scipy import signal, fftpack
from astropy.modeling.models import Gaussian2D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import ndimage
from turbustat.statistics import PowerSpectrum
from astropy.io import fits
import astropy.units as u
from spectral_cube import Projection
from radio_beam import Beam
from turbustat.io.sim_tools import create_fits_hdu
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
import statistics
from turbustat.moments import Moments 

os.chdir('/media/gingko/Elements/system999')

# FUNCTIONS

def sigma2fwhm(sigma):
   return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
   return fwhm / np.sqrt(8 * np.log(2))



def image_clean (image, id_):
    'creates new image with only the cloud specified by id'
    
    #open image 
    
    hdu=fits.open(image)[0]
    hd=hdu.header
    a=hdu.data 
     
    if id_ == 0.0:
       a[a != id_] = 1
       a[a == 0.0] = 2
       a[a == 1] = 0
       a[a != 0] = 1
       
    else:
        a[a!=id_]=0
        a[a != 0] = 1
#
    a[np.isnan(a)] = 0
    
    print('Writing clean SCIMES asgn image')    
    fits.writeto('clean.fits',a, hd, overwrite=True)
          
    return a 

def image_mask (clean):
    'creates new image with only the cloud specified by id'
    
    #open image 
    
    hdu=fits.open(clean)[0]
    hd=hdu.header
    a=hdu.data 
     
    a[a != 0] = 1
       
    print('Making mask')    
    fits.writeto('mask.fits',a, hd, overwrite=True)




def cloud_emission (cloud_map, emission_map):
    'from a cloud cube and emission map, extracts the emission of the cloud'
    
       
    hdu1 = fits.open(cloud_map)[0]
    hd1 = hdu1.header
    a1 = hdu1.data 
    
    #make a mask for the cloud
    a1[a1 != 0] = 1
    
    hdu2 = fits.open(emission_map)[0]
    hd2 = hdu2.header
    a2 = hdu2.data 
    a2[np.isnan(a2)] = 0


    a1 = np.multiply(a1, a2)

    
    fits.writeto('cloud_emission.fits', a1, hd1, overwrite=True)



    
def discontinuity (image, idx):
    'check for disconnected clouds'
    
    hdu=fits.open(image)[0]
    hd=hdu.header
    a=hdu.data 

    if idx == 0:
       a[a != idx] = 1
       a[a == 0] = 2
       a[a == 1] = 0
       
        
    else:
        a[a!=idx]=0
    
    
    a = np.array(a).byteswap().newbyteorder()
    labeled_array, num_features = ndimage.label(a, np.ones((3,3,3)))
    
    return num_features


def noise_map (cloud, noise, l, b):
     
    hdu = fits.open(cloud)[0]
    hd = hdu.header
    a = hdu.data 

    hdu1 = fits.open(noise)[0]
    hd1 = hdu1.header
    n = hdu1.data 
    
    a[a != 0] = 1

    (b, l) = a.shape
    
    e = n[:b,:l]
    
    f = np.multiply(a, e)
    
    fits.writeto('cloud_noise.fits', f, hd1, overwrite=True)

    
def calculate_obs_moments (cloud_emission):
    'moments map in the frame of the observer'
    
    hdu = fits.open(cloud_emission)[0]
    hd = hdu.header
    a = hdu.data 
    
    dv = abs(hd["CDELT3"])
    cube = SpectralCube.read(cloud_emission)
    v_range = cube.spectral_axis.value
    
    voxels = np.where(a != 0)
    
    
    cloud0 = np.zeros(a.shape)
    
    for i in range(len(voxels[0])):
        cloud0[voxels[0][i]][voxels[1][i]][voxels[2][i]] = \
                    a[voxels[0][i]][voxels[1][i]][voxels[2][i]]*dv
    
    
    cloud1 = np.zeros(a.shape)
    
    for i in range(len(voxels[0])):
       cloud1[voxels[0][i]][voxels[1][i]][voxels[2][i]] = \
                    a[voxels[0][i]][voxels[1][i]][voxels[2][i]]*v_range[voxels[0][i]]*dv
    
    cloud2 = np.zeros(a.shape)
    
    for i in range(len(voxels[0])):
                cloud2[voxels[0][i]][voxels[1][i]][voxels[2][i]] = \
                    dv*a[voxels[0][i]][voxels[1][i]][voxels[2][i]]*v_range[voxels[0][i]]**2
    
    
    W0 = cloud0.sum( axis = 0 )
    W1_obs = cloud1.sum( axis = 0 )
    W2_obs = cloud2.sum( axis = 0 )
    
    fits.writeto('W0.fits', W0, hd, overwrite=True)
    fits.writeto('W1_obs.fits', W1_obs, hd, overwrite=True)
    fits.writeto('W2_obs.fits', W2_obs, hd, overwrite=True)

    return W0, W1_obs, W2_obs


def velocity_c (W0, W1_obs):
    'calcualtate the center-of-mass velocity'   
   
    return spatial_average(W1_obs)/spatial_average(W0)



def spatial_average (WF): 
    
    hdu1 = fits.open(WF)[0]
    hd1 = hdu1.header
    W = hdu1.data 

    c = float(np.count_nonzero(W))
    d = float(np.sum(W))

    # pixels=np.where(W != 0)
    
    # v = []
    # for i in range(len(pixels[0])):
    #     v.append(W[pixels[0][i]][pixels[1][i]])
    
    # a_W = sum(v)/len(v)
    
    return d/c
    


def square_field (WF, name = 'field'): 
    
    hdu1 = fits.open(WF)[0]
    hd1 = hdu1.header
    W = hdu1.data 

    W2 = np.multiply(W,W)

    fits.writeto(name + '_squared.fits', W2, hd1, overwrite=True)

    


def calculate_com_moments (oW0, oW1, oW2, V_c):
    #calculate moment fields in the center-of-mass frame
    
    hdu0 = fits.open(oW0)[0]
    hd0 = hdu0.header
    W0 = hdu0.data 
    
    hdu1 = fits.open(oW1)[0]
    hd1 = hdu1.header
    W1_obs = hdu1.data 
    
    hdu2 = fits.open(oW2)[0]
    hd2 = hdu2.header
    W2_obs = hdu2.data 
   
    W1 = W1_obs - V_c*W0
    W2 = W2_obs + (V_c**2)*W0 - (2*V_c)*W1_obs

    fits.writeto('W1.fits', W1, hd1, overwrite=True)
    fits.writeto('W2.fits', W2, hd2, overwrite=True)


def sigma_v (W0, W2):
    #calculate the velcity dispersion the center-of-mass frame
    
    hdu0 = fits.open(W0)[0]
    hd0 = hdu0.header
    W0 = hdu0.data 

    hdu2 = fits.open(W2)[0]
    hd2 = hdu2.header
    W2 = hdu2.data 
   
   
    B = np.divide(W0, W2)
    flow = np.sqrt(B)
#   k = 2*np.log10(2#   s = 2*np.sqrt(k)
    sigma_v = 0.7759252*flow
 
    #W0.write('W0.fits', overwrite = True)
    fits.writeto('flow.fits', flow, hd0, overwrite=True)
    fits.writeto('sigma_v.fits', sigma_v, hd0, overwrite=True)

    
def cloud_extraction (image, side, pad, label):
    'extracts the cloud emission (v,x,x) size, with x equal'
    'to the -side- dimension of the cloud plus the value pad '
    
    hdu=fits.open(image)[0]
    hd=hdu.header
    a=hdu.data 
    
    
    pixels=np.where(a != 0) 

    l_min = min(pixels[1])
    b_min = min(pixels[0])
   
    l_max = max(pixels[1])
    b_max = max(pixels[0])
   
    
    #move the origin at point (l_min, b_min)
    new_pixels_l = [0]*len(pixels[0])
    new_pixels_b = [0]*len(pixels[0])
   
#    
    for i in range(len(pixels[0])):
        new_pixels_b[i] = pixels[0][i] - b_min
        new_pixels_l[i] = pixels[1][i] - l_min

     
   
    #make new matrix
  
    #l_max = max(new_pixels_l)
    #b_max = max(new_pixels_b)

    t = int(side) # max(l_max, b_max)


    cloud = np.zeros((t+2*pad, t+2*pad))
        
#now let's fill the square matrix that contains the cloud
    
 
    for i in range(len(new_pixels_l)):
            
        l = new_pixels_l[i]
        b = new_pixels_b[i]
        
        cloud[b +pad][l + pad] = a[pixels[0][i]][pixels[1][i]]
 
    
    fits.writeto(label + '_cut.fits', cloud, hd, overwrite = True)



def cloud_extraction_2D (clean, cube, pad, label = 'density'):
    
    hdu=fits.open(clean)[0]
    hd=hdu.header
    a=hdu.data 
    
    hdu=fits.open(cube)[0]
    hd=hdu.header
    c=hdu.data 
    
    pixels=np.where(a != 0) 

    v_min = min(pixels[0])
    l_min = min(pixels[2])
    b_min = min(pixels[1])
   
    v_max = max(pixels[0])
    l_max = max(pixels[2])
    b_max = max(pixels[1])
   
    
    #move the origin at point (l_min, b_min)
    new_pixels_l = [0]*len(pixels[0])
    new_pixels_b = [0]*len(pixels[0])
    new_pixels_v = [0]*len(pixels[0])
   
#    
    for i in range(len(pixels[0])):
        new_pixels_b[i] = pixels[1][i] - b_min
        new_pixels_l[i] = pixels[2][i] - l_min
        new_pixels_v[i] = pixels[0][i] - v_min

     
   
    #make new matrix
  
    l_max = max(new_pixels_l)
    b_max = max(new_pixels_b)
    v_max = max(new_pixels_v)

    t = max(l_max, b_max, v_max)


    cloud = np.zeros((t, t+2*pad, t+2*pad))
        
#now let's fill the square matrix that contains the cloud
    
 
    for i in range(len(new_pixels_l)):
            
        l = new_pixels_l[i]
        b = new_pixels_b[i]
        v = new_pixels_v[i]
        
        cloud[v][b +pad][l + pad] = c[pixels[0][i]][pixels[1][i]][pixels[2][i]]
 
        dmap = cloud.sum(axis = 0)
        
        dmap[np.isnan(dmap)] = 0

        
    fits.writeto('cloud_cut_' + label + '.fits', dmap, hd, overwrite = True)


def field_size (cloud, dimension):
    'Determines size of field in (v-b) or (v-b-l) space, depending on the'
    'dimesions of the field'
    
    hdu=fits.open(cloud)[0]
    hd=hdu.header
    a=hdu.data 
    
    pixels=np.where(a != 0)
        
    
  #  p0=len(np.unique(pixels[0]))
  #  p1=len(np.unique(pixels[1]))
    if dimension == 3:
        p0_min = min(pixels[0])
        p1_min = min(pixels[1])
        p2_min = min(pixels[2])

   
        p0_max = max(pixels[0])
        p1_max = max(pixels[1])
        p2_max = max(pixels[2])

    
        lv = float(p0_max - p0_min)
        lb = float(p1_max - p1_min)
        ll = float(p2_max - p2_min)

        return lv, lb, ll
    
    else:
        p0_min = min(pixels[0])
        p1_min = min(pixels[1])

   
        p0_max = max(pixels[0])
        p1_max = max(pixels[1])

    
        lv = float(p0_max - p0_min)
        lb = float(p1_max - p1_min)

        return lv, lb



def cloud_mask (cloud):
    'Returns a mask corresponding to the cloud (an array with zeros and ones)'
    
    hdu=fits.open(cloud)[0]
    hd=hdu.header
    a=hdu.data 
    
    a[a != 0] = 1

    fits.writeto('cloud_mask.fits',a, hd, overwrite = True)
    
    return 'mask: OK'
    
    

def noise_cut (noise_map, cloud_mask):
   'creates a subcube of the noise cube, calculates and'
   'returns the power spectrum'
  
   hdu=fits.open(noise_map)[0]
   a=hdu.data     
   
   hdu_b=fits.open(cloud_mask)[0]
   hd=hdu_b.header
   b=hdu_b.data     
   
   
   size_b, size_l = b.shape
   
   image = a[0:size_b,0: size_l]
   
   b[b!=0] = 1
   
   c = np.multiply(image,b)
   
   fits.writeto('cloud_noise.fits',c, hd, overwrite = True)
   
   return c #noise array 
   
   for i in range(len(image[0])):
      # Take the fourier transform of the image.
      array = np.multiply(np.array(image[i]), np.array(b))
      
      #smooth
      kernel = Gaussian2DKernel(2, x_size=5, y_size = 5)
      
    #  smoothed_array = convolve_fft(array, kernel, normalize_kernel=True)
      
      noise_array.append(array)
      
      F1 = np.fft.fftn(array)
     
      F2 = np.fft.fftshift( F1 )

    # Calculate a 2D power spectrum
      psd2D = np.abs( F2 )**2      #
      
# Calculate the azimuthally averaged 1D power spectrum
      p = radial_data(psd2D, annulus_width = 1 )
      #plt.loglog(p.mean)

      psd1D.append(p.mean)
        
   c = psd1D[0]
   d = noise_array[0]
   
   for i in range(len(image[0])-1):
       c = np.add(c, psd1D[i+1]) 
       d = np.add(d, noise_array[i+1])
   ps = c/len(image[0]) 

   return ps, d


def density_map (image):
    
    hdu = fits.open(image)[0]
    hd = hdu.header
    a = hdu.data 
    
    a[np.isnan(a)] = 0

    
    d = np.count_nonzero(a)
    c = np.sum(a)
    
    av_cd = c/d
    
    # pixels=np.where(a != 0) 
    
    # av_cd =  np.sum(a)/len(pixels[0])

    b = a/av_cd

    fits.writeto('density_map.fits',b, hd, overwrite = True)


def variance_2 (image):
    
    hdu = fits.open(image)[0]
    hd = hdu.header
    a = hdu.data 
     
    c = np.count_nonzero(a)
 
    n0 = np.sum(a)/c #mean value 
    
    d = a-n0
    
    e = np.multiply(d,d)
    
    v = sum(e)
    
    
    return np.sum(e)/c


def remove_noise (emission_map, c, sigma): 
    
    hdu = fits.open(emission_map)[0]
    hd = hdu.header
    a = hdu.data 
    
    a[a<=c*sigma] = 0
        
    fits.writeto('cloud_no_noise.fits',a, hd, overwrite = True)

    return ('Noise filtered')
    


def cloud_emission (cloud_map, emission_map):
    'from a cloud cube and emission map, extracts the emission of the cloud'
    
       
    hdu1 = fits.open(cloud_map)[0]
    hd1 = hdu1.header
    a1 = hdu1.data 
    
    #make a mask for the cloud
    a1[a1 != 0] = 1
    
    hdu2 = fits.open(emission_map)[0]
    hd2 = hdu2.header
    a2 = hdu2.data 
    a2[np.isnan(a2)] = 0


    a1 = np.multiply(a1, a2)

    
    fits.writeto('cloud_emission.fits', a1, hd1, overwrite=True)



def cloud_density (cloud_map, density_map):
    'from a cloud cube and CD cube extracts the emission of the cloud'
    
       
    hdu1 = fits.open(cloud_map)[0]
    hd1 = hdu1.header
    a1 = hdu1.data 
    a1[np.isnan(a1)] = 0

     
    
    #make a mask for the cloud
    a1[a1 != 0] = 1
    
    hdu2 = fits.open(density_map)[0]
    hd2 = hdu2.header
    a2 = hdu2.data 
    a2[np.isnan(a2)] = 0

    a3 = np.multiply(a2, a1)
    
    fits.writeto('cloud_density.fits', a3, hd1, overwrite=True)


def av_vel (cloud):
    
    hdu=fits.open(cloud)[0]
    hd=hdu.header
    a=hdu.data 
    
    # Read in the header keywords
#v
    crpix3=hd["CRPIX3"]
    crval3=hd["CRVAL3"]
    cdelt3=hd["CDELT3"]
    naxis3=hd["NAXIS3"]
    
    x=[None]*naxis3
    
    for i in range(0, naxis3):
                x[i]=((i-(crpix3-1))*cdelt3+crval3)
        
    
    pixels=np.where(a != 0) 
    
    v = [0]*len(pixels[0])
    
    for i in range(len(pixels[0])):
        print (i)
        
        v[i] = x[pixels[0][i]]
    
    return sum(v)/len(v)

