# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:53:18 2021

@author: Ibrahim SHAABAN
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', abs_thresh=(20, 100), k_size = 5):
    '''
    Parameters
    ----------
    img : Image
        Input raw image
    orient : char, optional
        With values x or y, this selects the direction of the sobel..
    abs_thresh : int tuple, optional
        Low and high thresholds for the sobel, respectively.
    k_size : int, optional
        Kernel size for sobel. The default is 5.

    Returns
    -------
    binary_output : Image
        Binary Image with sobel threshold applied.
    sobel : list
        sobel list to be used lated by the magnitude of the direction of the gradient.

    '''
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = k_size)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = k_size)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output, sobel

def magn_threshold(sobelx, sobely, mag_thresh=(30, 100)):
    '''
    Parameters
    ----------
    sobelx : list
        The already calculated sobel in the X direction.
    sobely : list
        The already calculated sobel in the Y direction.
    mag_thresh : int tuple, optional
        Low and high thresholds for the magnitude of the sobel, respectively.

    Returns
    -------
    binary_output : Image
        Binary Image with sobel magnitude threshold applied.

    '''
    # 1) Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # 2) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 3) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel<=mag_thresh[1])] = 1
    # 4) Return this mask as your binary_output image
    return binary_output

def dir_threshold(sobelx, sobely, dir_thresh=(0.7, 1.3)):
    '''
    Parameters
    ----------
    sobelx : list
        The already calculated sobel in the X direction.
    sobely : list
        The already calculated sobel in the Y direction.
    dir_thresh : int tuple, optional
        Low and high thresholds for the direction of the gradient, respectively.

    Returns
    -------
    binary_output : Image
        Binary Image with gradient direction threshold applied.
    '''
    
    # 1) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 2) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir  = np.arctan2(abs_sobely, abs_sobelx)
    # 3) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir <= dir_thresh[1]) & (grad_dir >= dir_thresh[0])] = 1
    # 4) Return this mask as your binary_output image
    return binary_output

def combined_sobel(img, abs_thresh=(30, 100), mag_thresh=(30, 100),
                    dir_thresh=(0.7, 1.3), k_size = 5):
    '''
    Parameters
    ----------
    img : Image
        Input raw image
    abs_thresh : int tuple, optional
        Low and high thresholds for the sobel, respectively.
    mag_thresh : int tuple, optional
        Low and high thresholds for the magnitude of the sobel, respectively.
    dir_thresh : int tuple, optional
        Low and high thresholds for the direction of the gradient, respectively.
    k_size : int, optional
        Kernel size for sobel. The default is 5.

    Returns
    -------
    combined_sob : TYPE
        DESCRIPTION.

    '''
    #Computing the Sobel in X, Y direction
    binary_x, sobelx = abs_sobel_thresh(img, 'x', abs_thresh, k_size)
    binary_y, sobely = abs_sobel_thresh(img, 'y', abs_thresh, k_size)
    
    #Computing the magnitude of the gradient
    binary_mag = magn_threshold(sobelx,sobely, mag_thresh)
    #Computing the direction of the gradient
    binary_dir = dir_threshold(sobelx,sobely, dir_thresh)
    #create an empty copy of the binary images
    combined_sob = np.zeros_like(binary_dir)
    #Combining all the thresholds output and returning a binray image 
    combined_sob[((binary_x == 1)  &  (binary_y == 1)) | ((binary_dir == 1) & (binary_mag == 1))] = 1
    return combined_sob
    
    
def hls_select(img, thresh=(90, 255)):
    '''
    Parameters
    ----------
    img : Image
        Input raw image
    thresh : int tuple, optional
        Low and high thresholds for the saturation level of the image, respectively.

    Returns
    -------
    binary_output : Image
        Binary Image with saturation threshold applied.

    '''
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s = hls[:,:,2]
    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def combined_thresholds(img, abs_thresh=(20, 100), mag_thresh=(30, 100),
                        dir_thresh=(0.7, 1.3), k_size = 5, sat_thresh=(90, 255)):
    '''
    Parameters
    ----------
    img : Image
        Input raw image.
    abs_thresh : int tuple, optional
        Low and high thresholds for the sobel, respectively.
    mag_thresh : int tuple, optional
        Low and high thresholds for the magnitude of the sobel, respectively.
    dir_thresh : int tuple, optional
        Low and high thresholds for the direction of the gradient, respectively.
    k_size : int, optional
        Kernel size for sobel. The default is 5.
    sat_thresh : int tuple, optional
        Low and high thresholds for the saturation level of the image, respectively.

    Returns
    -------
    combined_threshold : Image
        Binary Image with gradient and color thresholds applied.

    '''
    #Computing the gradient after thresholds
    sobel_thresholds = combined_sobel(img, abs_thresh, mag_thresh, dir_thresh, k_size)
    
    #Computing the color after thresholds
    color_threshold = hls_select(img, sat_thresh)
    
    #create an empty copy of the binary images
    combined_threshold = np.zeros_like(color_threshold)
    
    #Combining all the thresholds output and returning a binray image 
    combined_threshold[(color_threshold == 1) | (sobel_thresholds == 1)] = 1 
    return combined_threshold
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    from CamCal import *
    import glob
    input_img_dir = ['camera_cal\calibration*.jpg', 
                 ['test_images\straight_lines*.jpg', 'test_images\\test*.jpg']]
    output_dir = ['Thresholded\Thresholded', 'Thresholded\Thresholded_Wrapped']
    matrix, dist = getCalibCoeff(input_img_dir[0])
    img_list = glob.glob(input_img_dir[1][0])+ glob.glob(input_img_dir[1][1])
    for idx in range(len(img_list)):
        img = mpimg.imread(img_list[idx])
        undist = undistort(img, matrix, dist)
        thresh_img = combined_thresholds(undist,(20, 150), (50, 150),
                                         sat_thresh=(150, 255) )
        plt.imshow(thresh_img, cmap='gray')
        plt.savefig(output_dir[0] + str(idx) +'.png')
        plt.close()
        wrapped_image = wrap(thresh_img)
        plt.imshow(wrapped_image, cmap='gray')
        plt.savefig(output_dir[1] + str(idx) +'.png')
        
    