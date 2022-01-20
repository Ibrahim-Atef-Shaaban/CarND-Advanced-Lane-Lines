# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:54:26 2021

@author: Ibrahim SHAABAN
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import datetime 
#========================Define some variables========================#
Mniv= [None]

def getCalibCoeff (img_dir = 'camera_cal\calibration*.jpg', nx = 9, ny = 6):
    """
    

    Parameters
    ----------
    img_dir : String, optional
        A path to where the chessboard calibration images are saved.
        The default is 'camera_cal\calibration*.jpg'.
    nx : int, optional
        number of chessboard corners in X direction.
    ny : int, optional
        number of chessboard corners in Y direction.

    Returns
    -------
    matrix : 3X3 list
        Camera matrix.
    dist : TYPE
        Distortion Coefficient of the camera .

    """
    #Read images in the directory
    img_list = glob.glob(img_dir)
    #Set empty lists for object and image points
    object_pts = [] #3D of image in real worled
    image_pts = [] #2D of image taken by the camera
    #Creating a meshgrid for how the chessboard corners look like in real world 
    objectp = np.zeros((nx*ny, 3), np.float32)
    objectp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    #Iterate over all images
    for idx in range(len(img_list)):
        #Read image
        img = mpimg.imread(img_list[idx])
        
        #Get the X and Y pixels in the image
        img_size = (img.shape[1], img.shape[0])
        
        #convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        #Get the image chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        if ret:
            #Store the 3d object poits and there equivalent image points
            object_pts.append(objectp)
            image_pts.append(corners)
            
            #Draw chessboard corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    
    #If there are bot image and object poits, get the calibration Coefficients
    #of the camera and return them
    if object_pts and image_pts:
        _, matrix, dist, _, _ = cv2.calibrateCamera(object_pts, image_pts,
                                                    img_size, None, None)
    return matrix, dist

def undistort(img, matrix, dist, save_img = False, output_dir = 'Undistorted_images/'):
    """
    Parameters
    ----------
    img : Image
        This is the input image for the function, to be undistorted.
    matrix : 3*3 array
        This reporesents the camera matrix.
    dist : 
        distortion coeffitients of the camera.
    save_img: boolean
        If this flag is set, save the image at the specified directory with timestamp
    output_dir : system path
        optional path to save the undistorted images at.

    Returns
    -------
    undist : Image
        This is the return of the function which represents undistorted image.

    """
    #Use the calibration coef. to undistort a given image
    undist = cv2.undistort(img, matrix, dist, None, matrix)
    
    #Save the file if needed
    if save_img == True:
        current_time = datetime.datetime.now()  
        timestamp = str(current_time.year) + str(current_time.month) + \
                    str(current_time.day) + str(current_time.hour) + \
                    str(current_time.minute) + str(current_time.second)
        plt.imshow(undist)
        plt.savefig(output_dir + timestamp +'.png')
    return undist

def wrap(undist, save_img = False, output_dir = 'Warped_images/'):
    """
    

    Parameters
    ----------
    undist : Image
        This represetns the undistorted image.
    save_img : Boolean, optional
        Set it to true, to save the output files to the output directory.
    output_dir : String, optional
        A path to save the output if needed.

    Returns
    -------
    warped : Image
        This represents the warped Image.

    """
    
    global Minv
    #Setting variable to indetify the trapezium_pts, for the specified image size
    xdim = undist.shape[1]
    ydim = undist.shape[0]
    offset = 200
    
    #Identifying the trapezium(src) and rectangle(dst) points
    trapezium_pts = np.float32([[xdim/2 -offset/4, ydim * 0.625], #Top left
                            [xdim/2 +offset/4, ydim * 0.625], #Top right
                            [xdim - offset, ydim], #Bottom right
                            [offset, ydim]]) #Bottom left

    rectangle_pts = np.float32([[xdim/4, 0], #Top left
                                [xdim - (offset*3/2), 0], #Top right
                                [xdim - (offset*3/2), ydim], #Bottom right
                                [xdim/4, ydim]]) #Bottom left
    #Get the Prespective coef. and its inverse to use it later for the unwrapping
    M = cv2.getPerspectiveTransform(trapezium_pts, rectangle_pts)
    Minv = cv2.getPerspectiveTransform(rectangle_pts, trapezium_pts)
    
    #Wrapping the image
    warped = cv2.warpPerspective(undist, M, (xdim, ydim))
    
    #Saving the image if needed and returning the wrapped image
    if save_img == True:
        current_time = datetime.datetime.now()  
        timestamp = str(current_time.year) + str(current_time.month) + \
                    str(current_time.day) + str(current_time.hour) + \
                    str(current_time.minute) + str(current_time.second)
        plt.imshow(warped)
        plt.savefig(output_dir + timestamp +'.png')
    return (warped)

def unwrap(warped, undist, left_lane_fit, right_lane_fit):
    '''
    Parameters
    ----------
    warped : Image
        Wrapped Image.
    undist : Image
        Undistorted Image.
    left_lane_fit : list
        Left lane line polynomial fit coef.
    right_lane_fit : list
        Right lane line polynomial fit coef.

    Returns
    -------
    result : Image
        Unwrapped image with the lane lines drawn on it.

    '''
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0] )
    left_fitx = left_lane_fit[0]*ploty**2 +\
                left_lane_fit[1]*ploty + left_lane_fit[2]
    right_fitx = right_lane_fit[0]*ploty**2 +\
                 right_lane_fit[1]*ploty + right_lane_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int32([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

if __name__ == '__main__':
    input_img_dir = ['camera_cal\calibration*.jpg', 
                 ['test_images\straight_lines*.jpg', 'test_images\\test*.jpg']]
    
    matrix, dist = getCalibCoeff(input_img_dir[0])
    
    img_list = glob.glob(input_img_dir[1][0])+ glob.glob(input_img_dir[1][1])
    
    for idx in range(len(img_list)):
        img = mpimg.imread(img_list[idx])
        undist = undistort(img, matrix, dist, save_img = True)
        wrapped_image = wrap(undist, True)
        plt.imshow(wrapped_image)
        
    
    