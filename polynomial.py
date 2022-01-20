# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:37:36 2021

@author: Ibrahim SHAABAN
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ===============Define some variables=============== #
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
   
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = [None, None, None] 
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #Activated lane pixel indices
        self.lane_inds = [None] 
        #number of N previous fits
        self.max_fits = 5
        
    def addFit (self, new_fit):
        self.recent_xfitted.append(new_fit)
        if (len(self.recent_xfitted) > (self.max_fits)):
            self.recent_xfitted.pop(0)
        self.current_fit = np.array(new_fit)
        self.calcBestFit()
        return
    
    def calcBestFit(self):
        sum = [0, 0, 0]
        num_fits = min(self.max_fits, len(self.recent_xfitted))
        for idx in range(num_fits):
            sum[0] += self.recent_xfitted[idx][0]
            sum[1] += self.recent_xfitted[idx][1]
            sum[2] += self.recent_xfitted[idx][2]
            
        self.best_fit[0] = sum[0]/num_fits
        self.best_fit[1] = sum[1]/num_fits
        self.best_fit[2] = sum[2]/num_fits
        
    
    
def getLanesBase(img):
    '''
    Parameters
    ----------
    img : Image
        Wrapped binary image.

    Returns
    -------
    leftx_base : int
        Start index of the left lane line, at the bottom og the image.
    rightx_base : int
        Start index of the right lane line, at the bottom og the image.

    '''
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    return leftx_base, rightx_base

def find_lane_pixels(binary_warped, left_lane_line, right_lane_line):
    '''
    Parameters
    ----------
    binary_warped : Image
        Binary wrapped image.
    left_lane_line : object
        Left line object to store the computed data.
    right_lane_line : object
        Right line object to store the computed data.

    Returns
    -------
    left_lane_inds : list
        Left lane active pixels.
    right_lane_inds : TYPE
        Right lane active pixels.
    nonzero : Image_like matrix
        Contains the active pixels of the image
    

    '''
    # Take a histogram of the bottom half of the image
    leftx_base, rightx_base = getLanesBase(binary_warped)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ###Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ##If you found > minpix pixels, recenter next window ###
        if len(good_left_inds) >= minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) >= minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        ### (`right` or `leftx_current`) on their mean position ###

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    return left_lane_inds, right_lane_inds, nonzero


def fit_polynomial(binary_warped, left_lane_line, right_lane_line):
    '''
    Parameters
    ----------
    binary_warped : Image
        Binary wrapped image.
    left_lane_line : object
        Left line object to store the computed data.
    right_lane_line : object
        Right line object to store the computed data.

    Returns
    -------
    None.

    '''
    # Find our lane pixels first
    left_lane_inds, right_lane_inds, nonzero\
        = find_lane_pixels(binary_warped, left_lane_line, right_lane_line)
     
    # Extract left and right line pixel positions.
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    ### Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty , leftx , 2)
    right_fit = np.polyfit(righty , rightx , 2)        
    
    #Calculating the Curvature of both lines and assigning all the computed
    #data to the objects of the lane line
    left_lane_line.radius_of_curvature, right_lane_line.radius_of_curvature =\
        calcCurverad(ploty, leftx, lefty, rightx, righty)
    left_lane_line.addFit(left_fit)
    right_lane_line.addFit(right_fit)
    left_lane_line.detected = True
    right_lane_line.detected = True
    left_lane_line.lane_inds = left_lane_inds
    right_lane_line.lane_inds= right_lane_inds
    
    # Assign line distance to center
    right_lane_line.line_base_pos = right_fit[0]*binary_warped.shape[0]**2 +\
        right_fit[1]*binary_warped.shape[0] + right_fit[2]
    left_lane_line.line_base_pos = left_fit[0]*binary_warped.shape[0]**2 +\
        left_fit[1]*binary_warped.shape[0] + left_fit[2]
    return 


def search_around_poly(binary_warped, left_lane_line, right_lane_line):
    '''
    Parameters
    ----------
    binary_warped : Image
        Binary wrapped image.
    left_lane_line : object
        Left line object to store the computed data.
    right_lane_line : object
        Right line object to store the computed data.

    Returns
    -------
    None.

    '''
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100
    
    #Get left and right line fits
    left_fit = left_lane_line.best_fit
    right_fit = right_lane_line.best_fit
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ###Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_fitx = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]
    right_fitx = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]
    left_lane_inds = (np.absolute(nonzerox - left_fitx) <= margin)
    right_lane_inds = (np.absolute(nonzerox - right_fitx) <= margin)
    #If the number of indices is not sufficient, return false
    min_inds = 15
    left_lane_line.detected = left_lane_inds.shape[0] > min_inds
    right_lane_line.detected = right_lane_inds.shape[0] > min_inds
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    if len(lefty) == 0:
        #Failed to assign lefty, though lenths of nonzeroy and left_lane_inds are not zeros
        return
    
    
    
    # Fit new polynomials
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    ### Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty , leftx , 2)
    right_fit = np.polyfit(righty , rightx , 2)
    #Do a sanity check for the found fits, according to the distance between 
    #the 2 lines, if the reading are invalid, reset the detect flag for both lines
    if (sanityChk(binary_warped, left_fit, right_fit) != True):
        left_lane_line.detected = False
        right_lane_line.detected = False
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    left_curve, right_curve =calcCurverad(ploty, leftx, lefty, rightx, righty)
    
    #If the lane line is detected correctly, Assign the new fit, line indices, 
    #curvature and distance to center
    if left_lane_line.detected == True :
        left_lane_line.addFit(left_fit)
        left_lane_line.lane_inds = left_lane_inds
        left_lane_line.radius_of_curvature = left_curve
        # Assign line distance to center
        left_lane_line.line_base_pos = left_fit[0]*binary_warped.shape[0]**2 +\
        left_fit[1]*binary_warped.shape[0] + left_fit[2]
    
    # else:
    #     #print("Failed to detect Left lane")
        
    if right_lane_line.detected == True :
        right_lane_line.addFit(right_fit)
        right_lane_line.lane_inds = right_lane_inds
        right_lane_line.radius_of_curvature = right_curve
        right_lane_line.line_base_pos = right_fit[0]*binary_warped.shape[0]**2 +\
        right_fit[1]*binary_warped.shape[0] + right_fit[2]
    # else:
    #     #print("Failed to detect Right lane")
    return

def calcCurverad(ploty, leftx, lefty, rightx, righty):
    '''
    Parameters
    ----------
    ploty : numpy linspace
        Points from 0 to maximum size of Y axis
    leftx : list
        Left X pixel positions.
    lefty : list
        Left Y pixel positions.
    rightx : list
        Right X pixel positions.
    righty : list
        Right Y pixel positions.

    Returns
    -------
    left_curverad : int
        left curve radius in meters.
    right_curverad : int
        Right curve radius in meters.

    '''
    #Get the left and right polynomial fit coef.
    left_fit = np.polyfit(lefty*ym_per_pix , leftx*xm_per_pix , 2)
    right_fit = np.polyfit(righty*ym_per_pix , rightx*xm_per_pix, 2)
    y_eval = np.max(ploty)
    
    #Calculating the curve radius using the formula
    #           [1+(dy/dx)^2]^(3/2)
    #Rcurve = ------------------------
    #               [d2x/dy2]
    left_curverad = ((1+(2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**(3/2))\
        /abs(2*left_fit[0])  ## Implement the calculation of the left line here
        
    right_curverad = ((1+(2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**(3/2))\
        /abs(2*right_fit[0])  ## Implement the calculation of the left line here
    
    return left_curverad, right_curverad

def sanityChk(binary_warped, left_fit, right_fit):
    '''
    Parameters
    ----------
    binary_warped : Image
        Binary wrapped image.
    left_fit : list
        Polynomial coef for the left line.
    right_fit : list
        Polynomial coef for the right line.

    Returns
    -------
    bool
        Boolean representing if the line are valid or not.

    '''
    #lane witdth conversion from m to pixels
    lane_width = 700
    #Setting 3 test points, bottom, middle and top
    threepoints = np.linspace(0, binary_warped.shape[0]-1, 3)
    
    #Computing the 3 points for left and right polynomial
    left_lane_line_pos = left_fit[0]*threepoints**2 +\
    left_fit[1]*threepoints + left_fit[2]
    right_lane_line_pos = right_fit[0]*threepoints**2 +\
    right_fit[1]*threepoints + right_fit[2]
    
    #Setting thrsholds for comparison with 15% tolerance
    upper_limit = lane_width *1.15
    lower_limit = lane_width *0.85   
    
    #If the values of the differences are out of the limits, then these are
    #invalid fits, but if they fall in the defined boundaries, the they are valid
    diff = [right_lane_line_pos[0]-left_lane_line_pos[0],
            right_lane_line_pos[1]-left_lane_line_pos[1],
            right_lane_line_pos[2]-left_lane_line_pos[2]]
    if (diff[0] <= upper_limit) and (diff[0] >= lower_limit) and\
        (diff[1] <= upper_limit) and (diff[1] >= lower_limit) and\
        (diff[2] <= upper_limit) and (diff[2] >= lower_limit):
            return True
    else:
            return False
        
def distToCenter(left_center, right_center, img_width):
    '''
    Parameters
    ----------
    left_center : int
        Center of the left line at the bottom of the image.
    right_center : int
        Center of the right line at the bottom of the image.
    img_width : int
        Number of X pixels in the image

    Returns
    -------
    Int
        Position from the center of the lane, negative values are closer to the
        left and positive values are closer to the right.

    '''
    lane_midpoint = (left_center + right_center) / 2
    vehi_midpoint = img_width/2    
    return (vehi_midpoint - lane_midpoint)*3.7/700

def Visualize(binary_warped, left_lane_line, right_lane_line):
    '''
    Parameters
    ----------
    binary_warped : Image
        Binary wrapped image.
    left_lane_line : object
        Left line object having lane pixel indices and fit to be used for visuallization.
    right_lane_line : object
        Right line object having lane pixel indices and fit to be used for visuallization.

    Returns
    -------
    result : image
        Output wrapped Image with visualized lane lines. 

    '''
    left_lane_inds = left_lane_line.lane_inds
    right_lane_inds = right_lane_line.lane_inds
    left_fit = left_lane_line.current_fit
    right_fit = right_lane_line.current_fit
    width = 100
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-width, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+width, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-width, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+width, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    return result
   
if __name__ == '__main__':
    import CamCal
    import Thresholding
    import glob
    input_img_dir = ['camera_cal\calibration*.jpg', 
                    ['test_images\straight_lines*.jpg', 'test_images\\test*.jpg']]
    output_dir = ['output_images\\test_image_wrapped', 'output_images\\test_image_original']
    img_list = glob.glob(input_img_dir[1][0])+ glob.glob(input_img_dir[1][1])
    #Get camera calibration coefficients
    matrix, dist = CamCal.getCalibCoeff(input_img_dir[0])
    left_lane_line = Line()
    right_lane_line = Line()
    
#===============================Start of example images pipeline===============================#
    for idx in range(len(img_list)):
        #load the image
        img = mpimg.imread(img_list[idx])
        #apply undistortion
        undist = CamCal.undistort(img, matrix, dist)        
        #Get binary image after combined thresholding
        thresh_img = Thresholding.combined_thresholds(undist,(20, 150), (50, 150),
                                                      sat_thresh=(150, 255) )
        #wraping the binary image
        wrapped_image = CamCal.wrap(thresh_img)
        #getting a polonomial fit
        fit_polynomial(wrapped_image, left_lane_line, right_lane_line)
        line_drawing = Visualize(wrapped_image, left_lane_line, right_lane_line)
    
        plt.imshow(line_drawing, cmap='gray')
        plt.savefig(output_dir[0] + str(idx) +'.png')
        plt.close()
        #applying wrap inverse
        road_curve = (left_lane_line.radius_of_curvature + left_lane_line.radius_of_curvature) /2
        proc_img = CamCal.unwrap(wrapped_image, undist, 
                                 left_lane_line.current_fit, right_lane_line.current_fit)
        #writing the curvature and the vehicle offset to the output image
        road_curve = (left_lane_line.radius_of_curvature +\
                      right_lane_line.radius_of_curvature) /2
        if road_curve < 7000:
            label_str = 'Radius of curvature: %.1f m' % road_curve
        else:
            label_str = 'Radius of curvature: No curvature, Straightline detected'
        proc_img = cv2.putText(proc_img, label_str, (30,40), 0, 1,
                               (255,0,0), 2, cv2.LINE_AA)
        veh_offset = distToCenter(left_lane_line.line_base_pos,
                                  right_lane_line.line_base_pos, wrapped_image.shape[1])
        if veh_offset < 0 : 
            label_str = 'Vehicle offset from lane center: %.2f m toward left \
                        lane line' % abs(veh_offset)
        elif veh_offset > 0:
            label_str = 'Vehicle offset from lane center: %.2f m toward right \
                        lane line' % veh_offset
        elif veh_offset == 0:
            label_str = 'Vehicle is at the center of the lane'
        proc_img = cv2.putText(proc_img, label_str, (30,80), 0, 1,
                               (255,0,0), 2, cv2.LINE_AA)
        
        plt.imshow(proc_img)
        plt.savefig(output_dir[1] + str(idx) +'.png')
        plt.close()
    
                