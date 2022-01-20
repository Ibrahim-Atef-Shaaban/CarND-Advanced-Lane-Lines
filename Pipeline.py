import CamCal
import Thresholding
import polynomial
import cv2
from moviepy.editor import VideoFileClip


max_fail_frames = 5
left_failure_counter = max_fail_frames
right_failure_counter = max_fail_frames

#======================Start of video Pipeline======================#
camcalibration_imgs = 'camera_cal\calibration*.jpg'
matrix, dist = CamCal.getCalibCoeff(camcalibration_imgs)
left_lane_line = polynomial.Line()
right_lane_line = polynomial.Line()


def Pipeline(img):
    global max_fail_frames, left_failure_counter, right_failure_counter,\
        left_lane_line, right_lane_line, dist, matrix
    #apply undistortion
    undist = CamCal.undistort(img, matrix, dist)        
    #Get binary image after combined thresholding
    thresh_img = Thresholding.combined_thresholds(undist,(20, 100), (50, 100),
                                                  sat_thresh=(150, 255) )
    #wraping the binary image
    wrapped_image = CamCal.wrap(thresh_img)
    #getting a polonomial fit
    if max(left_failure_counter, right_failure_counter) >= max_fail_frames:
        polynomial.fit_polynomial(wrapped_image, left_lane_line, right_lane_line)
        right_failure_counter = 0
        left_failure_counter = 0
    else:
        polynomial.search_around_poly(wrapped_image, left_lane_line, right_lane_line)
    if left_lane_line.detected == False:
        left_failure_counter += 1
    else:
        left_failure_counter = 0
        
    if right_lane_line.detected == False:
        right_failure_counter += 1
    else:
        right_failure_counter = 0
    
     
    
    road_curve = (left_lane_line.radius_of_curvature + right_lane_line.radius_of_curvature) /2
    #applying wrap inverse
    proc_img = CamCal.unwrap(wrapped_image, undist, 
                             left_lane_line.current_fit, right_lane_line.current_fit)
    #writing the curvature and the vehicle offset to the output image
    road_curve = (left_lane_line.radius_of_curvature +\
                  left_lane_line.radius_of_curvature) /2
    label_str = 'Radius of curvature: %.1f m' % road_curve
    proc_img = cv2.putText(proc_img, label_str, (30,40), 0, 1,
                           (255,0,0), 2, cv2.LINE_AA)
    veh_offset = polynomial.distToCenter(left_lane_line.line_base_pos,
                              right_lane_line.line_base_pos, wrapped_image.shape[1])
    if veh_offset < 0 : 
        label_str = 'Vehicle offset from lane center: %.2f m toward left lane line' % abs(veh_offset)
    elif veh_offset > 0:
        label_str = 'Vehicle offset from lane center: %.2f m toward right lane line' % veh_offset
    elif veh_offset == 0:
        label_str = 'Vehicle is at the center of the lane'
    proc_img = cv2.putText(proc_img, label_str, (30,80), 0, 1,
                           (255,0,0), 2, cv2.LINE_AA)
    return proc_img

input_video = "project_video.mp4"
white_output = 'test_videos_output/'+ input_video +'.mp4'
clip1 = VideoFileClip(input_video)#.subclip(0,2)
white_clip = clip1.fl_image(Pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

input_video = "challenge_video.mp4"
white_output = 'test_videos_output/'+ input_video +'.mp4'
clip1 = VideoFileClip(input_video)#.subclip(0,2)
white_clip = clip1.fl_image(Pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

input_video = "harder_challenge_video.mp4"
white_output = 'test_videos_output/'+ input_video +'.mp4'
clip1 = VideoFileClip(input_video)#.subclip(0,2)
white_clip = clip1.fl_image(Pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)