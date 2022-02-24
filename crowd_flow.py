# importing numpy,cv2 and matplotlib packages
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# To find the Streak Flow
def streak_flow(image, flow, i=9):
    # setting the Height and Width of the Frame
    h, w = image.shape[:2]
    y, x = np.mgrid[i/2:h:i,i/2:w:i].reshape(2,-1).astype(int)
    flowx, flowy = flow[y, x].T
    #np.vstack is used to stak the sequence of input arrays vertically to make a single array
    array_lines = np.vstack([x, y, x - flowx, y - flowy]).T.reshape(-1, 2, 2)
    array_lines = np.int32(array_lines+ 0.5)
    # Converting the image from GrayScale to BGR format
    bgr_format = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # cv.polylines() is used to draw polygonal curves on any image.
    cv.polylines(bgr_format, array_lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in array_lines:
        #cv.circle draws a simple or filled circle with a given center and radius
        cv.circle(bgr_format, (x1, y1), 1, (0, 255, 0), -1)
    return bgr_format


# Heatmap functions
def eroded_image(image):
    Grayimage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blured_image = cv.GaussianBlur(Grayimage, (5, 5), 25)
    img_canny = cv.Canny(blured_image, 5, 50)
    new_array = np.ones((3, 3))
    dilated_image = cv.dilate(img_canny, new_array, iterations=5)
    eroded_image = cv.erode(dilated_image, new_array, iterations=3)
    return eroded_image

def get_image_contours(image, img_original):
    image_contours = img_original.copy()
    contrs, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(image_contours, contrs, -1, (0, 255, 0), -1)
    # If you want to omit smaller contrs we will take only contourArea(cnt)>500
    # for cnt in contrs:
    #     if cv2.contourArea(cnt) > 500:
    #         cv2.drawContours(image_contours, [cnt], -1, (0, 255, 0), -1)
    return image_contours

#watershed code
def watershed_segmentation(image):
    #converting image to grayscale image
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    # applying thresholding makes all pixels value above 0 are set to 255 with inverted opposite case of threshold binary
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # Noise Removal
    new_array = np.ones((3,3),np.uint8)
    #cv.morphologyEx can perform advanced morphological transformations using an erosion and dilation
    transformed = cv.morphologyEx(thresh,cv.MORPH_OPEN,new_array, iterations = 2)
    # Background area
    dilated_bg = cv.dilate(transformed,new_array,iterations=3)
    # Foreground area
    distance_transform = cv.distanceTransform(transformed,cv.DIST_L2,5)
    ret, foreground_image = cv.threshold(distance_transform,0.7*distance_transform.max(),255,0)
    # Finding the unknown region
    foreground_image = np.uint8(foreground_image)
    #subtracting one image from another image
    unknown = cv.subtract(dilated_bg,foreground_image)
    #  labelling the Marker
    ret, markers = cv.connectedComponents(foreground_image)
    #Applying watershed algorithm and marking the regions segmented
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv.watershed(image,markers)
    image[markers == -1] = [0,255,255]
    #Displaying the segmented image
    resized_image = cv.resize(image, (612, 368))
    return resized_image


#Main function
#Reading the video to the variable captured_video
captured_video = cv.VideoCapture('2581-2_70.mov')
# Reading the first frame from the captured_video variable
# ret is a boolean variable that returns true if the frame is available.
# frame is an image array vector captured based on the default frames per second 
ret, frame = captured_video.read()
# Converting the frame to GrayScale and stored in gray_scale
gray_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# Creating a np array of similar size to the frame
hsv = np.zeros_like(frame)
hsv[..., 1] = 255
#creating a np array of given size initialized with zeros
heat_map = np.zeros(frame.shape[:-1])


while True:
    # Reading each frame from the video
    ret, image = captured_video.read()
    if not ret:
        print('No frames grabbed!')
        break
    #code for Dense optical flow
    # Converting the frame into GrayScale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Computing the Optical-flow from the each frame
    flow = cv.calcOpticalFlowFarneback(gray_scale, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computing the magnitude and angle from the flow of each frame
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converting the image from hsv to bgr format
    bgr_format = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # code for Heat_map
    diff = cv.absdiff(frame, image)
    image_contours = get_image_contours(eroded_image(diff), frame)
    heat_map[np.all(image_contours == [0, 255, 0], 2)] += 4 
    heat_map[np.any(image_contours != [0, 255, 0], 2)] -= 4
    heat_map[heat_map < 0] = 0
    heat_map[heat_map > 255] = 255
    mapped_image = cv.applyColorMap(heat_map.astype('uint8'), cv.COLORMAP_JET)
    # Making the present frame as previous frame
    gray_scale = gray
    #.imshow to show the results over each frame
    cv.imshow('original video', image)
    cv.imshow('Dense Optical Flow', bgr_format)
    cv.imshow('Instantaneous velocity vectors', streak_flow(gray, flow))
    cv.imshow("Similarities", mapped_image)
    cv.imshow('Watershed Segmentation', watershed_segmentation(image_contours))
    key = cv.waitKey(10)
    if key == ord('q'):
        break

# Releasing the Capture
captured_video.release()
# Destroying/clearing all the windows
cv.destroyAllWindows()
