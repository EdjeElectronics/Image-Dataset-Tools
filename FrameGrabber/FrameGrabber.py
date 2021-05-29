######## Video Frame Grabber #########
#
# Author: Evan Juras, EJ Technology Consultants
# Date: 5/15/21
# Description: 
# FrameGrabber.py plays through a video file, extracts individual frames from it, and saves the exracted frames as images.
# The SKIP_FRAMES variable can be used to set how many images are extracted from the video.
# Example: Setting SKIP_FRAMES = 90 for a 30FPS video will extract one frame for every three seconds of video.

# Import necessary packages
import os
import cv2

# User-defined settings
VIDEO_NAMES = ['bison-video.mp4', 'wolf-video.mp4']  # Names of video files to extract frames from
OUTPUT_FOLDER_NAME = 'extracted_pics' # Name of folder to save extracted images to
SKIP_FRAMES = 90 # Number of frames to skip between extracted images
RESIZE_FACTOR = 0.5 # Factor to resize images to when saving

# Set up filepaths
CWD_PATH = os.getcwd()
FOLDER_PATH = os.path.join(CWD_PATH,OUTPUT_FOLDER_NAME)

# If specified folder doesn't exist, create it
if not os.path.isdir(FOLDER_PATH):
    os.mkdir(FOLDER_PATH)



# Go through each video and extract frames
for file in VIDEO_NAMES:
    
    # Get the file path and base name for the file
    filepath = os.path.join(CWD_PATH,file)
    basefn = file.split('.')[0]
    
    # Open video file
    video = cv2.VideoCapture(filepath)
    
    # Program counters
    im_count = 0
    frame_count = 0

    # Go through each frame in the video
    while (video.isOpened()):
        
        # Check if the end of the video has been reached. If it has, wait for user keypress
        hasFrame, frame = video.read()
        if not hasFrame:
            print('Reached end of %s !' % file)
            cv2.waitKey()
            break

        # Increment frame count
        frame_count = frame_count + 1
        
        # If enough frames have passed, grab this frame and save it as an image
        if frame_count == SKIP_FRAMES:
            # Resize the frame
            frame = cv2.resize(frame,None,fx=RESIZE_FACTOR,fy=RESIZE_FACTOR)
            
            # Save the frame
            im_count = im_count + 1
            im_name = basefn + str(im_count) + '.jpg'
            savepath = os.path.join(FOLDER_PATH,im_name)
            cv2.imshow('Extracted image',frame)
            cv2.waitKey(10)
            cv2.imwrite(savepath, frame)
            frame_count = 0

    # Close the video file
    video.release()
    video = []

# All done!
cv2.destroyAllWindows()
