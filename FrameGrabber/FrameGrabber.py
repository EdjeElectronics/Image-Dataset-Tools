######## Video Frame Grabber #########
#
# Author: Evan Juras
# Date: 5/15/21
# Description: 
# FrameGrabber.py plays through a video file, extracts individual frames from it, and saves the exracted frames as images.
# The SKIP_FRAMES variable can be used to set how many images are extracted from the video. For example, setting
# SKIP_FRAMES = 90 for a 30FPS video will extract one frame for every three seconds of video.

import os
import cv2


CWD_PATH = os.getcwd()
VIDEO_NAMES = ['bison-video.mp4']
FOLDER_PATH = os.path.join(CWD_PATH,'extracted_pics')

SKIP_FRAMES = 90
im_count = 0

# Go through each video and extract frames
for file in VIDEO_NAMES:
    filepath = os.path.join(CWD_PATH,file)

    # Open video file
    video = cv2.VideoCapture(filepath)

    frame_count = 0

    while (video.isOpened()):
        hasFrame, frame = video.read()
        if not hasFrame:
            print('Reached end of %s !' % file)
            cv2.waitKey()
            break

        frame_count = frame_count + 1
        
        if frame_count == SKIP_FRAMES:
            # Resize the frame
            frame = cv2.resize(frame,(640,360))
            
            # Save the frame
            im_count = im_count + 1
            im_name = 'BisonPic' + str(im_count) + '.jpg'
            fp = os.path.join(FOLDER_PATH,im_name)
            cv2.imshow('Bison',frame)
            cv2.waitKey(10)
            cv2.imwrite(fp, frame)
            frame_count = 0

    video.release()
    video = []


cv2.destroyAllWindows()
