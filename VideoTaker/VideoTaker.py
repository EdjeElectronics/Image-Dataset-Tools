######## Simple Video Recording Script #########
#
# Author: Evan Juras, EJ Technology Consultants
# Date: 8/8/21
# Description: 
# This program records a video file (.avi format) from a connected webcam.
#
# Example usage to record a video named 'test1.avi' at 1280x720 resolution and 10 FPS:
# python3 VideoTaker.py --vidname=test1.avi

import cv2
import os
import sys
import argparse

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--vidname', help='Filename to save the video as (must have a .avi appendix',
                   default='demo.avi')
parser.add_argument('--resolution', help='Desired camera resolution in WxH.',
                   default='1280x720')
parser.add_argument('--FPS', help='Approximate FPS to record video at.',
                   default='10')

args = parser.parse_args()

# Get video filename and check if it already exists
vidname = args.vidname
cwd = os.getcwd()
filepath = os.path.join(cwd,vidname)
if os.path.exists(filepath):
    response = input('The file %s already exists! Do you want to overwrite it? (y/n) ' % vidname)
    if (response == 'y') or (response == 'Y'):
        print('%s will be overwritten.' % vidname)
    else:
        print('Please restart the program and use a different filename.')
        sys.exit()

# Get desired resolution and framerate
imW = int(args.resolution.split('x')[0])
imH = int(args.resolution.split('x')[1])
FPS = int(args.FPS)
wait_time = int((1/FPS)*1000) # Determine wait time from desired FPS

# For capturing a demo video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outVid = cv2.VideoWriter(vidname, fourcc, FPS, (imW,imH))

# Initialize webcam
index = 0 # Change index to 1, 2, etc. if there are multiple cameras connected to computer
cap = cv2.VideoCapture(index)
ret = cap.set(3, imW)
ret = cap.set(4, imH)

print('Recording video! Press q to quit.')

# Continuously grab frames from webcam and append to video file
while True:
    hasFrame, frame = cap.read()
    cv2.imshow('Camera',frame)
    outVid.write(frame)

    key = cv2.waitKey(wait_time) # Wait time in milliseconds to achieve approximate desired framerate
    if key == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
outVid.release()
cap.release()
