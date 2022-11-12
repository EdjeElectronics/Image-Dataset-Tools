######## Automatic Image Labeling Using Darknet YOLO #########
#
# Author: Evan Juras
# Date: 10/23/22
# Description: 
# This program automatically labels object detection training images. It runs a pre-trained
# detection model to propose the bounding box and class of each object in each image. It
# then saves the label data in an XML file in Pascal VOC format. Users can accept or reject
# the proposed labels for each image. Reject images are moved to a separate folder for manual labeling.

# Recommended use: train an initial model using 200 images, then use that model with this program
# to assist with labeling remaining images.

# Import packages
import os
import cv2
import numpy as np
import sys
import glob

# Import darknet (change darknetPath the path where built darknet files are located)
darknetPath = 'C:\\darknet\\build\\darknet\\x64'
sys.path.append(darknetPath)

import darknet

### User-defined variables

# Define model parameters
MODEL_WEIGHTS = 'yolov4.weights'
MODEL_CONFIG = 'yolov4.cfg'
MODEL_META = 'cfg/coco.data'

# Define image folder names
FOLDER_NAME = 'images'
UNLABELED_DIR = 'unlabeled' # TO DO - make it so this is automatically created if folder doesn't exist
LABELED_DIR = 'labeled' # TO DO - make it so this is automatically created if folder doesn't exist

# Detection threshold
MIN_THRESH = 0.5
IOU_THRESH = 0.5

### Function and format definitions
# Define function to draw detection boxes and labels
def drawPred(draw_frame, classId, conf, x, y, w, h):

    # Convert coordinates from YOLO format to cv2 format
    left = int(round(x - (w / 2)))
    right = int(round(x + (w / 2)))
    top = int(round(y - (h / 2)))
    bottom = int(round(y + (h / 2)))
    
    # Draw a bounding box.
    cv2.rectangle(draw_frame, (left, top), (right, bottom), (10, 255, 0),3)

    #label = '%s: %.2f' % (classId, conf)
    label = '%s: %d%%' % (classId, int(conf))

    # Print a label of class. (Not needed in darknet mode)
    """
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)"""

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(draw_frame, (left, top - labelSize[1] - 12), (left + labelSize[0]+40, top + baseLine - 8), (255, 255, 255), cv2.FILLED)
    cv2.putText(draw_frame, label, (left, top-7), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

    return draw_frame

# Function to get image size
def get_size(imPath):
    img = cv2.imread(imPath)
    imH, imW, chan = img.shape
    return imH, imW

# Define XML file format for Pascal VOC annotation data

xml_body_1="""<annotation>
        <folder>{FOLDER}</folder>
        <filename>{FILENAME}</filename>
        <path>{PATH}</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>{WIDTH}</width>
                <height>{HEIGHT}</height>
                <depth>3</depth>
        </size>
"""
xml_object=""" <object>
                <name>{CLASS}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{XMIN}</xmin>
                        <ymin>{YMIN}</ymin>
                        <xmax>{XMAX}</xmax>
                        <ymax>{YMAX}</ymax>
                </bndbox>
        </object>
"""
xml_body_2="""</annotation>        
"""

# Function to create XML files
def create_xml(imPath, imBBs):

    # Create XML filename
    imFn = imPath.split('\\')[-1]
    xmlFn = imFn[0:len(imFn)-4] + '.xml'
    xmlPath = os.path.join(cwd_path,FOLDER_NAME,LABELED_DIR,xmlFn)

    # Get image size
    imH, imW = get_size(imPath)

    # Create XML file and write data
    with open(xmlPath,'w') as f:
        f.write(xml_body_1.format(**{'FOLDER':FOLDER_NAME, 'FILENAME':imFn, 'PATH':imPath,
                                     'WIDTH':imW, 'HEIGHT':imH}))

        for bbox in imBBs:
            f.write(xml_object.format(**{'CLASS':bbox[0], 'XMIN':bbox[1][0], 'YMIN':bbox[1][1],
                                          'XMAX':bbox[1][2], 'YMAX':bbox[1][3]}))

        f.write(xml_body_2)

    return xmlFn

### Darknet isn't as good non-max suppression (i.e. filtering overlapping bounding boxes), so we need our own functions for
### eliminating detection results that are right on top of eachother.
# Define function to calculate xmin, xmax, ymin, ymax, from x, y, w, h
def get_min_max(coords_wh):
    x, y, w, h = coords_wh
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    coords_minmax = [xmin, ymin, xmax, ymax]
    return coords_minmax

# Define function to calculate intersection over union (IoU)
# Read the following article for a better understanding of how IoU works:
# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iou(box_1, box_2):
    
    # Convert coordinates from x, y, w, h to xmin, ymin, xmax, ymax
    box_1_coords = get_min_max(box_1[2])
    box_1_xmin, box_1_ymin, box_1_xmax, box_1_ymax = box_1_coords

    box_2_coords = get_min_max(box_2[2])
    box_2_xmin, box_2_ymin, box_2_xmax, box_2_ymax = box_2_coords

    # Find width and height of overlap area
    width_of_overlap_area = min(box_1_xmax, box_2_xmax) - max(box_1_xmin, box_2_xmin)
    height_of_overlap_area = min(box_1_ymax, box_2_ymax) - max(box_1_ymin, box_2_ymin)
    
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    
    box_1_area = (box_1_ymax - box_1_ymin) * (box_1_xmax - box_1_xmin)
    box_2_area = (box_2_ymax - box_2_ymin) * (box_2_xmax - box_2_xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    
    if area_of_union == 0:
        return 0
    
    return area_of_overlap / area_of_union

# Define function to filter detections using IoU threshold
# YOLO tends to predict multiple bounding boxes over a single object. This function filters out extra bounding boxes.
def filter_iou(detections, iou_threshold):

    # First, sort detections by their confidence from highest to lowest
    detections = sorted(detections, key = lambda obj: float(obj[1]), reverse=True)

    # For each detection:
    for i in range(len(detections)):
        
        # If the confidence score is 0, skip this detection
        if detections[i][1] == 0:
            continue
        
        # Compare this detection to the remaining detections in the list
        for j in range(i + 1, len(detections)):

            # If the two detection boxes are overlapping enough,
            # get rid of the detection box with lower confidence by setting its score to 0
            if iou(detections[i], detections[j]) > iou_threshold:
                detections[j][1] = 0

    # Build a list of filtered detections
    true_detections = []
    for det in detections:
        if float(det[1]) > 0:
            true_detections.append(det)
    
    return true_detections


### Load YOLO model and other variables

# Full paths to image folders
cwd_path = os.getcwd()
img_dir = os.path.join(cwd_path,FOLDER_NAME)

# Load neural network into memory and get input width and height
network, class_names, class_colors = darknet.load_network(MODEL_CONFIG,MODEL_META,MODEL_WEIGHTS,batch_size=1)
darknet_width = darknet.network_width(network)
darknet_height = darknet.network_height(network)

# Misc program variables
winName = 'YOLOv4 detection results'
font = cv2.FONT_HERSHEY_SIMPLEX

# Grab names of every image in the folder
images = glob.glob(img_dir + '/*.jpg') + glob.glob(img_dir + '/*.png') + glob.glob(img_dir + '/*.bmp')
imnum = 0

for image_path in images:
    
    ### Pre-process image
    # Load image, convert frame to RGB, and resize it
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)

    # Reformat image into data type expected by darknet
    darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    ### Perform inference on image
    # Pass image into darknet YOLOv4 model to get detection results
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=MIN_THRESH)
    darknet.free_image(darknet_image)

    ### Post-process results
    # Need to re-scale bounding box coordinates to match original image.
    # Detections come in this format: [('label1','score1',(x, y, w, h)),('label2','score2',(x ,y w, h))] etc

    # Initialize array for holding converted bounding box coordinates
    detections_true = []
    image_results = np.copy(image)

    # Go through each detection result
    for label, confidence, bbox in detections:

        # First, calculate normalized bounding box coordinates
        x, y, w, h = bbox
        x_norm = x/darknet_width
        y_norm = y/darknet_height
        w_norm = w/darknet_width
        h_norm = h/darknet_height

        # Then, multiply normalized coordinates by original image width and height to get true coordinates
        frame_h, frame_w = image.shape[0:2]
        x_true = int(x_norm * frame_w)
        y_true = int(y_norm * frame_h)
        w_true = int(w_norm * frame_w)
        h_true = int(h_norm * frame_h)
        bbox_true = (x_true, y_true, w_true ,h_true)

        # Create new detections array with true bounding box coordinates, draw results on frame
        detections_true.append([str(label), confidence, bbox_true])

    # Filter detections to remove overlapping bounding boxes
    detections_true = filter_iou(detections_true, IOU_THRESH)

    # Draw bounding box results
    for detection in detections_true:
        x, y, w, h = detection[2]
        image_results = drawPred(image_results, detection[0], float(detection[1]), x, y, w, h)

    ### Display results and ask user to accept or decline labels
    # Draw auto-labeler prompts
    cv2.putText(image_results,'Label good? (y/n)',(30,50),font,1,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(image_results,'Label good? (y/n)',(30,50),font,1,(0,255,0),2,cv2.LINE_AA)

    # Display the image
    cv2.imshow('Label attempt', image_results)

    # Wait for user to accept or decline label
    saveLabels = False
    stop = False

    key = cv2.waitKey() # Wait for user keypress
    if key == ord('y'): # Press y to accept label
        saveLabels = True
    elif key == ord('n'): # Press n to decline label
        saveLabels = False
    elif key == ord('q'): # Press q to quit
        saveLabels = False
        stop = True

    # Break out of the loop and close the program if user presses q
    if stop:
        break

    ### Save label data (if accepted by user)
    if saveLabels:
        bboxdata = []
        # Loop over every object, save class and bounding box coordinates
        for detection in detections_true:

            # Class
            label = detection[0]

            # Convert coordinates from YOLO format to cv2 format
            x, y, w, h = detection[2]
            coords = get_min_max([x,y,w,h])
            
            # Append to list of boxes
            bboxdata.append([label,coords])

        # Create XML file with bounding box data
        xml_fn = create_xml(image_path, bboxdata)
        print(xml_fn)

        # Move image to labeled folder
        img_fn = image_path.split('\\')[-1]
        new_image_path = os.path.join(cwd_path,FOLDER_NAME,LABELED_DIR,img_fn)
        os.rename(image_path,new_image_path)

    else:
        # If bounding box data is declined, move image to a separate folder
        img_fn = image_path.split('\\')[-1]
        new_image_path = os.path.join(cwd_path,FOLDER_NAME,UNLABELED_DIR,img_fn)
        os.rename(image_path,new_image_path)

# Clean up
cv2.destroyAllWindows()