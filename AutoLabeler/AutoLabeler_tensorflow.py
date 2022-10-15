######## Automatic Image Labeling Using TensorFlow #########
#
# Author: Evan Juras
# Date: 9/25/19
# Description: 
# This program automatically labels object detection training images. It runs a pre-trained
# detection model to propose the bounding box and class of each object in each image. It
# then saves the label data in an XML file in Pascal VOC format. Users can accept or reject
# the proposed labels for each image. Reject images are moved to a separate folder for manual labeling.

# Recommended use: train an initial model using 200 images, then use that model with this program
# to assist with labeling remaining images.

# TO DO: Make it so this doesn't depend on TF Object Detection API utils

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'fine_tuned_model'
FOLDER_NAME = 'images'
UNLABELED_DIR = 'unlabeled'
LABELED_DIR = 'labeled'
CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labelmap.pbtxt')
PATH_TO_IMAGES = os.path.join(CWD_PATH,FOLDER_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

thresh = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX

# Function get image size
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
    xmlPath = os.path.join(CWD_PATH,FOLDER_NAME,LABELED_DIR,xmlFn)

    # Get image size
    imH, imW = get_size(imPath)

    # Create XML file and write data
    with open(xmlPath,'w') as f:
        f.write(xml_body_1.format(**{'FOLDER':FOLDER_NAME, 'FILENAME':imFn, 'PATH':imPath,
                                     'WIDTH':imW, 'HEIGHT':imH}))

        for bbox in imBBs:
            f.write(xml_object.format(**{'CLASS':bbox[0], 'XMIN':bbox[1][1], 'YMIN':bbox[1][0],
                                          'XMAX':bbox[1][3], 'YMAX':bbox[1][2]}))

        f.write(xml_body_2)

    return xmlFn


# Load the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Outputs
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Grab names of every image in the folder

images = glob.glob(PATH_TO_IMAGES+'/*.jpg')

stop = False

for image_path in images:
    
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    imH, imW, chan = image_rgb.shape

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.20)

    cv2.putText(image,'Label good? (y/n)',(30,50),font,1,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(image,'Label good? (y/n)',(30,50),font,1,(0,255,0),2,cv2.LINE_AA)
    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Label attempt', image)

    # Wait for user to accept or decline label
    saveLabels = False
    paused = True
    while paused: # can only exit this loop if 'y', 'n', or 'q' is pressed
        key = cv2.waitKey(0)
        if key == ord('y'):
            saveLabels = True
            paused = False
        if key == ord('n'):
            paused = False
        if key == ord('q'):
            stop = True
            paused = False

    if stop:
        break
    
    if saveLabels:
        bboxdata = []
        # Loop over every object, save class and bounding box coordinates
        for i in range(len(scores[0])):
            if scores[0][i] > thresh:
                ymin = int(boxes[0][i][0]*imH)
                xmin = int(boxes[0][i][1]*imW)
                ymax = int(boxes[0][i][2]*imH)
                xmax = int(boxes[0][i][3]*imW)
                coords = [ymin, xmin, ymax, xmax] # TO DO - swap these so it's xmin ymin xmax ymin
                label = 'fire'
                bboxdata.append([label, coords])

        # Create XML file with bounding box data
        xmlFn = create_xml(image_path, bboxdata)
        print(xmlFn)

        # Move image to labeled folder
        imFn = image_path.split('\\')[-1]
        new_image_path = os.path.join(CWD_PATH,FOLDER_NAME,LABELED_DIR,imFn)
        os.rename(image_path,new_image_path)

    else:
        # If bounding box data is declined, move image to a separate folder
        imFn = image_path.split('\\')[-1]
        new_image_path = os.path.join(CWD_PATH,FOLDER_NAME,UNLABELED_DIR,imFn)
        os.rename(image_path,new_image_path)

# Clean up
cv2.destroyAllWindows()
