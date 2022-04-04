# Video Taker
Video Taker is a simple script for recording videos with OpenCV-Python. This script can be used in conjuction with [FrameGrabber](https://github.com/EdjeElectronics/Image-Dataset-Tools/tree/main/FrameGrabber) to accelerate the process of collecting images for training a machine learning model. Use Video Taker to record a scene containing objects you'd like to classify or detect, and then use FrameGrabber to automatically extract individual frames from the video. Video Taker is also great for recording videos to test your model on.

## Requirements
To use Video Taker, you need to install OpenCV-Python. If you're on the Raspberry Pi, see [Steps 1a - 1c in my TensorFlow Lite on the Raspberry Pi guide](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md) for instructions on how to install OpenCV-Python. (I'm planning to make a guide showing how to quickly and easily install OpenCV on Windows, Linux, and macOS, but just Google it for now!)

You'll also need to have a camera (such as a USB webcam) connected to your system.

## Usage
First, download the [VideoTaker.py](VideoTaker.py) file to a convenient location on your computer. To run the script, open a command terminal, navigate to the folder where you downloaded PictureTaker.py to, and issue:

```
python3 VideoTaker.py
```
A window will appear showing the camera's view and the script will begin recording. The script will continuously record video until the `q` key is pressed, which stops the program.
