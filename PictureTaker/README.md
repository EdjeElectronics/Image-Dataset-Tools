# Picture Taker
Picture Taker is a simple Python script for taking pictures with OpenCV and a connected camera. The script makes it easy to collect images for training a machine learning vision model.

## Requirements
To use Picture Taker, you need to install OpenCV-Python. If you're on the Raspberry Pi, see [Steps 1a - 1c in my TensorFlow Lite on the Raspberry Pi guide](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md) for instructions on how to install OpenCV-Python. (I'm planning to make a guide showing how to quickly and easily install OpenCV on Windows, Linux, and macOS, but just Google it for now!)

You'll also need to have a camera (such as a USB webcam) connected to your system.

## Usage
First, download the [PictureTaker.py file](https://raw.githubusercontent.com/EdjeElectronics/Image-Dataset-Tools/main/PictureTaker/PictureTaker.py) to a convenient location on your computer. To run the script, open a command terminal, navigate to the folder where you downloaded PictureTaker.py to, and issue: 

```python3 PictureTaker.py```

A window will open showing the camera's view. Press `p` on the keyboard to take a picture. The first picture will be saved in a folder called "Pics" as filename `Pics_1.jpg`. If the "Pics" folder doesn't exist, it will be created automatically. Continue pressing `p` to take more pictures. Each additional picture will be saved as `Pics_2.jpg`, `Pics_3.jpg`, and so on. Press `q` to quit.

If `Pics_1.jpg` already exists in the Pics folder, the program will automatically increase the picture number until it reaches a filename that doesn't exist. (For example, if `Pics_1.jpg` through `Pics_20.jpg` already exist in the "Pics" folder, the program will start with `Pics_21.jpg` for the first saved picture.) This way, you can take pictures without having to worry about overwriting existing pictures.

### Command Arguments
By default, the script runs at a resolution of 1280x720, saves pictures in a folder called "Pics", and uses "Pics" as the base filename for the pictures. The defaults can be changed using the following command arguments:
* `--res` : specify the camera resolution in WxH to change the resolution
* `--imgdir` : specify the folder to save pictures in (and to use as the base filename for the pictures)

For example, to take pictures at a 1920x1080 resolution and save them in a folder called "Birds", issue the following command:

```python3 PictureTaker.py --res=1920x1080 --imgdir=Birds```

Have fun snapping pictures!
