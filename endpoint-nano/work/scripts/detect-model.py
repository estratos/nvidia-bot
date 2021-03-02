import jetson.inference
import jetson.utils
import cv2
import numpy as np
import time
width=1280
height=720
dispW=width
dispH=height


# On versions of L4T previous to L4T 28.1, flip-method=2
# Use the Jetson onboard camera
def open_mipi_camera():
    return cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

#  cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

#working cmd command 
#gst-launch-1.0 nvcamerasrc fpsRange="30.0 30.0" sensor-id=0 ! 'video/x-raw(memory:NVMM),width=(int)3872, height=(int)2144, format=(string)I420, framerate=(fraction)30/1' ! nvtee ! nvvidconv flip-method=2 ! 'video/x-raw, format=(string)I420' ! xvimagesink -e



#camSet= 'nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, fpsRange="30.0 30.0" sensor-id=0 ! 'video/x-raw(memory:NVMM),width=(int)3872, height=(int)2144, format=(string)I420, framerate=(fraction)30/1' ! nvtee ! nvvidconv flip-method=2 ! 'video/x-raw, format=(string)I420' ! xvimagesink -e'




#camSet= "nvcamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)800, height=(int)600, format=(string)I420, framerate=(fraction)30/1 ! nvtee ! nvvidconv flip-method=2 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
cam=open_mipi_camera()
#cam.release()
#cam=cv2.VideoCapture(camSet)
# we going to use mipi camera at Video0
#cam1=cv2.VideoCapture('/dev/video0')
#cam1.set(cv2.CAP_PROP_FRAME_WIDTH,dispW) 
#cam1.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
net=jetson.inference.imageNet('googlenet')
#net=jetson.inference.imageNet('resnet-18',['--model=/home/nvidia/work/Model-Train-Data/trainModel/resnet18.onnx','--input_blob=input_0','--output_blob=output_0','--labels=/home/nvidia/work/ModelTrainData/Train/labels.txt'])
font=cv2.FONT_HERSHEY_SIMPLEX
timeMark=time.time()
fpsFilter=0
while True:
    _,frame=cam.read()
    if frame is not None:
       img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)

    else:
        print("Error opening stream.. frame output is None ")
        cam.release()
        cv2.destroyAllWindows()
    	break 
    img=jetson.utils.cudaFromNumpy(img)

    classID, confidence =net.Classify(img, width, height)
    item=''
    
    item =net.GetClassDesc(classID)
    dt=time.time()-timeMark
    fps=1/dt
    fpsFilter=.95*fpsFilter +.05*fps
    timeMark=time.time()
    # print out the result
    print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(item, classID, confidence * 100))
    if (confidence*100) > 22:
    	cv2.putText(frame,str(round(fpsFilter,1))+' fps '+item,(0,30),font,1,(0,0,255),2)
    cv2.imshow('recCam',frame)
    cv2.moveWindow('recCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
