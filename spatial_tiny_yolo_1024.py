#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import pigpio
import math
import RPi.GPIO as GPIO #RPi.GPIO 라이브러리를 GPIO로 사용
from time import sleep  #time 라이브러리의 sleep함수 사용

class MovingAverage: ###################################### 추가 -> 이동평균 필터 적용
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.readings = []

    def add(self, value):
        self.readings.append(value)
        if len(self.readings) > self.window_size:
            self.readings.pop(0)
        return self.average()

    def average(self):
        return sum(self.readings) / len(self.readings)

############################
# motor setting
pi = pigpio.pi()

servo_list = [12, 13, 18, 19, 16, 23] ## 33을 12로, 35를 16으로, 37를 18로
for i in range(len(servo_list)):
    pi.set_servo_pulsewidth(servo_list[i], 90*11+600)

# LED setting
GPIO.setmode(GPIO.BCM)

#led_list=[8, 10, 22, 21, 23, 15] # 29를 15로 수정
led_list=[11, 10, 14, 9, 15, 8]
bottom_led = 26

GPIO.setup(bottom_led, GPIO.OUT)
#for i in len(led_list):
for i in range(len(led_list)):
    GPIO.setup(led_list[i], GPIO.OUT)


#prev_ang= [90*11 + 600, 90*11 + 600, 90*11 + 600, 90*11 + 600, 90*11 + 600, 90*11 + 600]
#ang = 90*11 + 600
prev_ang = [90, 90, 90, 90, 90, 90]
ang = [90, 90, 90, 90, 90, 90]
############################

'''
#################
pi.set_servo_pulsewidth(servo1, 0)
pi.set_servo_pulsewidth(servo2, 0)
pi.set_servo_pulsewidth(servo3, 0)
pi.set_servo_pulsewidth(servo4, 0)
pi.set_servo_pulsewidth(servo5, 0)
pi.set_servo_pulsewidth(servo6, 0)
sleep(2)
###################

#prev_ang = 90*11 + 600
pi.set_servo_pulsewidth(servo1, ang)
pi.set_servo_pulsewidth(servo2, ang)
pi.set_servo_pulsewidth(servo3, ang)
pi.set_servo_pulsewidth(servo4, ang)
pi.set_servo_pulsewidth(servo5, ang)
pi.set_servo_pulsewidth(servo6, ang)
sleep(2)
'''

'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
'''

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('/home/r/project/depthai/depthai-python/examples/SpatialDetection/1024.blob')).resolve().absolute())
if 1 < len(sys.argv):
    arg = sys.argv[1]
    if arg == "yolo3":
        nnBlobPath = str((Path(__file__).parent / Path('/home/r/project/depthai/depthai-python/examples/SpatialDetection/1024.blob')).resolve().absolute())
    elif arg == "yolo4":
        nnBlobPath = str((Path(__file__).parent / Path('/home/r/project/depthai/depthai-python/examples/SpatialDetection/1024.blob')).resolve().absolute())
    else:
        nnBlobPath = arg
else:
    print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Tiny yolo v3/4 label texts
labelMap = [
    "car",
    "tunnel"
]

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")

# Properties
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
stereo.setSubpixel(True)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(2)
spatialDetectionNetwork.setCoordinateSize(4)
#spatialDetectionNetwork.setAnchors([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326])
spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
spatialDetectionNetwork.setAnchorMasks({ "side26": [0,1,2], "side13": [3,4,5] })
#spatialDetectionNetwork.setAnchorMasks({ "side52": [0,1,2], "side26": [3,4,5], "side13": [6,7,8] })
## 위 두줄을 이렇게 수정함 일단
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    printOutputLayersOnce = True

    is_tunnel = False
    is_car = []
   
    tunnel_dept=0

#################           #################################3
    depth_filter = MovingAverage(window_size=5)
    confidence_filter = MovingAverage(window_size=5)
####################
    tunnel_label_index = labelMap.index("tunnel")
    while True:

##################################################
        flag = 0
#################################################

        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        inNN = networkQueue.get()

        # if printOutputLayersOnce:
        #     toPrint = 'Output layer names:'
        #     for ten in inNN.getAllLayerNames():
        #         toPrint = f'{toPrint} {ten},'
        #     print(toPrint)
        #     printOutputLayersOnce = False;

        frame = inPreview.getCvFrame()
       

        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        #min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        min_depth = None
        if np.any(depth_downscaled != 0):
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
            max_depth = np.percentile(depth_downscaled, 99)
            depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            epthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        else:
            print("No non-zero depth values found")
            max_depth = np.percentile(depth_downscaled, 99)
            depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            #depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
   

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections
        ####################################
        highest_confidence_tunnel = None
        car_detections = []
       
        for det in detections:
            if det.label == tunnel_label_index:
                if highest_confidence_tunnel is None or det.confidence > highest_confidence_tunnel.confidence:
                    highest_confidence_tunnel = det
            else:
                car_detections.append(det)

        if highest_confidence_tunnel:
            detections = [highest_confidence_tunnel] + car_detections
        else:
            detections = car_detections


        ########################################
        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]

        is_car.clear()
       
        #for detection in detections:
        #for detection in [darkest_tunnel] if tunnel_detections else detections:
        #for detection in [darkest_tunnel] if darkest_tunnel else detections:
        for detection in detections:
            ###################3
            smoothed_depth = depth_filter.add(detection.spatialCoordinates.z)
            #smoothed_confidence = confidence_filter.add(detection.confidence)
            ########################
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)

            if detection.label == tunnel_label_index:
                flag = 1
            #cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)



            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(smoothed_depth)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            if str(label) == "car":
                is_car.append([int((x1+x2)/2), int((y1+y2)/2)])

            if str(label) == "tunnel" :
                if detection.confidence > 0.5:
                    if int(smoothed_depth) <= 3000 and int(smoothed_depth) >50: # 시스템 가동 시작하는 지점
                        is_tunnel = 1
                        print("########################11111111111111111")
                    #ang = 90 + (int(((height/2)-y1)/20) * 7)

                        center_one = (x1 + int((x2 - x1) * 0.1), y1 + int((y2 - y1) * 0.4))
                        center_two = (x1 + int((x2 - x1) * 0.2), y1 + int((y2 - y1) * 0.2))
                        center_three = (x1 + int((x2 - x1) * 0.4), y1 + int((y2 - y1) * 0.1))
                        center_four = (x1 + int((x2 - x1) * 0.6), y1 + int((y2 - y1) * 0.1))
                        center_five = (x1 + int((x2 - x1) * 0.8), y1 + int((y2 - y1) * 0.2))
                        center_six = (x1 + int((x2 - x1) * 0.9), y1 + int((y2 - y1) * 0.4))
                       
                       
                        """
                        #중앙 상단부 중심으로 하는 헤드라이트 배치->간격 더 좁게
                        center_one = (x1 + int((x2 - x1) * 0.2), y1 + int((y2 - y1) * 0.2))
                        center_two = (int((x2 - x1) * 0.3), y1 + int((y2 - y1) * 0.1))
                        center_three = (x1 + int((x2 - x1) * 0.4), y2)
                        center_four = (x1 + int((x2 - x1) * 0.5), y2)
                        center_five = (int((x2 - x1) * 0.6), y1 + int((y2 - y1) * 0.1))
                        center_six = (x1 + int((x2 - x1) * 0.7), y1 + int((y2 - y1) * 0.2))
                        """
                       
                        #tunnel_dept=int(smoothed_depth)

                    # Drawing the dots
                        radius = 5
                        color = (0, 255, 0)  # Green color
                        thickness = -1  # Filled circle

                        cv2.circle(frame, center_one, radius, color, thickness)
                        cv2.circle(frame, center_two, radius, color, thickness)
                        cv2.circle(frame, center_three, radius, color, thickness)
                        cv2.circle(frame, center_four, radius, color, thickness)
                        cv2.circle(frame, center_five, radius, color, thickness)
                        cv2.circle(frame, center_six, radius, color, thickness) ## 각 6개 지점에 원 그리기

                        s=[center_one, center_two, center_three, center_four, center_five, center_six] ### 6개 원 리스트 저장

                    #터널 6개의 좌표들을 x좌표 기준으로 sort.
                        s.sort(key=lambda x:x[0])
                        s=np.array(s)
                        for i in range(6):
                        #정렬된 좌표 순으로 범위 내 일경우 y좌표로 각도 구해 각 @@모터 pin번호에 전달@@
                            if s[i][0]>int(width/6) and s[i][0]<int(width*5/6):
                                ang[i]=90+int(((height/2)-s[i][1])/20) * 10
                                if ang[i]>120:
                                    ang[i]=120
                                elif ang[i]<90:
                                    ang[i] =90
                            else:
                                ang[i]=prev_ang[i]
                           
                    elif int(smoothed_depth)>=3000: ## 터널이 인식은 됐는데 거리가 너무 멀어서, 즉 터널에 인접하고는 있는데 아직 거리가 멀기 때문에 시스템 가동시키지 않고 모든 모터의 각도는 항상 90도로 유지
                        is_tunnel=2
                        ang = [90, 90, 90, 90, 90, 90]
                   
                        tunnel_dept=int(smoothed_depth) # 터널이 감지가 되면 일단 무조건 현재 터널의 depth 저장
               
                    else:     #터널과의 거리가 매우 짧을 경우 원위치
                        is_tunnel=0
                        ang=[90,90,90,90,90,90]
         
                        GPIO.output(bottom_led, False)
           
                        for i in range(len(led_list)):
                            GPIO.output(led_list[i], False)
               
                        #for i in range(len(servo_list)):
                            #if(prev_ang[i] != ang[i]):
                                #pi.set_servo_pulsewidth(servo_list[i], ang[i]*11+600)
                                #pi.set_servo_pulsewidth(servo_list[i], (ang[i])*11+600)
                        #prev_ang=ang
                   
                    #1.모터 조정 거리
                    #2.터널 인식/모터 조정x
                    #3.터널 인식x-터널이 실제로 없는 경우,-터널이 있는데 인식을 못한 경우->터널 거리 저장, 거리가 Nm 이하인 경우만 터널 실제 없는 경
        if flag == 0:
            is_tunnel=0

        if is_tunnel:
            GPIO.output(bottom_led, True)
            #for i in len(led_list):
            for i in range(len(led_list)):
                GPIO.output(led_list[i], True) ## 각도는 이미 다 세팅되어 있고, 따라서 불을 일단 전부 킨다 -> 이후 아래 줄에 나오듯 선행 차량 감지되면 그 부분 불을 꺼준는 방식
            # for i in range(6):
            #     if(prev_ang[i] != ang[i]):
            #         #pi.set_servo_pulsewidth(servo_list[i], ang[i]*11+600)
            #         print(" ")

                ##prev_ang[i] = ang[i] # 터널이 인지됐을 때는 현재 각도를 모두 prev_ang에 저장
                ##print("case1", " num: ", is_tunnel) ## is_tunnel == 1(Tunnel Vision이 가동되는 시기), is_tunnel == 2(터널 인지는 되는데 거리가 멀어서 시스템 가동 X)모두 포함하는 케이스
                tunnel_dept=0
            #print(prev_ang)

            #if is_car length > 0:
        for i in range(len(is_car)):
                #print("led num ", math.floor(is_car[i][0]/int(width/6)), " off")
            GPIO.output(led_list[(math.floor(is_car[i][0]/int(width/6))%6)], False)
       
                #tunnel_dept=0
            tunnel_dept=0
               
        for i in range(len(servo_list)):
            if(prev_ang[i] != ang[i]):
                if(i == 2):
                    pi.set_servo_pulsewidth(servo_list[i], (180-ang[i])*11+600)
                else:
                    pi.set_servo_pulsewidth(servo_list[i], ang[i]*11+600)
                prev_ang[i] = ang[i]

        #print(ang)
       
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        #cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            GPIO.OUTPUT(bottom_led, False)
            for i in range(len(led_list)):
                GPIO.OUTPUT(led_list[i], False)
            GPIO.cleanup()
            break
