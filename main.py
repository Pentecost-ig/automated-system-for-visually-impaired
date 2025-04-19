import RPi.GPIO as gpio
from gtts import gTTS
from picamera2 import Picamera2
import os
import cv2
import time
import threading
import numpy as np
import serial
import smbus
from flask import *

app=Flask(__name__)


ser1 = serial.Serial('/dev/ttyS0', baudrate=9600, timeout=1)

picam = Picamera2()
picam.preview_configuration.main.size = (480, 480)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.main.align()
picam.configure("preview")
picam.start()



help_btn = 21
face_btn = 20
GPIO_TRIGGER = 17                   
GPIO_ECHO = 27

fall_detected_time = None


flag1=0
flag2=0
flag3=0
flag4=0

# MPU-6050 I2C address
MPU_ADDR = 0x68

# MPU-6050 Registers
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
ACCEL_XOUT_L = 0x3C
ACCEL_YOUT_H = 0x3D
ACCEL_YOUT_L = 0x3E
ACCEL_ZOUT_H = 0x3F
ACCEL_ZOUT_L = 0x40
GYRO_XOUT_H = 0x43
GYRO_XOUT_L = 0x44
GYRO_YOUT_H = 0x45
GYRO_YOUT_L = 0x46
GYRO_ZOUT_H = 0x47
GYRO_ZOUT_L = 0x48

# Initialize I2C bus
bus = smbus.SMBus(1)

# Wake up the MPU-6050 (it starts in sleep mode)
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
time.sleep(0.1)

classesFile = '/home/pi/VISUALLY_IMPAIRED_GADGET/coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelconfig = '/home/pi/VISUALLY_IMPAIRED_GADGET/yolov3tiny.cfg'
modelweight = '/home/pi/VISUALLY_IMPAIRED_GADGET/yolov3tiny.weights'
net = cv2.dnn.readNetFromDarknet(modelconfig, modelweight)


def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)
    val = (high << 8) + low
    return val

def read_word_2c(reg):
    val = read_word(reg)
    if val > 0x8000:
        return val - 65536
    else:
        return val

def read_accel_data():
    ax = read_word_2c(ACCEL_XOUT_H)
    ay = read_word_2c(ACCEL_YOUT_H)
    az = read_word_2c(ACCEL_ZOUT_H)
    return ax, ay, az

def read_gyro_data():
    gx = read_word_2c(GYRO_XOUT_H)
    gy = read_word_2c(GYRO_YOUT_H)
    gz = read_word_2c(GYRO_ZOUT_H)
    return gx, gy, gz
    
                        
def main():
    global s,fall_detected_time
    global flag1,flag2,flag3,flag4
    gpio.setwarnings(False)
    gpio.setmode(gpio.BCM)
    gpio.setup(face_btn, gpio.IN)
    gpio.setup(help_btn, gpio.IN)
    gpio.setup(GPIO_TRIGGER, gpio.OUT)
    gpio.setup(GPIO_ECHO, gpio.IN)
    threading.Thread(target=distance).start()
  
    speak("Welcome to the Smart Assistive System for the Blind")
    print("Welcome to the Smart Assistive System for the Blind")
    while True:
        global s
        help_val = gpio.input(help_btn)
        face_val = gpio.input(face_btn)
        ax, ay, az = read_accel_data()
        print(f"Accel (X): {ax}")
        
        if ax >= 5000 and fall_detected_time is None:  # Detect fall and mark the time
            print("Fall detected")
            fall_detected_time = time.time()

        if fall_detected_time is not None:  # If a fall was detected
            if gpio.input(help_btn) == gpio.LOW:  # Check if the button is pressed
                print("Help button pressed. Fall message cancelled.")
                speak("Help button pressed. Fall message cancelled.")
                fall_detected_time = None  # Reset fall detection
            elif time.time() - fall_detected_time > 10:  # Check if 10 seconds have passed
                print("FALL DETECTED EMERGENCY")
                speak("FALL DETECTED EMERGENCY")
                fall_detected_time = None  # Reset after sending emergency message

        time.sleep(0.2)
        if face_val == 0:
            flag4 = 1
        if face_val == 1 and flag4 == 1:
            flag4 = 0
            print("System is going to check face")
            speak("System is going to check face")
            face_check()


def detect():
    cap = cv2.VideoCapture(0)
    while True:
        # Display camera preview for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            img = picam.capture_array()
            cv2.imshow('Live Detection', img)
            cv2.waitKey(1)
            
# Save the captured frame as an image
        cv2.imwrite("captured_image.jpg", img)
        # Close all windows
        cv2.destroyAllWindows()
        img = cv2.imread("captured_image.jpg")
        blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        a = net.getUnconnectedOutLayers()
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        h, w, c = img.shape
        bbox = []
        classids = []
        conf = []
        for output in outputs:
            for det in output:
                score = det[5:]
                classid = np.argmax(score)
                confidence = score[classid]
                if confidence > 0.4:
                    wd, ht = int(det[2] * w), int(det[3] * h)
                    x, y = int((det[0] * w) - wd / 2), int((det[1] * h) - ht / 2)
                    bbox.append([x, y, wd, ht])
                    classids.append(classid)
                    conf.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox, conf, 0.5, 0.3)
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            res=classNames[classids[i]]
            print(res)
            speak(res)
            obj_flag = 1

def distance():
    while True:
        gpio.output(GPIO_TRIGGER, False)
        time.sleep(0.0001)
        gpio.output(GPIO_TRIGGER, True)
        time.sleep(0.0001)
        gpio.output(GPIO_TRIGGER, False)

        start_time = time.time()
        stop_time = time.time()

        while gpio.input(GPIO_ECHO) == 0:
            start_time = time.time()

        while gpio.input(GPIO_ECHO) == 1:
            stop_time = time.time()

        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2
        print(distance)
        if (distance<=20):
            print("object detected")
            detect()
            

    
def speak(text):
    myobj = gTTS(text=text, lang='en', slow=False)
    myobj.save("/home/pi/VISUALLY_IMPAIRED_GADGET/speech.mp3")
    os.system("mpg321 /home/pi/VISUALLY_IMPAIRED_GADGET/speech.mp3")        

if name == "__main__":
    main()