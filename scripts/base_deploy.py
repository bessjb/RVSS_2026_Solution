#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

from stop_sign import detect_stop_sign, detect_stop_sign_hsv


parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--model', type=str)
args = parser.parse_args()

bot = PiBot(ip=args.ip)
sign_detected = False
allow_detection = True
detection_chrono = 0
# stop the robot 
bot.setVelocity(0, 0)

#INITIALISE NETWORK HERE

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        # With the current preprocessing (Resize to 40x60 then crop to 20x60),
        # the feature map after conv2 has shape (16, 4, 24). Pooling again would
        # collapse the height to 0, so we omit the second pool and flatten 16*4*24.
        self.fc1 = nn.Linear(3888, 256)
        self.fc2 = nn.Linear(256, 5)

        self.relu = nn.ReLU()


    def forward(self, x):
        #extract features with convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        #linear layer for classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
       
        return x
    

net = Net()

#LOAD NETWORK WEIGHTS HERE
# Load path from relative to args (no folder hardcoded)
net.load_state_dict(torch.load(args.model))

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    while True:
        # get an image from the the robot
        im = bot.getImage()
        im_np = np.asarray(im)

        #TO DO: apply any necessary image transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((240, 120)),
                                transforms.Lambda(lambda img: transforms.functional.crop(img, 190, 0, 50, 120)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

        #TO DO: pass image through network get a prediction
        #image = Image.open("data/track3/0000020.00.jpg")
        input = transform(im)
        input = input.unsqueeze(0)  # Add batch dimension
        output = net(input)
        _, predicted = torch.max(output, 1)
        print(f'Predicted class: {predicted.item()}')

        #TO DO: convert prediction into a meaningful steering angle
        if predicted.item() == 0:
            angle = -0.5  # sharp left
        elif predicted.item() == 1:
            angle = -0.25  # slight left
        elif predicted.item() == 2:
            angle = 0.0  # straight
        elif predicted.item() == 3:
            angle = 0.25  # slight right
        elif predicted.item() == 4:
            angle = 0.5  # sharp right
        #TO DO: check for stop signs?

        # Detection allowance
        if allow_detection:
            sign_detected = detect_stop_sign_hsv(im_np)
            if sign_detected:
                print("Stop sign detected! Stopping robot.")
                bot.setVelocity(0, 0)
                allow_detection = False  # Prevent further detections for a short period
                time.sleep(0.7)  # Stop for 5 seconds
                detection_chrono = time.time()  # Start cooldown timer

        if time.time() - detection_chrono > 3:  # Cooldown period of 5 seconds
            allow_detection = True




        #angle = 0

        Kd = 35 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 35 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)

        # Catch keyboard inte
        
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
    print("Stopping robot")