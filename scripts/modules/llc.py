import os
import sys
import concurrent.futures
from collections import deque
from enum import Enum, auto

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

class MessageTimer:
    class MsgState(Enum):
        ## No Inference results, ready for request
        IDLE = auto()
        ## Processing inference results
        INPROGRESS = auto()
        ## Inference has completed, ready for retrieval
        WAITING = auto()

    def __init__(self, request, response, executor, message_rate):
        self.request =request 
        self.response = response
        self.executor = executor
        self.message_duration = (1/message_rate)
        self.state = self.MsgState.IDLE
        self.last_msg_time = 0

    def poll_request(self, now):
        if self.state is self.MsgState.IDLE:
            self.last_msg_time = now
            self.network_future = self.executor.submit(
                self.request)
            self.state = self.MsgState.INPROGRESS
        elif self.state is self.MsgState.INPROGRESS:
            if self.network_future.done():
                if self.last_msg_time + self.message_duration < now:
                    ret_result = self.network_future.result()
                    print("WARNING: Message Timing Exceeded")
                    self.state = self.MsgState.IDLE
                else:
                    ret_result = self.network_future.result()
                    self.response(ret_result)
                    self.state = self.MsgState.WAITING
        elif self.state is self.MsgState.WAITING:
            if self.last_msg_time + self.message_duration < now:
                self.state = self.MsgState.IDLE


class LLC:
    def __init__(self, ip, executor, velocity_msg_rate, camera_msg_rate):
        self.bot = PiBot(ip=ip)
        self.velocity = [0, 0]
        self.image_queue = deque()
        self.odom_queue = deque()
        self.velocity_message = MessageTimer(
                self.velocity_publisher,
                self.velocity_response_callback,
                executor,
                velocity_msg_rate)
        self.camera_message = MessageTimer(
                self.camera_request_publisher,
                self.camera_msg_callback,
                executor,
                camera_msg_rate)

    def set_velocity(self, velocity):
        self.velocity = velocity

    def get_odom(self):
        odom = self.odom_message
        return ret_image

    def velocity_publisher(self):
        self.bot.setVelocity(*self.velocity)

    def velocity_response_callback(self, response):
        pass

    def camera_request_publisher(self):
        return self.bot.getImage()

    def camera_msg_callback(self, img):
        self.image_queue.append(img)

    def get_latest_image(self):
        ret_img = None 
        if len(self.image_queue) > 0:
            ret_img = self.image_queue.pop()
            self.image_queue.clear()
        return ret_img

    def poll_messages(self, now):
        self.velocity_message.poll_request(now)
        self.camera_message.poll_request(now)
        #self.odom_message.poll_request(now)

