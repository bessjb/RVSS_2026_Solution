#!/usr/bin/env python3
import time
from enum import Enum, auto
from collections import deque
import os
import sys
import argparse
import concurrent.futures
from datetime import datetime
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
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
    def __init__(self, ip, executor, velocity_msg_rate, odom_msg_rate, camera_msg_rate):
        self.bot = PiBot(ip=ip)
        self.velocity = [0.0, 0.0]
        self.image_queue = deque()
        self.odom_queue = deque()
        self.velocity_message = MessageTimer(
                self.velocity_publisher,
                self.velocity_response_callback,
                executor,
                velocity_msg_rate)
        self.odom_message = MessageTimer(
                self.camera_request_publisher,
                self.camera_msg_callback,
                executor,
                odom_msg_rate)
        self.camera_message = MessageTimer(
                self.velocity_publisher,
                self.velocity_response_callback,
                executor,
                camera_msg_rate)

    def set_velocity(self, velocity):
        self.velocity = velocity

    def get_odom(self):
        odom = self.odom_message
        return ret_image

    def velocity_publisher(self):
        self.bot.setVelocity(self.velocity)

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


class RobotController:
    class ControlState(Enum):
        IDLE = auto()
        STOP = auto()
        GO = auto()

    def __init__(self, speed_limit, control_duration):
        self.speed_limit = speed_limit
        self.control_duration = control_duration
        self.next_tick = 0

        # stop the robot 
        self.velocity_setpoint = [0, 0]
        self.inference_results = None

        # Set the initial state to go
        self.state = self.ControlState.IDLE

        # Set transition condition defaults
        self.request_stop = False
        self.request_start = False
        self.latch_state_message = False

    def set_stop_sign_state(self, state):
        last_stopsign_timeout = False
        if state is True and not last_stopsign_timeout: self.request_stop = True

    def set_inference_results(self, results):
        self.inference_results = results

    def calculate_wheel_velocities(self):
        left  = int(self.speed_limit + self.inference_results)
        right = int(self.speed_limit - self.inference_results)
        return left, right

    def print_state(self, state):
        if self.latch_state_message is False:
            print(state)
            self.latch_state_message = True
    
    def run_control_step(self, current_time):
        velocity_setpoint = None
        if self.next_tick < current_time:

            ## Robot is is Control State IDLE
            if self.state is self.ControlState.IDLE:
                self.print_state(self.state)
                ## IDLE ---> GO
                if self.inference_results is not None:
                    self.latch_state_message = False
                    self.state = self.ControlState.GO

            ## Robot is is Control State STOP 
            elif self.state is self.ControlState.STOP:
                self.print_state(self.state)
                self.velocity_setpoint = [0, 0]
                ## Stop ---> GO
                if self.request_start is True:
                    self.state = self.ControlState.GO

            ## Robot is is Control State GO 
            elif self.state is self.ControlState.GO:
                self.print_state(self.state)
                velocity_setpoint = self.calculate_wheel_velocities()
                ## GO ---> STOP
                if self.request_stop is True:
                    self.state = self.ControlState.STOP
            self.next_tick = current_time + self.control_duration

        return velocity_setpoint

class NeuralNet:
    class NNState(Enum):
        ## No Inference results, ready for request
        READY = auto()
        ## Processing inference results
        INPROGRESS = auto()
        ## Inference has completed, ready for retrieval
        COMPLETE = auto()
    
    def __init__(self, executor):
        self.state = self.NNState.READY
        self.executor = executor

    @staticmethod
    def run_neural_network(image):
        time.sleep(0.1)
        return 5

    def run_inference(self, image):
        ret_result = None
        if self.state is self.NNState.READY:
            if image is not None:
                self.network_future = self.executor.submit(
                    self.run_neural_network, image)
                self.state = self.NNState.INPROGRESS
        elif self.state is self.NNState.INPROGRESS:
            if self.network_future.done():
                self.state = self.NNState.COMPLETE
        elif self.state is self.NNState.COMPLETE:
            ret_result = self.network_future.result()
            self.state = self.NNState.READY
        return ret_result

class RacerRobot:
    def __init__(self, args):
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.llc = LLC(args.ip,
                       self.executor,
                       args.velocity_msg_rate,
                       args.camera_msg_rate,
                       args.odom_msg_rate)
        self.neural_net = NeuralNet(self.executor)
        self.robot_controller = RobotController(
                speed_limit=args.speed_limit,
                control_duration= (args.control_duration/ 1000))
        self.tick_duration = (args.tick_duration / 1000)
        self.stop_signs_enabled = args.stop_signs_enabled

    def step_1_init(self):
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

    @staticmethod
    def check_for_stop_signs(image):
       return True 

    def main(self):
        next_tick = time.time()
        while True:
            current_time = time.time()

            self.llc.poll_messages(current_time)

            # get the latest image as fast as possible
            last_image = self.llc.get_latest_image()

            # Run the neural network on the latest image as frequently as possible
            inference = self.neural_net.run_inference(last_image)
            if inference is not None:
                self.robot_controller.set_inference_results(inference)

            # Run a single tick of the control algorithm
            velocity_commanded = self.robot_controller.run_control_step(current_time)
            if velocity_commanded is not None:
                self.llc.set_velocity(velocity_commanded)
                    
            #TO DO: check for stop signs?
            if self.stop_signs_enabled:
                stop_sign_state = self.check_for_stop_signs(last_image)
                self.robot_controller.set_stop_sign_state(stop_sign_state)

            ## Tick Rate logic
            next_tick += self.tick_duration
            sleep_duration = next_tick - current_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                print("WARNING: Missing deadlines")
                next_tick = current_time

def parse_args():
    parser = argparse.ArgumentParser(description='PiBot client')
    ## LLC Parameters
    parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
    parser.add_argument('--velocity_msg_rate', type=int, default=50, help='velocity command message rate in hz')
    parser.add_argument('--camera_msg_rate', type=int, default=10, help='Camera message rate in hz')
    parser.add_argument('--odom_msg_rate', type=int, default=50, help='Odometry rate in hz')
    ## Control Parameters
    parser.add_argument('--speed_limit', type=int, default=10, help='control duration in ms')
    parser.add_argument('--control_duration', type=int, default=100, help='control duration in ms')
    ## Neural Network parameters
    ## TODO
    ## High level logic Parameters 
    parser.add_argument('--tick_duration', type=int, default=5, help='tick duration in ms')
    parser.add_argument('--stop_signs_enabled', type=bool, default=False,
                        help='Whether stop sign detection is avaulable')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    racer_robot = RacerRobot(args)
    racer_robot.step_1_init()
    racer_robot.main()
