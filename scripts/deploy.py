#!/usr/bin/env python3
import time
import argparse
import concurrent.futures
from datetime import datetime
from modules import LLC, NeuralNetWrapper, RobotController


class RacerRobot:
    def __init__(self, args):
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.llc = LLC(args.ip,
                       self.executor,
                       args.velocity_msg_rate,
                       args.camera_msg_rate,
                       args.odom_msg_rate)
        self.neural_net = NeuralNetWrapper.load_model(
            self.executor, args.model_path)
        self.robot_controller = RobotController(
            speed_limit=args.speed_limit,
            control_duration=(args.control_duration / 1000))
        self.tick_duration = (args.tick_duration / 1000)
        self.stop_signs_enabled = args.stop_signs_enabled

    def step_1_init(self):
        # countdown before beginning
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
            velocity_commanded = self.robot_controller.run_control_step(
                current_time)
            if velocity_commanded is not None:
                self.llc.set_velocity(velocity_commanded)

            # TO DO: check for stop signs?
            if self.stop_signs_enabled:
                stop_sign_state = self.check_for_stop_signs(last_image)
                self.robot_controller.set_stop_sign_state(stop_sign_state)

            # Tick Rate logic
            next_tick += self.tick_duration
            sleep_duration = next_tick - current_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                print("WARNING: Missing deadlines")
                next_tick = current_time


def parse_args():
    parser = argparse.ArgumentParser(description='PiBot client')
    parser.add_argument('--model_path', type=str)
    # LLC Parameters
    parser.add_argument('--ip', type=str, default='localhost',
                        help='IP address of PiBot')
    parser.add_argument('--velocity_msg_rate', type=int,
                        default=50, help='velocity command message rate in hz')
    parser.add_argument('--camera_msg_rate', type=int,
                        default=10, help='Camera message rate in hz')
    parser.add_argument('--odom_msg_rate', type=int,
                        default=50, help='Odometry rate in hz')
    # Control Parameters
    parser.add_argument('--speed_limit', type=int,
                        default=10, help='control duration in ms')
    parser.add_argument('--control_duration', type=int,
                        default=100, help='control duration in ms')
    # Neural Network parameters
    # TODO
    # High level logic Parameters
    parser.add_argument('--tick_duration', type=int,
                        default=5, help='tick duration in ms')
    parser.add_argument('--stop_signs_enabled', type=bool, default=False,
                        help='Whether stop sign detection is avaulable')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args.model_path)
    racer_robot = RacerRobot(args)
    racer_robot.step_1_init()
    racer_robot.main()
