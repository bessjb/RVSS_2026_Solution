from enum import Enum, auto

class RobotController:
    class ControlState(Enum):
        IDLE = auto()
        STOP = auto()
        GO = auto()

    def __init__(self, speed_limit, steer_angle_const, control_duration):
        self.speed_limit = speed_limit
        self.control_duration = control_duration
        self.steer_angle_const = steer_angle_const
        self.next_tick = 0

        self.start_delay_timer = 5 
        self.stop_duration = 0.5

        self.stop_time = 0
        self.last_start = 0

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
        self.request_stop = state

    def set_inference_results(self, results):
        self.inference_results = results

    def calculate_wheel_velocities(self):
        if self.inference_results == 0:
            angle = -0.5  # sharp left
        elif self.inference_results == 1:
            angle = -0.25  # slight left
        elif self.inference_results == 2:
            angle = 0  # straight
        elif self.inference_results == 3:
            angle = 0.25  # slight right
        elif self.inference_results == 4:
            angle = 0.5  # sharp right
        left  = int(self.speed_limit + self.steer_angle_const*angle)
        right = int(self.speed_limit - self.steer_angle_const*angle)
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
                velocity_setpoint = [0, 0]
                timeout = self.stop_time + self.stop_duration < current_time
                ## Stop ---> GO
                if timeout is True:
                    self.latch_state_message = False
                    self.state = self.ControlState.GO
                    self.last_start = current_time

            ## Robot is is Control State GO 
            elif self.state is self.ControlState.GO:
                self.print_state(self.state)
                velocity_setpoint = self.calculate_wheel_velocities()
                stop_delay = self.last_start + self.start_delay_timer < current_time
                ## GO ---> STOP
                if self.request_stop is True and stop_delay is True:
                    self.latch_state_message = False
                    self.stop_time = current_time
                    self.state = self.ControlState.STOP
            self.next_tick = current_time + self.control_duration

        return velocity_setpoint
