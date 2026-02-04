from enum import Enum, auto

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
        left  = int(self.speed_limit + angle)
        right = int(self.speed_limit - angle)
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
