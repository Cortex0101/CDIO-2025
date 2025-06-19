from .StateBase import StateBase

import math

class StateRotateToObject(StateBase):
    def __init__(self, server, target_object=None):
        super().__init__(server)
        self.target_object = target_object

    def on_enter(self):
        self.robot_direction = self.server.course.get_robot().direction
        self.robot_center = self.server.course.get_robot().center
        self.target_direction = self._angle_to(self.robot_center, self.target_object.center)
        self.angle_has_been_correct_for_x_frame = 0

    def update(self, frame):
        self.robot_direction = self.server.course.get_robot().direction if self.server.course.get_robot() is not None else self.robot_direction
        self.robot_center = self.server.course.get_robot().center if self.server.course.get_robot() is not None else self.robot_center
        angle_to_target = self._angle_to(self.robot_center, self.target_object.center)
        
        # Compute turn command to face the target object
        instruction = self.server.pure_pursuit_navigator.compute_turn_command(
            self.robot_direction, angle_to_target, newKp=0.9, new_max_speed=10
        )
        self.server.send_instruction(instruction)

        # Check if the robot is facing the target object
        print(f"[SERVER] Robot direction: {self.robot_direction}, angle to target: {angle_to_target}")
        print(f"[SERVER] Angle difference: {abs(angle_to_target - self.robot_direction)}")
        if abs(angle_to_target - self.robot_direction) < 3:
            print("[SERVER] Robot is now facing the target object.")
            # Wait for a few frames to ensure it's stable
            self.angle_has_been_correct_for_x_frame += 1
            if self.angle_has_been_correct_for_x_frame > 10:
                print("[SERVER] Angle has been stable for 10 frames, switching to next state.")
                from .StateCollectBall import StateCollectBall
                self.server.set_state(StateCollectBall(self.server, self.target_object))

    def _angle_to(self, src, dst):
        # clamped to 0-360 where 0 is right, 90 is up, 180 is left, 270 is down
        angle = math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))
        if angle < 0:
            angle += 360
        return angle  # returns angle in degrees, 0 is right, 90 is up, 180 is left, 270 is down

    def on_exit(self):
        self.angle_has_been_correct_for_x_frame = 0

    def on_click(self, event, x, y):
        """
        Handle click events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        pass

    def on_key_press(self, key):
        """
        Handle key press events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        if key == ord('x'):
            # If 'g' is pressed, go back to idle state
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

'''
elif current_state == RobotState.TURN_TO_OBJECT_OR_POINT and angle_to_target != -1:
                # Compute turn command to face the target object
                instruction = self.pure_pursuit_navigator.compute_turn_command(robot_direction, angle_to_target, newKp=0.9, new_max_speed=10)
                self.send_instruction(instruction)

                # Check if the robot is facing the target object
                print(f"[SERVER] Robot direction: {robot_direction}, angle to target: {angle_to_target}")
                print(f"[SERVER] Angle difference: {abs(angle_to_target - robot_direction)}")
                if abs(angle_to_target - robot_direction) < 3:
                    print("[SERVER] Robot is now facing the target object.")
                    angle_has_been_correct_for_x_frame += 1
                    if angle_has_been_correct_for_x_frame > 5:  # Wait for 5 frames to ensure it's stable
                        print("[SERVER] Angle has been stable for 5 frames, switching to next state.")
                        angle_to_target = -1  # Reset angle to target
                        angle_has_been_correct_for_x_frame = 0
                        self.send_instruction({"cmd": "drive", "left_speed": 0, "right_speed": 0})
'''