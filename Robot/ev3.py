#!/usr/bin/env python3
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, SpeedPercent, MoveTank
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import TouchSensor
from ev3dev2.led import Leds

# TODO: Implement functions for the Robot class

class Robot:
    def forward(self) -> bool:
        raise NotImplementedError
    
    def backward(self) -> bool:
        raise NotImplementedError
    
    def turn_left(self, radians) -> bool:
        raise NotImplementedError
    
    def turn_right(self, radians) -> bool:
        raise NotImplementedError
    
    #def turn_relative(self, radians) -> bool:
    #    raise NotImplementedError

    def turn_to_direction(self, radians) -> bool:
        raise NotImplementedError
    
    def set_speed(self, left_speed, right_speed) -> bool:
        raise NotImplementedError
    
    def drive(self, position: tuple) -> bool:
        raise NotImplementedError
    
    def stop(self) -> bool:
        raise NotImplementedError
    
    def set_position(self, position: tuple) -> bool:
        raise NotImplementedError
    
    def set_direction(self, radians) -> bool:
        raise NotImplementedError
    
    def buttons_pressed(self) -> bool:
        raise NotImplementedError