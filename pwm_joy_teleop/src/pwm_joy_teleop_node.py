#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
from donkey_actuator.msg import DonkeyDrive, Control

if __name__ == '__main__':
    rospy.init_node('pwm_joy_teleop')
    drive_publisher = rospy.Publisher("donkey_drive", DonkeyDrive)
    control_publisher = rospy.Publisher("control", Control)

    constant_throttle = 0.1
    throttle_increase = 0.1

    def joy_callback(joy_msg):
        global constant_throttle
        
        control = Control()
        control_changed = False

        # A -> control from joystick, Y -> from simple steering
        if joy_msg.buttons[2] or joy_msg.buttons[0]:
            control.set_actuator_source = 'joystick' if joy_msg.buttons[2] else 'simple_steering'
            control_changed = True

        if joy_msg.buttons[6] or joy_msg.buttons[7]:
            constant_throttle += (1 if joy_msg.buttons[7] else -1) * throttle_increase
            control.set_constant_throttle = constant_throttle
            control_changed = True

        if control_changed:
            control_publisher.publish(control)

        donkey_drive = DonkeyDrive()
        donkey_drive.source = 'joystick'
        donkey_drive.steering = -joy_msg.axes[0]
        donkey_drive.throttle = joy_msg.axes[3]
        donkey_drive.header.stamp = joy_msg.header.stamp
        drive_publisher.publish(donkey_drive)

    rospy.Subscriber('joy', Joy, joy_callback)
    rospy.spin()
