#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
from pwm_joy_teleop.msg import DonkeyDrive

if __name__ == '__main__':
    rospy.init_node('pwm_joy_teleop')
    drive_publisher = rospy.Publisher("donkey_drive", DonkeyDrive)

    def joy_callback(joy_msg):
        donkey_drive = DonkeyDrive()
        donkey_drive.steering = -joy_msg.axes[0]
        donkey_drive.throttle = joy_msg.axes[3]
	donkey_drive.header.stamp = joy_msg.header.stamp

        drive_publisher.publish(donkey_drive)

    rospy.Subscriber('joy', Joy, joy_callback)
    rospy.spin()
