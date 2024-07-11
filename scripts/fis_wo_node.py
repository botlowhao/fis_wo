#!/usr/bin/env python3

import rospy
from vwio_eskf.msg import WOFISData

def wofis_data_callback(msg):
    rospy.loginfo("Received WOFISData message: delta_v={}, w_z={}".format(msg.delta_v, msg.w_z))

def wofis_subscriber():
    rospy.init_node('wofis_subscriber', anonymous=True)
    rospy.Subscriber('wofis_data', WOFISData, wofis_data_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        wofis_subscriber()
    except rospy.ROSInterruptException:
        pass