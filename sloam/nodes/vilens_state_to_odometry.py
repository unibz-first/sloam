#!/usr/bin/env python
import rospy
from vilens_msgs.msg import State
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import TwistWithCovariance

def callback(msg):
    m = Odometry()
    p = PoseWithCovariance()
    t = TwistWithCovariance()
    p.pose = msg.pose
    t.twist = msg.twist
    m.pose = p
    m.twist = t
    m.header = msg.header
    m.child_frame_id = "base"
    odom_pub.publish(m)





if __name__ == '__main__':
    rospy.init_node('vilens_state_to_odometry')
    pose_sub = rospy.Subscriber('/vilens/state_propagated', State, callback)
    odom_pub = rospy.Publisher('/vilens/odom', Odometry, queue_size=10)
    
    rospy.spin()
