#!/usr/bin/env python
import rospy
from vilens_msgs.msg import State
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import TwistWithCovariance
from geometry_msgs.msg import TransformStamped
import tf2_ros

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
    br = tf2_ros.TransformBroadcaster()
    tf_msg = TransformStamped()
    tf_msg.transform.translation.x = msg.pose.position.x
    tf_msg.transform.translation.y = msg.pose.position.y
    tf_msg.transform.translation.z = msg.pose.position.z
    tf_msg.transform.rotation.x = msg.pose.orientation.x
    tf_msg.transform.rotation.y = msg.pose.orientation.y
    tf_msg.transform.rotation.z = msg.pose.orientation.z
    tf_msg.transform.rotation.w = msg.pose.orientation.w

    tf_msg.header = msg.header
    tf_msg.child_frame_id = m.child_frame_id
    br.sendTransform(tf_msg)


if __name__ == '__main__':
    rospy.init_node('vilens_state_to_odometry')
    pose_sub = rospy.Subscriber('/vilens/state_propagated', State, callback)
    odom_pub = rospy.Publisher('/vilens/odom', Odometry, queue_size=10)
    
    rospy.spin()
