#pragma once
#include <ros/node_handle.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <deque>
#include <queue>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>


class InputManager
{
#include "../helpers/definitions.h"

  struct StampedSE3 {
    StampedSE3(SE3 p, ros::Time s) : pose(p), stamp(s) {}
    StampedSE3() : pose(SE3()), stamp(ros::Time::now()) {}
    SE3 pose;
    ros::Time stamp;
  };

  enum
  {
      CLOUD_TOO_OLD,
      CLOUD_TOO_NEW,
      CLOUD_FOUND
  };

public:
    explicit InputManager(ros::NodeHandle& nh);
    bool Run();


private:
    void OdomCb_(const nav_msgs::OdometryConstPtr &odom_msg);

    template<typename PT>
    int FindPC(const ros::Time stamp, boost::shared_ptr<pcl::PointCloud<PT>> cloud)
    {
        if (pcQueue_.empty())
            return false;
        while (!pcQueue_.empty())
        {
            if (pcQueue_.front()->header.stamp.toSec() < stamp.toSec() - 0.05)
            {
                // message too old
                ROS_DEBUG_THROTTLE(1, "PC MSG TOO OLD");
                pcQueue_.pop();
            }
            else if (pcQueue_.front()->header.stamp.toSec() > stamp.toSec() + 0.05)
            {
                // message too new
                ROS_DEBUG_THROTTLE(1, "PC MSG TOO NEW");
                return CLOUD_TOO_NEW;
            }
            else
            {
                pcl::fromROSMsg(*pcQueue_.front(), *cloud);
                pcQueue_.pop();
                ROS_DEBUG("Calling SLOAM");
                return CLOUD_FOUND;
            }
        }
        return CLOUD_TOO_OLD;
    }
    void PCCb_(const sensor_msgs::PointCloud2ConstPtr &cloudMsg);
    bool callSLOAM(SE3 relativeMotion, ros::Time stamp);
    void PublishAccumOdom_(const SE3 &relativeMotion);
    void Odom2SlamTf();
    void PublishOdomAsTf(const nav_msgs::Odometry &odom_msg,
                         const std::string &parent_frame_id,
                         const std::string &child_frame_id);

    std::queue<sensor_msgs::PointCloud2ConstPtr> pcQueue_;
    std::deque<StampedSE3> odomQueue_;
    ros::NodeHandle& nh_;
    ros::Publisher pubPose_;
    ros::Subscriber OdomSub_;
    ros::Subscriber PCSub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2_ros::TransformBroadcaster broadcaster_;

    // params
    std::string map_frame_id_;
    std::string odom_frame_id_;
    std::string robot_frame_id_;
    std::string cloud_topic_;
    std::string odom_topic_;
    float minOdomDistance_;
    float minSLOAMAltitude_;
    size_t maxQueueSize_;

    // vars
    boost::shared_ptr<sloam::SLOAMNode> sloam_ = nullptr;
    std::vector<SE3> keyPoses_;
    StampedSE3 latestOdom;
    bool firstOdom_;
    bool publishTf_;
    size_t odomCounter_;
    size_t odomFreqFilter_;
    bool use_hesai_;
};
