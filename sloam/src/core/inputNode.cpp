#include <queue>
#include <deque>
#include <ros/ros.h>
#include <ros/console.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Header.h>
#include <pcl_ros/point_cloud.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <definitions.h>
#include <sloamNode.h>
#include "../inputNode.hpp"
#include "../helpers/hesai_point_types.h"



InputManager::InputManager(ros::NodeHandle &nh) : nh_(nh), tf_listener_{tf_buffer_}
{
    nh_.param<float>("min_odom_distance", minOdomDistance_, 0.5);
    nh_.param<float>("min_sloam_z", minSLOAMAltitude_, 0.5);
    maxQueueSize_ = nh_.param("maxQueueSize", 30);
    odomFreqFilter_ = nh_.param("odom_freq_filter", 20);
    publishTf_ = nh_.param("publish_tf", true);
    nh_.param<std::string>("robot_frame_id", robot_frame_id_, "robot");
    nh_.param<std::string>("odom_frame_id", odom_frame_id_, "odom");
    nh_.param<std::string>("map_frame_id", map_frame_id_, "map");
    nh_.param<std::string>("cloud_topic", cloud_topic_, "cloud");
    nh_.param<std::string>("odom_topic", odom_topic_, "/quadrotor/vio/odom");
    nh_.param<bool>("use_hesai", use_hesai_, false);

    pubPose_ =
        nh_.advertise<geometry_msgs::PoseStamped>("debug/ref_frame_sloam_pose", 10);
    OdomSub_ = nh_.subscribe(odom_topic_, 10, &InputManager::OdomCb_, this);
    PCSub_ = nh_.subscribe(cloud_topic_, 10, &InputManager::PCCb_, this);

    auto sloam_ptr = boost::make_shared<sloam::SLOAMNode>(nh_);
    sloam_ = std::move(sloam_ptr);
    firstOdom_ = true;
    odomCounter_ = 0;

    ROS_INFO("SLOAM initialized");
}

void InputManager::OdomCb_(const nav_msgs::OdometryConstPtr &odom_msg)
{
    odomCounter_++;
    if (odomCounter_ % odomFreqFilter_ != 0)
        return;
    odomCounter_ = 0;

    auto pose = odom_msg->pose.pose;
    ros::Time odomStamp = odom_msg->header.stamp;
    Quat rot(pose.orientation.w, pose.orientation.x,
             pose.orientation.y, pose.orientation.z);
    Vector3 pos(pose.position.x, pose.position.y, pose.position.z);

    SE3 odom = SE3();
    odom.setQuaternion(rot);
    odom.translation() = pos;

    // odom is in cam frame, we need it in robot frame
    geometry_msgs::TransformStamped transform_cam_body;
    try
    {
        auto transform_cam_body = tf_buffer_.lookupTransform(
            odom_msg->child_frame_id, robot_frame_id_, ros::Time(0));
        SE3 cam_body_tf(tf2::transformToEigen(transform_cam_body).matrix().cast<double>());
        // I think the following is wrong, it should be:
        odom = odom * cam_body_tf.inverse();
        //odom = odom * cam_body_tf;

    }
    catch (tf2::TransformException &ex)
    {
        ROS_INFO_THROTTLE(10, "Camera to body transform is not found, now setting it as identity... If you're in simulator, ignore this.");
    }

    if (firstOdom_ && pose.position.z < minSLOAMAltitude_)
    {
        ROS_INFO_THROTTLE(5, "Quad is too low, will not call sloam");
        return;
    }

    odomQueue_.emplace_back(odom, odomStamp);
    if (odomQueue_.size() > 10 * maxQueueSize_)
        odomQueue_.pop_front();
}

bool InputManager::Run()
{

    if (odomQueue_.empty() || pcQueue_.empty())
        return false;

    // ROS_DEBUG_THROTTLE(5, "First odom stamp: %f", odomQueue_.front().stamp.toSec());
    // ROS_DEBUG_THROTTLE(5, "First cloud stamp: %f", pcQueue_.front()->header.stamp.toSec());
    // ROS_DEBUG_THROTTLE(5, "Last odom stamp: %f", odomQueue_.back().stamp.toSec());
    // ROS_DEBUG_THROTTLE(5, "Last cloud stamp: %f", pcQueue_.back()->header.stamp.toSec());

    for (auto i = 0; i < odomQueue_.size(); ++i)
    {
        auto odom = odomQueue_[i];
        // Use odom to estimate motion since last key frame
        SE3 currRelativeMotion = latestOdom.pose.inverse() * odom.pose;

        if (firstOdom_)
        {
            ROS_INFO_THROTTLE(1.0, "first sloam call");
            if (callSLOAM(currRelativeMotion, odom.stamp))
            {
                firstOdom_ = false;
                latestOdom.pose = odom.pose;
                latestOdom.stamp = odom.stamp;
                if(publishTf_)
                    Odom2SlamTf();
                return true;
            }
        }
        else
        {
            double accumMovement = currRelativeMotion.translation().norm();
            if (accumMovement > minOdomDistance_)
            {
                ROS_DEBUG_THROTTLE(1.0, "Distance %f", (accumMovement));
                if (callSLOAM(currRelativeMotion, odom.stamp))
                {
                    latestOdom.pose = odom.pose;
                    latestOdom.stamp = odom.stamp;
                    if(publishTf_)
                        Odom2SlamTf();
                    return true;
                }
            }
        }
    }
    return false;
}

int InputManager::FindHesaiCloud(const ros::Time stamp,
                                  CloudT::Ptr &cloud_out)
{
  HesaiPointCloud::Ptr hesai_cloud = pcl::make_shared<HesaiPointCloud>();
  // TODO: change templating to CloudT instead of PointT?
    // cloud is not dense because it might contain invalid points
  if (cloud_out.get() == nullptr) {
      std::cerr << "no cloud_out CloudT::Ptr +++++++++++++++++++00\n";
      cloud_out = pcl::make_shared<CloudT>();
  }
  cloud_out->is_dense = false;
  cloud_out->header = hesai_cloud->header;

  std::cerr << "++++++++++++++++++++++++++++++++++++0\n";
  int r = FindPC(stamp, hesai_cloud);
  std::cerr << "++++++++++++++++++++++++++++++++++++1\n";
  std::cerr << r << " returned\n";
  if(r < CLOUD_FOUND) {
      std::cerr << "++++++++++++++++++++++++++++++++++++2\n";
      std::cerr << r << " is return\n";
    return r;
  }
  std::cerr << r << " means CLOUD_FOUND!!!!!!!!!!!!!!!\n";

  size_t ctr = 0;
  size_t null_ctr = 0;
  std::cerr << sloam_->lidarH() << ", " << sloam_->lidarW() << "= [h,w] \n";
  cloud_out->width = sloam_->lidarW();
  cloud_out->height = sloam_->lidarH();
  PointT p_invalid;
  p_invalid.x = 0.0;//std::numeric_limits<float>::quiet_NaN();
  p_invalid.y = p_invalid.x;
  p_invalid.z = p_invalid.x;
  p_invalid.intensity = p_invalid.x;
  for (int i = 0; i < sloam_->lidarW(); i++) {
    for (int j = 0; j < sloam_->lidarH(); j++) {
      if (hesai_cloud->points[ctr].ring == j) {
        PointT p;
        pcl::copyPoint(hesai_cloud->points[ctr], p);
        std::cerr << "cloud_out->points.size: " << cloud_out->points.size() << "\n";
        std::cerr << "hesai_cloud->points.size: " << hesai_cloud->points.size() << "\n";

        std::cerr << p.x << ", " << p.y << ", " << p.z << ", " << p.intensity
                  << ", " << ctr << ", " << i << ", " << j
                  << "= p[x,y,z,intensity,counter,i,j] \n";
        // std::cerr << "index out of range: " << ctr << "\n";
        std::cerr << "cloud_out->points.size: " << cloud_out->points.size()
                  << "\n";
        std::cerr << "hesai_cloud->points.size: " << hesai_cloud->points.size()
                  << "\n";

        cloud_out->points.push_back(p);
        ctr++;
      } else {
        // add an invalid point where there is a missing "pixel"
        cloud_out->points.push_back(p_invalid);
        null_ctr++;
      }

      std::cerr << "null_ctr = " << null_ctr << "\n";
      size_t null_check = cloud_out->points.size() - hesai_cloud->points.size();
      std::cerr << "cloud_out - hesai_cloud = "
                << cloud_out->points.size() - hesai_cloud->points.size()
                << "\n";
      std::cerr << "null_check = " << null_check << "\n";
    }
  }
  return r;
}

bool InputManager::callSLOAM(SE3 relativeMotion, ros::Time stamp)
{
    CloudT::Ptr cloud = pcl::make_shared<CloudT>();
    int r;
    if(use_hesai_){
        r = FindHesaiCloud(stamp, cloud);
    } else {
        r = FindPC(stamp, cloud);
    }

    if (r == CLOUD_FOUND)
    {
        odomQueue_.pop_front();
        SE3 keyPose = SE3();
        SE3 prevKeyPose = firstOdom_ ? SE3() : keyPoses_[keyPoses_.size() - 1];

        bool success;

        pcl_conversions::toPCL(stamp, cloud->header.stamp);
        success = sloam_->run(relativeMotion, prevKeyPose, cloud, stamp,
                              keyPose);

        if (success) {
          keyPoses_.push_back(keyPose);
          return true;
        }
    }
    else
    {
        if (r == CLOUD_TOO_NEW)
            odomQueue_.pop_front();
        ROS_DEBUG_THROTTLE(1, "Corresponding point cloud not found. Skipping.");
    }
    return false;
}

void InputManager::PCCb_(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{
    pcQueue_.push(cloudMsg);
    if (pcQueue_.size() > maxQueueSize_)
        pcQueue_.pop();
}



void InputManager::Odom2SlamTf()
{
    if (keyPoses_.size() == 0)
        return;

    auto slam_pose = keyPoses_[keyPoses_.size()-1];
    //vio_odom is odomTb, slam_pose which is slamTb
    //get odom2slam transform slamTodom = slamTb * inv(odomTb)

    // compute the tf based on odom when the graph slam optimization is called
    SE3 vio_odom = latestOdom.pose;
    SE3 odom2slam = slam_pose * (vio_odom.inverse());

    std::string parent_frame_id = map_frame_id_;
    std::string child_frame_id = odom_frame_id_;
    PublishOdomAsTf(sloam::toRosOdom_(odom2slam, map_frame_id_, latestOdom.stamp), parent_frame_id, child_frame_id);
    // publish pose AFTER TF so that the visualization looks correct
    pubPose_.publish(sloam::toRosOdom_(slam_pose, map_frame_id_, latestOdom.stamp));
}

void InputManager::PublishOdomAsTf(const nav_msgs::Odometry &odom_msg,
                                   const std::string &parent_frame_id,
                                   const std::string &child_frame_id)
{
    geometry_msgs::TransformStamped tf;
    tf.header = odom_msg.header;
    // note that normally parent_frame_id should be the same as
    // odom_msg.header.frame_id
    tf.header.frame_id = parent_frame_id;
    tf.child_frame_id = child_frame_id;
    tf.transform.translation.x = odom_msg.pose.pose.position.x;
    tf.transform.translation.y = odom_msg.pose.pose.position.y;
    tf.transform.translation.z = odom_msg.pose.pose.position.z;
    tf.transform.rotation = odom_msg.pose.pose.orientation;
    broadcaster_.sendTransform(tf);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sloam");
    ros::NodeHandle n("sloam");
    InputManager in(n);
    // ros::spin();

    ros::Rate r(20); // 10 hz
    while (ros::ok())
    {
        for (auto i = 0; i < 10; ++i)
        {
            ros::spinOnce();
            if (i % 5 == 0)
                in.Run();
            r.sleep();
        }
    }

    return 0;
}
