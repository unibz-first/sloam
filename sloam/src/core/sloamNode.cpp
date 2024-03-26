#include <sloamNode.h>
#include <pcl/common/io.h>
#include <chrono>
#include <boost/predef/version.h>

namespace sloam
{
SLOAMNode::SLOAMNode(const ros::NodeHandle &nh)
    : nh_(nh), it(nh_), maskPub_(it.advertise("debug/mask", 1))
{
  ROS_DEBUG_STREAM("Defining topics" << std::endl);

  debugMode_ = nh_.param("debug_mode", false);
  if (debugMode_) {
    ROS_DEBUG_STREAM("Running SLOAM in Debug Mode" << std::endl);
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                       ros::console::levels::Debug)) {
      ros::console::notifyLoggerLevelsChanged();
    }
  } else {
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                       ros::console::levels::Info)) {
      ros::console::notifyLoggerLevelsChanged();
    }
  }

  // Debugging Publishers
  groundCloudPub_ = nh_.advertise<sensor_msgs::PointCloud2>(
              "debug/labeled_ground", 1);
  treeCloudPub_ = nh_.advertise<sensor_msgs::PointCloud2>(
              "debug/labeled_trees", 1);
  // pubMapTreeFeatures_ = nh_.advertise<CloudT>("debug/map_tree_features", 1);
  pubMapGroundFeatures_ = nh_.advertise<CloudT>("debug/map_ground_features", 1);
  pubObsTreeFeatures_ = nh_.advertise<CloudT>("debug/obs_tree_features", 1);
  pubObsGroundFeatures_ = nh_.advertise<CloudT>("debug/obs_ground_features", 1);
  pubTrajectory_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "debug/trajectory", 1, true);
  pubMapTreeModel_ =
      nh_.advertise<visualization_msgs::MarkerArray>("map", 1, true);
  pubSubmapTreeModel_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "debug/map_tree_models", 1, true);
  pubObsTreeModel_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "debug/obs_tree_models", 1, true);
  pubMapGroundModel_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "debug/map_ground_model", 1);
  pubObsGroundModel_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "debug/obs_ground_model", 1);

  // SLOAM publishers
  pubObs_ = nh_.advertise<sloam_msgs::ROSObservation>("observation", 10);
  pubMapPose_ = nh_.advertise<geometry_msgs::PoseStamped>("map_pose", 10);

  // sub_graph_map_ = new obs_sub_type(nh_, "/graph_slam/submap", 20);

  ROS_DEBUG_STREAM("Init params" << std::endl);
  firstScan_ = true;
  tf_listener_.reset(new tf2_ros::TransformListener(tf_buffer_));
  initParams_();
}

  void SLOAMNode::initParams_()
  {
    // PARAMETERS
    float beam_cluster_threshold = nh_.param("beam_cluster_threshold", 0.1);
    int min_vertex_size = nh_.param("min_vertex_size", 2);
    float max_dist_to_centroid = nh_.param("max_dist_to_centroid", 0.2);
    int min_landmark_size = nh_.param("min_landmark_size", 4);
    float min_landmark_height = nh_.param("min_landmark_height", 1);

    Instance::Params params;
    params.beam_cluster_threshold = beam_cluster_threshold;
    params.min_vertex_size = min_vertex_size;
    params.max_dist_to_centroid = max_dist_to_centroid;
    params.min_landmark_size = min_landmark_size;
    params.min_landmark_height = min_landmark_height;

    // Creating objects for submodules
    std::string modelFilepath;
    nh_.param<std::string>("seg_model_path", modelFilepath, "");
    ROS_DEBUG_STREAM("MODEL PATH: " << modelFilepath);
    float fov = nh_.param("seg_lidar_fov", 22.5);
    lidar_w = nh_.param("seg_lidar_w", 2048);
    lidar_h = nh_.param("seg_lidar_h", 64);
    bool do_destagger = nh_.param("do_destagger", true);
    // img_d = 1 ...
    auto temp_seg = boost::make_shared<seg::Segmentation>(modelFilepath, fov, -fov, lidar_w, lidar_h, 1, do_destagger);
    segmentator_ = std::move(temp_seg);

    semanticMap_ = MapManager();

    graphDetector_ = Instance();
    graphDetector_.set_params(params);
    graphDetector_.reset_tree_id();

    fmParams_.scansPerSweep = 1;
    fmParams_.minTreeModels = nh_.param("min_tree_models", 15);
    fmParams_.minGroundModels = nh_.param("min_ground_models", 50);
    fmParams_.maxLidarDist = nh_.param("max_lidar_dist", 15.0);
    fmParams_.maxGroundLidarDist = nh_.param("max_ground_dist", 30.0);
    fmParams_.minGroundLidarDist = nh_.param("min_ground_dist", 0.0);

    fmParams_.twoStepOptim = nh_.param("two_step_optim", false);

    fmParams_.groundRadiiBins = nh_.param("ground_radii_bins", 5);
    fmParams_.groundThetaBins = nh_.param("ground_theta_bins", 36);
    fmParams_.groundMatchThresh = nh_.param("ground_match_thresh", 2.0);
    fmParams_.groundRetainThresh = nh_.param("ground_retain_thresh", 2.0);;

    fmParams_.maxTreeRadius = 0.3;
    fmParams_.maxAxisTheta = 10;
    fmParams_.maxFocusOutlierDistance = 0.5;
    fmParams_.roughTreeMatchThresh = nh_.param("rough_tree_match_thresh", 3.0);
    fmParams_.treeMatchThresh = nh_.param("tree_match_thresh", 1.0);

    fmParams_.AddNewTreeThreshDist = nh_.param("add_new_tree_thresh_dist", 2.0);

    fmParams_.featuresPerTree = nh_.param("features_per_tree", 2);
    fmParams_.numGroundFeatures = nh_.param("num_ground_features", 10);

    fmParams_.defaultTreeRadius = nh_.param("default_tree_radius", 0.1);

    // Mapper
    setFmParams(fmParams_);

    // Frame Ids
    nh_.param<std::string>("world_frame_id", world_frame_id_, "world");
    nh_.param<std::string>("map_frame_id", map_frame_id_, "map");
    nh_.param<std::string>("robot_frame_id", robot_frame_id_, "robot");
    // nh_.param<std::string>("velodyne_frame_id", velodyne_frame_id_, "world");
    ROS_DEBUG_STREAM("WORLD FRAME " << world_frame_id_);
    ROS_DEBUG_STREAM("MAP FRAME " << map_frame_id_);
    ROS_DEBUG_STREAM("ROBOT FRAME " << robot_frame_id_);
  }

  void SLOAMNode::publishMap_(const ros::Time stamp)
  {
    sloam_msgs::ROSObservation obs;
    obs.header.stamp = stamp;
    obs.header.frame_id = map_frame_id_;
    if(pubMapTreeModel_.getNumSubscribers() > 0)
    {
      auto semantic_map = semanticMap_.getMap();
      // for (const auto &cyl : semantic_map)
      // {
      //   sloam_msgs::ROSCylinder cylMsg;
      //   cylMsg.id = cyl.id;
      //   cylMsg.ray = {cyl.model.ray[0], cyl.model.ray[1], cyl.model.ray[2]};
      //   cylMsg.root = {cyl.model.root[0], cyl.model.root[1], cyl.model.root[2]};
      //   cylMsg.radius = cyl.model.radius;
      //   cylMsg.radii = cyl.model.radii;
      //   obs.treeModels.push_back(cylMsg);
      // }
      // // complete model
      // pubObs_.publish(obs);
      // viz only
      visualization_msgs::MarkerArray mapTMarkerArray;
      size_t cid = 500000;
      vizTreeModels(semantic_map, mapTMarkerArray, cid);
      pubMapTreeModel_.publish(mapTMarkerArray);
    }
  }

  CloudT::Ptr SLOAMNode::trellisCloud(const std::vector<std::vector<TreeVertex>> &landmarks)
  {
    CloudT::Ptr vtxCloud = pcl::make_shared<CloudT>();
    std::vector<float> color_values((int)landmarks.size());
    std::iota(std::begin(color_values), std::end(color_values), 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(color_values.begin(), color_values.end(), gen);
    int color_id = 0;

    for (auto landmark : landmarks)
    {
      for (auto vtx : landmark)
      {
        for (auto point : vtx.points)
        {
          point.intensity = color_values[color_id];
          vtxCloud->points.push_back(point);
        }
      }
      color_id++;
    }
    vtxCloud->height = 1;
    vtxCloud->width = vtxCloud->points.size();
    vtxCloud->header.frame_id = map_frame_id_;
    return vtxCloud;
  }


  bool SLOAMNode::runSegmentation(CloudT::Ptr cloud, ros::Time stamp,
                                  SE3 &outPose, const cv::Mat& rMask,
                                  SloamInput& sloamIn, SloamOutput& sloamOut) {
//    cv::imshow("whatever", rMask);
//    cv::waitKey(1);

    CloudT::Ptr groundCloud = pcl::make_shared<CloudT>();
    segmentator_->maskCloud(cloud, rMask, groundCloud, 1, true); //old: false
    ROS_INFO_STREAM("sloamNode.cpp organiSed groundCloud filled");

    //publishing labeled groundCloud
    sensor_msgs::PointCloud2 groundCloud_msg;
    pcl::toROSMsg(*groundCloud, groundCloud_msg);
    groundCloud_msg.header.frame_id = groundCloud->header.frame_id;
    groundCloud_msg.header.stamp = pcl_conversions::fromPCL(groundCloud->header.stamp);
    SLOAMNode::groundCloudPub_.publish(groundCloud_msg);

    CloudT::Ptr treeCloud = pcl::make_shared<CloudT>();
    //ROS_INFO_STREAM("sloamNode.cpp the treeCloud init'd!");
    segmentator_->maskCloud(cloud, rMask, treeCloud, 255, true);

    //publishing labeled groundCloud
    sensor_msgs::PointCloud2 treeCloud_msg;
    pcl::toROSMsg(*treeCloud, treeCloud_msg);
    treeCloud_msg.header.frame_id = treeCloud->header.frame_id;
    treeCloud_msg.header.stamp = pcl_conversions::fromPCL(treeCloud->header.stamp);
    SLOAMNode::treeCloudPub_.publish(treeCloud_msg);

    groundCloud->header = cloud->header;
    sloamIn.groundCloud = groundCloud;
    ROS_WARN_STREAM("Num ground features available: " << groundCloud->width);

    // Trellis graph instance segmentation
    graphDetector_.computeGraph(cloud, treeCloud, sloamIn.landmarks);
    ROS_INFO_STREAM(
        "Num Landmarks detected with Trellis: " << sloamIn.landmarks.size());
    ROS_INFO_STREAM("Num Map Landmarks: " << sloamIn.mapModels.size());

    if (debugMode_ && !firstScan_) {
      // DEBUG TOPICS
      auto mapGFeats = getPrevGroundFeatures();
      mapGFeats.header.frame_id = map_frame_id_;
      pubMapGroundFeatures_.publish(mapGFeats);
      pubMapGroundModel_.publish(
          vizGroundModel(getPrevGroundModel(), map_frame_id_, 444));
      // maskCloudPub_.publish()
    }

    bool success = RunSloam(sloamIn, sloamOut);
    semanticMap_.updateMap(sloamOut.tm, sloamOut.matches);
    ROS_DEBUG_STREAM("Publishing Results. Success? " << success << std::endl);

    if (success) {
        // this needs some delay finagling
      if (firstScan_ && sloamOut.tm.size() > 0) firstScan_ = false;

      publishMap_(stamp);
      pubMapPose_.publish(makeROSPose(sloamOut.T_Map_Curr, map_frame_id_));
      trajectory.push_back(sloamOut.T_Map_Curr);
      outPose = sloamOut.T_Map_Curr;

      //////////////////////////////////////////////////////
      // FOR DEBUGGING
      // Republish input map
      if (debugMode_) {
        visualization_msgs::MarkerArray trajMarkers = vizTrajectory(trajectory);
        pubTrajectory_.publish(trajMarkers);

        visualization_msgs::MarkerArray mapTMarkerArray;
        size_t cid = 200000;
        vizTreeModels(sloamIn.mapModels, mapTMarkerArray, cid);
        pubSubmapTreeModel_.publish(mapTMarkerArray);

        // Publish aligned observation
        size_t cylinderId = 100000;
        visualization_msgs::MarkerArray obsTMarkerArray;
        vizTreeModels(sloamOut.tm, obsTMarkerArray, cylinderId);
        pubObsTreeModel_.publish(obsTMarkerArray);

        // groundCloud->header.frame_id = map_frame_id_;
        // pubObsGroundFeatures_.publish(groundCloud);
        auto trellis = trellisCloud(sloamIn.landmarks);
        pcl::transformPointCloud(*trellis, *trellis,
                                 sloamOut.T_Map_Curr.matrix());
        pubObsTreeFeatures_.publish(trellis);

        // After running sloam, prevGround is already relative to this obs
        auto obsGFeats = getPrevGroundFeatures();
        obsGFeats.header.frame_id = map_frame_id_;
        pubObsGroundFeatures_.publish(obsGFeats);
        pubObsGroundModel_.publish(
            vizGroundModel(getPrevGroundModel(), map_frame_id_, 111));
      }
    }
    return success;
  }

  bool SLOAMNode::prepSegmentation(const SE3 &initialGuess,
                                   const SE3 &prevKeyPose,
                                   SloamInput &sloamIn) {
    SE3 poseEstimate = prevKeyPose * initialGuess;
    // sloamIn.initialGuess = initialGuess;
    // sloamIn.prevPose = mapCurrPose_;
    sloamIn.poseEstimate = poseEstimate;
    sloamIn.distance = initialGuess.translation().norm();
    semanticMap_.getSubmap(poseEstimate, sloamIn.mapModels);

    if (!firstScan_ && sloamIn.mapModels.size() == 0)
    {
      ROS_DEBUG("Discarding msg");
      return false;
    }
    return true;
  }

  void SLOAMNode::maskImgViz(cv::Mat& mask_img) const {
    cv::MatIterator_<uchar> it;
    // create visible mask image
    for (it = mask_img.begin<uchar>(); it != mask_img.end<uchar>(); ++it) {
      if ((*it) == 1) {
        (*it) = 127;
      }
    }

  }

  bool SLOAMNode::run(const SE3 initialGuess, const SE3 prevKeyPose,
                      CloudT::Ptr cloud, ros::Time stamp, SE3 &outPose) {
    SloamInput sloamIn;
    SloamOutput sloamOut;
    if(!prepSegmentation(initialGuess, prevKeyPose, sloamIn)){
        return false;
    }
    ROS_INFO_STREAM("Entering Callback. Lidar data stamp: " << stamp);
    // RUN SEGMENTATION
    cv::Mat rMask = cv::Mat::zeros(lidar_h, lidar_w, CV_8U);

    //ROS_WARN_STREAM("sloamNode.cpp run() rMask[h,w]: " << rMask.rows << ", " << rMask.cols);
    segmentator_->run(cloud, rMask);

    // copy the mask for visualisation
    rMask.copyTo(maskViz_);
    cv::Size sz(2000,64);
    cv::resize(rMask, maskViz_, sz);

    // replace some colors to make the mask easier to see on RViz
    maskImgViz(maskViz_);
    std_msgs::Header h;
    h.frame_id = cloud->header.frame_id;
    h.stamp = stamp;
    auto maskMsg_ = cv_bridge::CvImage(h, "mono8", maskViz_).toImageMsg();
    // publish the mask image
    maskPub_.publish(maskMsg_);

    // pass the mask to a function that does the actual segmentation on the point clouds
    return runSegmentation(cloud, stamp, outPose, rMask, sloamIn, sloamOut);
  }

} // namespace sloam
