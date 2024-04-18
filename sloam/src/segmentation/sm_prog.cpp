#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ros/package.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "hesai_point_types.h"
#include "inference.h"
#include "trellis.h"

// using NetworkInput = std::vector<std::vector<float>>;

void netInputToImg(seg::Segmentation::NetworkInput ni, cv::Mat ni_img) {
  cv::Mat ni_img_out(64, 2048, CV_8U);
  // TODO make copy of cv::Mat so that original mask_img stays 1/0 for FCN
  std::cout << "got ni_img w/ dims= " << ni_img.rows << "x" << ni_img.cols
            << std::endl;
  for (int i = 0; i < ni_img.rows; ++i) {
    for (int j = 0; j < ni_img.cols; ++j) {
      ni_img.at<float>(i, j) = ni[i][j];
      std::cout << ni[i][j] << std::endl;
    }
  }
  std::cout << "finished loop" << std::endl;
  std::cout << "made ni_img_out w/ dims= " << ni_img_out.rows << "x"
            << ni_img_out.cols << std::endl;
  ni_img.convertTo(ni_img_out, CV_8U);
  cv::imshow("netInput_depth", ni_img_out);
  cv::waitKey(0);
  //    return ni_img_out;
}

void maskImgViz(cv::Mat mask_img, bool do_write = false) {
  cv::Mat maskViz_ = cv::Mat::zeros(mask_img.size(),mask_img.type());
  mask_img.copyTo(maskViz_); // avoids affecting mask_img input
  cv::MatIterator_<uchar> it;
  // create visible mask image
  for (it = maskViz_.begin<uchar>(); it != maskViz_.end<uchar>(); ++it) {
    if ((*it) == 1) {
      (*it) = 127;
    }
  }
  cv::Mat maskViz_resize_;
  cv::Size sz(mask_img.cols,128); //resize
  cv::resize(maskViz_, maskViz_resize_, sz);
  cv::imshow("maskviz", maskViz_resize_); //visalize
  cv::waitKey(0);
  cv::imshow("mask_og", mask_img); //visalize sanity check
  cv::waitKey(0);
}

void hesaiCloudToOrganizedCloud(const HesaiPointCloud& hesai_cloud,
                                CloudT::Ptr& org_cloud) {
  size_t counter = 0;
  org_cloud->width = 2000;//_img_w;
  org_cloud->height = 32; //_img_h;
  org_cloud->is_dense = false;
  org_cloud->points.resize(org_cloud->width * org_cloud->height);
  org_cloud->header = hesai_cloud.header;

  std::ofstream ohc_csv("/tmp/sloam_debug" +
                        std::to_string(hesai_cloud.header.stamp) +
                        "_ohc.csv");
  ohc_csv << "counter, proj_xs, proj_ys, x, y, z" << std::endl;

//  while(counter < hesai_cloud.points.size()){
    for (int i = 0; i < org_cloud->width; i++) {
      for (std::uint16_t j = 0; j < org_cloud->height; j++) {
        if (hesai_cloud.points[counter].ring == j) {
          PointT p;
          p.x = hesai_cloud.points[counter].x;
          p.y = hesai_cloud.points[counter].y;
          p.z = hesai_cloud.points[counter].z;
          p.intensity = hesai_cloud.points[counter].intensity;
          org_cloud->at(i, j) = p;
          ohc_csv << counter << ", " << i << ", " << j << ", " <<
                     p.x << ", " << p.y << ", " << p.z << std::endl;
          counter++;
        } else {
          org_cloud->at(i, j).x = std::numeric_limits<float>::quiet_NaN();
          org_cloud->at(i, j).y = std::numeric_limits<float>::quiet_NaN();
          org_cloud->at(i, j).z = std::numeric_limits<float>::quiet_NaN();
          org_cloud->at(i, j).intensity = std::numeric_limits<float>::quiet_NaN();
          ohc_csv << counter << ", " << i << ", " << j << ", " <<
                     ", nan, nan, nan" << std::endl;
        }
      }
    }
//  }
  ohc_csv.close();
  std::cout << "ohc_csv written ++++++++++++++++++++++++++++++++++++++++++++\n";
}
using CloudT = pcl::PointCloud<pcl::PointXYZI>;

int main() {
  // filesys shit:
  std::string home_dir = getenv("HOME");
//  std::string sloam_dir = ros::package::getPath("sloam");
  std::string seq_dir = home_dir + "/repos/lidar-bonnetal/digikittiforest/sequences";
  std::string pcd_dir = seq_dir + "/00/point_clouds/";//"/home/mcamurri/Datasets/";////
  //  std::string pcdfile = pcd_dir + "cloud_1693566299_538213000.pcd";//"1693566415538365.pcd";
  std::string model_path = home_dir + "/Downloads/darknet21pp_hpc.onnx";   // /logs/squeezenetV2_1_1_30_2k.onnx";
  std::string pcdfile = home_dir + "/bags/1693566424537803.pcd";//"1693566415538365.pcd";
  std::string tmp_dir = "/tmp/sloam_debug/";

  try {
    seg::Segmentation imgseg(model_path, 15, -16.1,
                             2000, 32, 1, false);
// create clouds
    HesaiPointCloud::Ptr hesaiCloud = pcl::make_shared<HesaiPointCloud>();
    CloudT::Ptr cloud = pcl::make_shared<CloudT>();
    CloudT::Ptr groundCloud = pcl::make_shared<CloudT>();
    CloudT::Ptr treeCloud = pcl::make_shared<CloudT>();
// load cloud
    std::cout << "Loading " << pcdfile << " ... \n";
    if (pcl::io::loadPCDFile(pcdfile, *hesaiCloud) == 0) {
      std::cout << "Loaded " << cloud->width * cloud->height
                << " data points from .pcd" << std::endl;
    } else {
      std::cerr << "Load failed\n";
    }

    // organize hesai cloud
    hesaiCloudToOrganizedCloud(*hesaiCloud, cloud);
    pcl::io::savePCDFile(tmp_dir + "hesai_in.pcd", *hesaiCloud);
    pcl::io::savePCDFile(tmp_dir + "hesai_org.pcd", *cloud);
    // MAKE MASK IMAGE
    cv::Size s(2000, 32);
    cv::Mat range_image(s, CV_8U);
    cv::Mat range_image_float(s, CV_32F);
    cv::Mat mask_image(s, CV_8U);

    // seg objects, projection memory
    seg::Segmentation::NetworkInput net_input;
    std::vector<float> scan;

    // creating CloudVector and net_input
    imgseg.cloudToCloudVector(cloud, scan);
    net_input = imgseg._doProjection(scan, cloud->width * cloud->height);

    // build mask image from network input
    imgseg.runNetwork(net_input, mask_image);
    maskImgViz(mask_image); // visualize using method above.

    // from sloamNode.cpp SloamNode::runSegmentation
    imgseg.maskCloud(cloud, mask_image, groundCloud, 1, false); // old: 1, f...
    groundCloud->header = cloud->header;
    ROS_WARN_STREAM("Num ground features available: " << groundCloud->width);

    imgseg.maskCloud(cloud, mask_image, treeCloud, 255, true);
    treeCloud->header = cloud->header;
    ROS_WARN_STREAM("Num tree features available: " << treeCloud->size());


    // trellis.cpp graph instance segmentation
    Instance graphDetector_;
    graphDetector_ = Instance ();
    std::vector<std::vector<TreeVertex>> landmarks;
    graphDetector_.computeGraph(cloud, treeCloud, landmarks);

  } catch (const Ort::Exception& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "Ort Error Code: " << e.GetOrtErrorCode() << std::endl;
  }
}
