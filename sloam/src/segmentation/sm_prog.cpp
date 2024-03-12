#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "hesai_point_types.h"
#include "inference.h"

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
  cv::MatIterator_<uchar> it;
  // create visible mask image
  for (it = mask_img.begin<uchar>(); it != mask_img.end<uchar>(); ++it) {
    if ((*it) == 1) {
      (*it) = 127;
    }
  }
  cv::imshow("mask", mask_img);
  cv::waitKey(0);
}

using CloudXYZI = pcl::PointCloud<pcl::PointXYZI>;

int main() {
  std::string home_dir = getenv("HOME");
  std::string seq_dir = home_dir + "/repos/lidar-bonnetal/digikittiforest/sequences";
  std::string pcd_dir = seq_dir + "/00/point_clouds/";
  std::string pcdfile = pcd_dir + "cloud_1693566299_538213000.pcd";
  std::string model_path = home_dir + "/Downloads/darknet21pp_hpc.onnx";   // /logs/squeezenetV2_1_1_30_2k.onnx";

  try {
    seg::Segmentation imgseg(model_path, 15, -16,
                             2000, 32, 1, false);
// create cloud
    CloudXYZI::Ptr cloud = pcl::make_shared<CloudXYZI>();

    std::cout << "Loading " << pcdfile << " ... \n";
    if (pcl::io::loadPCDFile(pcdfile, *cloud) == 0) {
      std::cout << "Loaded " << cloud->width * cloud->height
                << " data points from .pcd" << std::endl;
    } else {
      std::cerr << "Load failed\n";
    }

    cv::Size s(2000, 32);
    cv::Mat range_image(s, CV_8U);
    cv::Mat range_image_float(s, CV_32F);
    cv::Mat mask_image(s, CV_8U);

    // seg objects, projection memory
    seg::Segmentation::NetworkInput net_input;
    std::vector<float> rgs;
    std::vector<size_t> xps, yps;
    std::vector<float> scan;
    
// creating CloudVector
    imgseg.cloudToCloudVector(cloud, scan);
    // build network input
    net_input = imgseg._doProjection(scan, cloud->width * cloud->height, &rgs,
                                     &xps, &yps);
    for (int i = 0; i < xps.size(); i++) {
      // std::cerr << "Setting pixel y = " << yps[i] << " x = " << xps[i] << "
      // to " << rgs[i] << "\n"; SLOAM DEFAULT VERSION
      range_image_float.at<float>(yps[i], xps[i]) = rgs[i];
    }
    range_image_float *= 255.0/50.5; // 50 meters range tops
    // create visible network input
    range_image_float.convertTo(range_image, CV_8U);
    cv::imshow("range_image", range_image);
    cv::waitKey(0);

    // build mask image from network input
    imgseg.runNetwork(net_input, mask_image);

    maskImgViz(mask_image);
  } catch (const Ort::Exception& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "Ort Error Code: " << e.GetOrtErrorCode() << std::endl;
  }
}
