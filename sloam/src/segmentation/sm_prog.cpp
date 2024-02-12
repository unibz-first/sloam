#include <iostream>
#include <string>
#include "inference.h"
#include "hesai_point_types.h"
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

//using NetworkInput = std::vector<std::vector<float>>;

void netInputToImg(seg::Segmentation::NetworkInput ni, cv::Mat ni_img) {
    cv::Mat ni_img_out(64, 2048, CV_8U);
    // TODO make copy of cv::Mat so that original mask_img stays 1/0 for FCN
    std::cout << "got ni_img w/ dims= " << ni_img.rows
              << "x" << ni_img.cols << std::endl;
    for (int i = 0; i < ni_img.rows; ++i){
        for (int j = 0; j < ni_img.cols; ++j){
            ni_img.at<float>(i, j) = ni[i][j];
            std::cout << ni[i][j] <<std::endl;
        }
    }
    std::cout << "finished loop" << std::endl;
    std::cout << "made ni_img_out w/ dims= " << ni_img_out.rows
              << "x" << ni_img_out.cols << std::endl;
    ni_img.convertTo(ni_img_out, CV_8U);
    cv::imshow("netInput_depth", ni_img_out);
    cv::waitKey(0);



//    return ni_img_out;
}

void maskImgViz(cv::Mat mask_img, bool do_write = false){
    cv::MatIterator_<uchar> it;
    // create visible mask image
    for(it = mask_img.begin<uchar>(); it != mask_img.end<uchar>(); ++it){
        if((*it) == 1){
            (*it) = 127;
        }
    }
    cv::imshow("mask", mask_img);
    cv::waitKey(0);
//    cv::imshow("Point cloud range", range_image);
//    cv::waitKey(0);
    //    cv::imshow("Point cloud intensity", intensity_image);
    //    cv::waitKey(0);
//    if (do_write){
//        // change params here
//        cv::imwrite(std::string(pcd_dir+"83332000.png"), mask_img);
////        cv::imwrite("/home/mchang/test_hesai_range_image.png", range_image);
//        cv::imwrite("/home/mchang/test_hesai_intensity_image.png", intensity_image);
//    }

}

//seg::Segmentation segConstruct(){
////     construct the seg::Segmentation class
//    seg::Segmentation imgseg(
//                "/home/mchang/sloam/models/stable_bdarknet.onnx",
//                22.5, -22.5, 2048, 64, 1, true);
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
//    return imgseg();
//}

int main () {
    bool use_hesai = true;
    std::string pcd_dir = "/home/mchang/Downloads/bags/sloam_bags/";
    std::string pcdfile = pcd_dir + "83332000" + ".pcd";

    seg::Segmentation imgseg(
                "/home/mchang/sloam/models/stable_bdarknet.onnx",
                22.5, -22.5, 2048, 64, 1, true);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);

    // hesai variation ********************
    seg::Segmentation hesaiseg(
            "/home/mchang/sloam/models/stable_bdarknet.onnx",
            16.0, -15.0, 2000, 32, 1, false);
    HesaiPointCloud::Ptr hesai_cloud;
    CloudT::Ptr padded_cloud;
    if (use_hesai) {
        pcdfile = pcd_dir + "test_hesai_cloud" + ".pcd";
        // std::cout << pcdfile << std::endl;
        hesai_cloud = HesaiPointCloud::Ptr(new HesaiPointCloud);
        padded_cloud = CloudT::Ptr(new CloudT);
    }

    // for both
    pcl::io::loadPCDFile(pcdfile, *cloud);
    std::cout << "Loaded " << cloud->width * cloud->height
                << " data points from .pcd" << std::endl;
    cv::Mat range_image(64, 2048, CV_8U);
    cv::Mat range_image_float(64, 2048, CV_32F);
//    cv::Mat intensity_image(64, 2048, CV_8U);
    cv::Mat mask_image(64, 2048, CV_8U);

    // seg objects, projection memory
    seg::Segmentation::NetworkInput net_input;
    std::vector<float> rgs;
    std::vector<size_t> xps, yps;
    std::vector<float> scan;

    if (use_hesai) {
        //hesai cloud needs to become padded
        hesaiseg.hesaiPointcloudToImage(*hesai_cloud, hesaiseg._hesaiImages,
                                        padded_cloud);
//        hesaiseg.cloudToCloudVector(padded_cloud, scan);
        net_input = hesaiseg._doHesaiProjection(hesaiseg._hesaiImages,
                                                &rgs, &xps, &yps);
        cv::imshow("hesaiImages", hesaiseg._hesaiImages.range_resized_);
        cv::waitKey(0);
        hesaiseg.runNetwork(net_input, mask_image);
    } else {
        imgseg.cloudToCloudVector(cloud, scan);
        // build network input
        net_input = imgseg._doProjection(scan, cloud->width*cloud->height,
                                         &rgs, &xps, &yps);
        for(int i = 0; i<xps.size(); i++){
            // std::cerr << "Setting pixel y = " << yps[i] << " x = " << xps[i] << " to " << rgs[i] << "\n";
            // SLOAM DEFAULT VERSION
            range_image_float.at<float>(yps[i], xps[i]) = rgs[i];
        }
        // create visible network input
        range_image_float.convertTo(range_image,CV_8U);

        // build mask image from network input
        imgseg.runNetwork(net_input, mask_image);
    }
    maskImgViz(mask_image);

    //    std::cerr << " HELLOOO \n";
    //    std::cerr << " HELLOOO2 \n";
    //    std::cerr << " HELLOOO 3\n";

}
