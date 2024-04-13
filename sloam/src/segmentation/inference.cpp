#include "inference.h"


namespace seg {

using NetworkInput = Segmentation::NetworkInput;

Segmentation::Segmentation(const std::string modelFilePath,
                           const float fov_up, const float fov_down,
                           const int img_w, const int img_h, const int img_d, bool do_destagger)
{
  _fov_up = fov_up / 180.0 * M_PI;    // field of view up in radians
  _fov_down = fov_down / 180.0 * M_PI;  // field of view down in radians
  _fov = std::abs(_fov_down) + std::abs(_fov_up); // get field of view total in radians
  _img_w = img_w;
  _img_h = img_h;
  _img_d = img_d;
  _verbose = true;
  _do_destagger = do_destagger;
  //initializeImages();

  const std::string sessionName = "SLOAMSeg";
  // specify number of CPU threads allowed for semantic segmentation inference to use
  _startONNXSession(sessionName, modelFilePath, false, 3);
}

void Segmentation::_startONNXSession(
    const std::string sessionName, const std::string modelFilePath, bool useCUDA, size_t numThreads){
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(numThreads);
  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
  // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
  // (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible optimizations
  sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  if (useCUDA){
    // Using CUDA backend
    // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
    OrtCUDAProviderOptions cuda_options{0};
    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
  }

  auto env = boost::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, sessionName.c_str());
  _env = boost::move(env);

  auto session = boost::make_shared<Ort::Session>(*_env, modelFilePath.c_str(), sessionOptions);
  _session = boost::move(session);

  auto memInfo = boost::make_shared<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(
                                                       OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault));
  _memoryInfo = boost::move(memInfo);

  // Ort::AllocatorWithDefaultOptions allocator;
  // INPUT
  const char* inputName = _session->GetInputName(0, _allocator);
  std::cout << "[Segmentation] Input Name: " << inputName << std::endl;

  Ort::TypeInfo inputTypeInfo = _session->GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  _inputDims = inputTensorInfo.GetShape();
  // _inputDims = {1, 64, 2048, 2}; WRONG
  // _inputDims = {1, 1, 64, 2048}; VERIFIED: stable_bdarknet.onnx
  // _inputDims = {1, 1, 32, 2000}; darknet21pp_hpc.onnx
  std::cout << "[Segmentation] Input Dimensions: "; printVector(_inputDims);

  // OUTPUT
  const char* outputName = _session->GetOutputName(0, _allocator);
  std::cout << "[Segmentation] Output Name: " << outputName << std::endl;

  Ort::TypeInfo outputTypeInfo = _session->GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
  // ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

  _outputDims = outputTensorInfo.GetShape();
  // _outputDims = {1, 64, 2048, 2}; WRONG!!!!!!!!!!!!!!!!!!!!
  // _outputDims = {1, 3, 64, 2048}; VERIFIED: stable_bdarknet.onnx
  // _outputDims = {1, 3, 64, 2048}; darknet21pp_hpc.onnx
  std::cout << "[Segmentation] Output Dimensions: "; printVector(_outputDims);

  // _inputTensorSize = _img_w * _img_h * _img_d;
  // _outputTensorSize = _img_w * _img_h * _img_d;
  _inputTensorSize = vectorProduct(_inputDims);
  _outputTensorSize = vectorProduct(_outputDims);
  _inputNames = {inputName};
  _outputNames = {outputName};
}
/*
void Segmentation::hesaiPointcloudToImage(const HesaiPointCloud& cloud,
                                          HesaiImages& hesaiImages,
                                          CloudT::Ptr& padded_cloud) const {
  size_t counter = 0;
  for (int i = 0; i < hesaiImages.range_.cols; i++) {
    for (std::uint16_t j = 0; j < hesaiImages.range_.rows; j++) {
      if (cloud.points[counter].ring == j) {
        double range =
            std::sqrt(cloud.points[counter].x * cloud.points[counter].x +
                      cloud.points[counter].y * cloud.points[counter].y +
                      cloud.points[counter].z * cloud.points[counter].z);
        std::uint8_t range_pixel = ((range - 0.05) / (120.0 - 0.05)) * 255;
        // populate cv::Mat's w range & intensity vals
        hesaiImages.range_.at<std::uint8_t>((int)j, i) = range_pixel;
        hesaiImages.intensity_.at<std::uint8_t>(j, i) =
            cloud.points[counter].intensity;
        hesaiImages.range_precise_.at<double>(j, i) = range;
        PointT p;
        pcl::copyPoint(cloud.points[counter], p);
        //                std::cout << "x,y,z: " << p.x << "," << p.y << "," <<
        //                p.z << std::endl;
        padded_cloud->points.push_back(p);
        counter++;  // goes here.
      } else {
        // TODO: replace with the last pixel value?
        hesaiImages.range_.at<std::uint8_t>(j, i) = 0;
        hesaiImages.intensity_.at<std::uint8_t>(j, i) = 0;
        hesaiImages.range_precise_.at<double>(j, i) = 0.0;
        padded_cloud->points.push_back(PointT());
      }
    }
  }
  cv::resize(hesaiImages.range_precise_, hesaiImages.range_resized_,
             hesaiImages.range_resized_.size(), 0, 0, cv::INTER_NEAREST);
  std::cerr << "resized image \n";
  // critical for ONNX network
  hesaiImages.range_resized_.convertTo(hesaiImages.range_resized_float_,
                                       CV_32F);
  std::cerr << "resized image to float \n";
  hesaiImages.range_resized_.convertTo(hesaiImages.range_resized_show_, CV_8U);
  std::cerr << "resized image to 8bit \n";
  cv::imshow("resized range", hesaiImages.range_resized_show_);
  cv::waitKey(0);
}
*/
NetworkInput Segmentation::sortOrder(int h, int w,
                                     std::vector<float>* rgs,
                                     std::vector<size_t>* xps,
                                     std::vector<size_t>* yps){
  // order in decreasing depth
  int num_points = h*w;
  std::cerr << "sorting all your base \n";
  std::cerr << "h: " << h << "\n";
  std::cerr << "w: " << w << "\n";
  //    NetworkInput range_image;

  std::vector<size_t> orders;
  if (rgs!= nullptr) {
    orders = sort_indexes(*rgs);
  }

  //    std::cerr << "orders: ";
  //    for (size_t idx : orders) {
  //        std::cerr << idx << " ";
  //    }
  //    std::cerr << "\n";
  std::cerr << "orders ok \n";
  std::vector<size_t> sorted_proj_xs;
  sorted_proj_xs.reserve(num_points);
  std::cerr << "built spxs \n";
  std::vector<size_t> sorted_proj_ys;
  sorted_proj_ys.reserve(num_points);
  std::cerr << "built spys \n";
  std::vector<std::vector<float>> inputs;
  inputs.reserve(num_points);
  for (size_t idx : orders){ // ordered-range idx queries x,y vectors
    //        std::cerr << "in loop \n";
    sorted_proj_xs.push_back((*xps)[idx]);
    sorted_proj_ys.push_back((*yps)[idx]);
    std::vector<float> input = {(*rgs)[idx]};
    inputs.push_back(input);
  }
  // best easter egg typo ever, thx Jens. github.com/PRBonn/lidar-bonnetal
  // assing to images
  std::cerr << "Sizes: rgs=" << (rgs != nullptr ? rgs->size() : 0)
            << ", xps=" << (xps != nullptr ? xps->size() : 0)
            << ", yps=" << (yps != nullptr ? yps->size() : 0)
            << "\n";
  NetworkInput range_image(num_points);
  std::cerr << "range to net \n";
  std::vector<float> invalid_input =  {0.0f};

  std::cerr << "Size of range_image: " << range_image.size() << "\n";
  std::cerr << "w: " << w << "\n";
  // zero initialize
  for (uint32_t i = 0; i < range_image.size(); ++i) {
    range_image[i] = invalid_input;
  }
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    //       range_image[int(sorted_proj_ys[i] * w + sorted_proj_xs[i])] = inputs[i]; // w = 2048 **
    uint32_t index = int(sorted_proj_ys[i] * w + sorted_proj_xs[i]);
    if (index < range_image.size()) {
      range_image[index] = inputs[i];
    } else {
      std::cerr << "Index out of bounds: " << index << "\n";
    }
    range_image[int(sorted_proj_ys[i] * w + sorted_proj_xs[i])] = inputs[i]; // w = 2048 **
  }
  std::cerr << "returning range image \n";

  return range_image;  //DANGER: != cv::Mat HesaiImages.{range_,}
}

std::vector<std::vector<float>> Segmentation::_doProjection(
    const std::vector<float>& scan,
    const uint32_t& num_points,
    std::vector<float>* rgs,
    std::vector<size_t>* xps,
    std::vector<size_t>* yps)
{

  // std::vector<float> invalid_input =  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> invalid_input =  {0.0f};

  std::vector<float> ranges;
  std::vector<float> xs;
  std::vector<float> ys;
  std::vector<float> zs;
  std::vector<float> intensitys;

  std::vector<float> proj_xs_tmp;
  std::vector<float> proj_ys_tmp;

  for (uint32_t i = 0; i < num_points; i++) {
    float x = scan[4 * i];
    float y = scan[4 * i + 1];
    float z = scan[4 * i + 2];
    float intensity = scan[4 * i + 3];
    float range = std::sqrt(x*x+y*y+z*z);
    ranges.push_back(range);
    if(rgs != nullptr){
      rgs->push_back(range);
    }
    xs.push_back(x);
    ys.push_back(y);
    zs.push_back(z);
    intensitys.push_back(intensity);

    // get angles
    float yaw = -std::atan2(y, x);
    float pitch = std::asin(z / range);

    // get projections in image coords
    float proj_x = 0.5 * (yaw / M_PI + 1.0); // in [0.0, 1.0]
    float proj_y = 1.0 - (pitch + std::abs(_fov_down)) / _fov; // in [0.0, 1.0]

    // scale to image size using angular resolution
    proj_x *= _img_w; // in [0.0, W]
    proj_y *= _img_h; // in [0.0, H]

    // round and clamp for use as index
    proj_x = std::floor(proj_x);
    proj_x = std::min(_img_w - 1.0f, proj_x);
    proj_x = std::max(0.0f, proj_x); // in [0,W-1]
    proj_xs_tmp.push_back(proj_x);
    if(xps != nullptr){
      xps->push_back(proj_x);
    }

    proj_y = std::floor(proj_y);
    proj_y = std::min(_img_h - 1.0f, proj_y);
    proj_y = std::max(0.0f, proj_y); // in [0,H-1]
    proj_ys_tmp.push_back(proj_y);

    if(yps != nullptr){
      yps->push_back(proj_y);
    }
  }
  //  std::cerr << "inference.cpp _doProjection() HxW: " <<
  //               _img_h << ", " << _img_w << "\n";
  //  NetworkInput range_image = sortOrder(_img_h, _img_w, rgs, xps, yps);

  // stope a copy in original order
  proj_xs = proj_xs_tmp;
  proj_ys = proj_ys_tmp;

  // order in decreasing depth
  std::vector<size_t> orders = sort_indexes(ranges);
  std::vector<float> sorted_proj_xs;
  sorted_proj_xs.reserve(num_points);
  std::vector<float> sorted_proj_ys;
  sorted_proj_ys.reserve(num_points);
  std::vector<std::vector<float>> inputs;
  inputs.reserve(num_points);

  for (size_t idx : orders){
    sorted_proj_xs.push_back(proj_xs[idx]);
    sorted_proj_ys.push_back(proj_ys[idx]);
    // std::vector<float> input = {ranges[idx], xs[idx], ys[idx], zs[idx], intensitys[idx]};
    // std::vector<float> input = {ranges[idx], zs[idx], xs[idx], ys[idx], intensitys[idx]};
    std::vector<float> input = {ranges[idx]};
    inputs.push_back(input);
  }

  // assing to images
  NetworkInput range_image(_img_w * _img_h);
  // zero initialize
  for (uint32_t i = 0; i < range_image.size(); ++i) {
    range_image[i] = invalid_input;
  }

  for (uint32_t i = 0; i < inputs.size(); ++i) {
    range_image[int(sorted_proj_ys[i] * _img_w + sorted_proj_xs[i])] = inputs[i];
  }

  return range_image;
}

void Segmentation::_makeTensor(
    std::vector<std::vector<float>>& projected_data,
    std::vector<float>& tensor, std::vector<size_t>& invalid_idxs){
  // TODO LOAD THIS TOO
  std::vector<float> _img_means = {12.97};
  std::vector<float> _img_stds = {12.35};
  // arkansas
  // std::vector<float> _img_means = {12.97, -0.21, -0.13, 0.26, 942.62};
  // std::vector<float> _img_stds = {12.35, 12.04, 12.95, 2.86, 1041.79};
  // kitti
  // std::vector<float> _img_means = {12.12, 10.88, 0.23, -1.04, 0.21};
  // std::vector<float> _img_stds = {12.32, 11.47, 6.91, 0.86, 0.16};

  int channel_offset = _img_h * _img_w;
  bool all_zeros = false;
  int ctr = 0;

  std::cout << "projected_data.size() = " << projected_data.size() << "\n";

  for (uint32_t pixel_id = 0; pixel_id < projected_data.size(); pixel_id++){
    // check if the pixel is invalid
    all_zeros = std::all_of(projected_data[pixel_id].begin(), projected_data[pixel_id].end(), [](int i) { return ((i==0.0f) || (isnan(i))); });
    if (all_zeros) {
      invalid_idxs.push_back(pixel_id);
      //std::cout << "pixel_id = " << pixel_id <<"\n";
      ctr +=1;
    }
    //std::cout << "projected_data[pixel_id].data() = " << projected_data[pixel_id].data() << std::endl;
    for (int i = 0; i < _img_d; i++) {
      // normalize the data
      if (!all_zeros) {
        projected_data[pixel_id][i] = (projected_data[pixel_id][i] - _img_means[i]) / _img_stds[i];
      }

      int buffer_idx = channel_offset * i + pixel_id;
      //   ((float*)_hostBuffers[_inBindIdx])[buffer_idx] = projected_data[pixel_id][i];
      tensor[buffer_idx] = projected_data[pixel_id][i];
    }
  }
  std::cerr<< "_makeTensor all_zeros = true for " << ctr << " pixels \n";
}

void Segmentation::_destaggerCloud(const Cloud::Ptr cloud, Cloud::Ptr& outCloud){
  bool col_valid = true;

  for(auto irow = 0; irow < _img_h; irow++){
    for(auto icol = 0; icol < _img_w; icol++){
      auto im_col = icol;
      // Ouster data needs a shift every other row
      if(irow % 2 == 0){
        im_col += 32;
        if(im_col < 0 || im_col > _img_w){
          col_valid = false;
          im_col = im_col % _img_w;
        }
      }

      if(col_valid){
        const auto& pt = cloud->points[irow * _img_w + icol];
        auto& outPt = outCloud->points[irow * _img_w + im_col];
        // const auto& pt = cloud->at(icol, irow);
        // auto& outPt = outCloud->at(im_col, irow);
        outPt.x = pt.x;
        outPt.y = pt.y;
        outPt.z = pt.z;
      }

      col_valid = true;
    }
  }
}

void Segmentation::maskCloud(const Cloud::Ptr cloud,
                             cv::Mat mask,
                             Cloud::Ptr& outCloud,
                             unsigned char val,
                             bool organized) {

  CloudT::Ptr tempCloud = pcl::make_shared<CloudT>();
  CloudT::Ptr temp2Cloud = pcl::make_shared<CloudT>();
  size_t numPoints = mask.rows * mask.cols;
  Point p;
  p.x = p.y = p.z = p.intensity = std::numeric_limits<float>::quiet_NaN();
  size_t org_pt_ctr = 0;
  size_t org_nan_ctr = 0;
  size_t pt_ctr = 0;  
  size_t nan_ctr = 0;
  // for temp2Cloud
  auto tpxs = proj_xs;
  auto tpys = proj_ys;
  std::vector<std::vector<PointT>> org_pc(mask.cols,
                                          std::vector<PointT>(mask.rows, p));
  assert((mask.rows*mask.cols) == (cloud->width*cloud->height));
  std::cerr << "input cloud    w,h,size,is_dense: " <<
               cloud->width << ", " << cloud->height << ", " <<
               cloud->points.size() << ", " << cloud->is_dense << "\n";
  std::cerr << "pre outCloud   w,h,size,is_dense: " <<
               outCloud->width << ", " << outCloud->height << ", " <<
               outCloud->points.size() << ", " << outCloud->is_dense << "\n";

  // fill temp2Cloud if organized
  //        if(organized){

  std::set<int> tc_idcs, t2c_idcs;
  std::vector<std::vector<int>> oc_idcs(mask.cols, std::vector<int>(mask.rows, -1)); // for removeNaN test...

  for (size_t i = 0; i < cloud->points.size(); i++) {
    auto m = mask.at<uchar>(proj_ys[i], proj_xs[i]);
    if (m == val) {
      org_pc[proj_xs[i]][proj_ys[i]] = cloud->points[i];// !xy flip!
      oc_idcs[proj_xs[i]][proj_ys[i]] = i;
//      std::cout << i << ", " << proj_xs[i] << ", " << proj_ys[i] << "\n";
    }
  }

  for (size_t i = 0; i < mask.cols; ++i) {
    for (size_t j = 0; j < mask.rows; ++j) {
      auto np = org_pc[i][j];
      temp2Cloud->points.push_back(np);
      if(!std::isnan(np.x) && !std::isnan(np.y) && !std::isnan(np.z)){
        org_pt_ctr++;
        if (oc_idcs[i][j] != -1){
          t2c_idcs.insert(oc_idcs[i][j]);
        }
      } else {
//        std::cerr << "org_pc[i][j] [i,j,p.x,p.y,p.z,p.intensity]: [" <<
//                     i << ", " << j << ", " << p.x << ", " <<
//                     p.y << ", " << p.z << ", " << p.intensity << "]\n";
        ++org_nan_ctr;
      }
    }
  }
//  std::cerr << "mid t2c_idcs.size() : " << t2c_idcs.size() << "\n";
  std::cerr << "mid temp2Cloud w,h,size,is_dense: " <<
               temp2Cloud->width << ", " << temp2Cloud->height << ", " <<
               temp2Cloud->points.size() << ", " << temp2Cloud->is_dense << "\n";
  //        }

  for (int i = 0; i < numPoints; i++) {
    auto m = mask.at<uchar>(proj_ys[i], proj_xs[i]);
    if(m == val){
      tempCloud->points.push_back(cloud->points[i]);
      tc_idcs.insert(i);
      ++pt_ctr;

    } else if(organized){
      tempCloud->points.push_back(p);
      ++nan_ctr;
    }
  }
//  std::cerr << "tc_idcs.size() : " << tc_idcs.size() << "\n";
  std::cerr << "mid tempCloud  w,h,size,is_dense: " <<
               tempCloud->width << ", " << tempCloud->height << ", " <<
               tempCloud->points.size() << ", " << tempCloud->is_dense << "\n";
  std::vector<int> diff_idcs;
  std::set_difference(tc_idcs.begin(), tc_idcs.end(),
                      t2c_idcs.begin(), t2c_idcs.end(),
                      std::inserter(diff_idcs, diff_idcs.begin()));
  CloudT::Ptr diff_pc = pcl::make_shared<CloudT>();
  pcl::copyPointCloud(*cloud, diff_idcs, *diff_pc);
  diff_pc->width = diff_pc->points.size();
  diff_pc->height = 1;
  diff_pc->is_dense = false;
  std::cerr << "mid diffCloud  w,h,size,is_dense: " <<
               diff_pc->width << ", " << diff_pc->height << ", " <<
               diff_pc->points.size() << ", " << diff_pc->is_dense << "\n";
  pcl::io::savePCDFile("/home/mchang/Downloads/diff_tmp.pcd", *diff_pc);


  if(organized){
//    outCloud->clear();
    if (_do_destagger){
      _destaggerCloud(tempCloud, outCloud);
    } else {
      pcl::copyPointCloud(*tempCloud, *outCloud); // *temp2Cloud breaks it
      std::cerr << "mid outCloud   w,h,size,is_dense: " <<
                   outCloud->width << ", " << outCloud->height << ", " <<
                   outCloud->points.size() << ", " << outCloud->is_dense << "\n";
    }

    temp2Cloud->width = _img_w;
    temp2Cloud->height = _img_h;
    temp2Cloud->is_dense = true;
    pcl::io::savePCDFile("/home/mchang/Downloads/tCloud_tmp2.pcd", *temp2Cloud);
    outCloud->width = _img_w;
    outCloud->height = _img_h;
    outCloud->is_dense = (cloud->points.size() == numPoints); // old = true;
    pcl::io::savePCDFile("/home/mchang/Downloads/tCloud.pcd", *outCloud);

  } else {
    pcl::copyPointCloud(*temp2Cloud, *outCloud);
//    std::vector<int> indices;
//    pcl::removeNaNFromPointCloud(*temp2Cloud, indices);

    //change dims to save .pcd ASCII files
    tempCloud->width = tempCloud->points.size();
    tempCloud->height = 1;
    pcl::io::savePCDFile("/home/mchang/Downloads/gCloud_tmp.pcd", *tempCloud);

    temp2Cloud->width = temp2Cloud->points.size();
    temp2Cloud->height = 1;
    pcl::io::savePCDFile("/home/mchang/Downloads/gCloud_tmp2.pcd", *temp2Cloud);

    outCloud->width = outCloud->points.size(); //old: outCloud
    outCloud->height = 1;
    outCloud->is_dense = false;
    pcl::io::savePCDFile("/home/mchang/Downloads/gCloud.pcd", *outCloud);
//    pcl::io::savePCDFile("/home/mchang/Downloads/postNaN_gCloud_tmp2.pcd", *temp2Cloud);
  }

  std::cerr << "post tempCloud  w,h,size,is_dense: " <<
               tempCloud->width << ", " << tempCloud->height << ", " <<
               tempCloud->points.size() << ", " << tempCloud->is_dense << "\n";
  std::cerr << "post temp2Cloud w,h,size,is_dense: " <<
               temp2Cloud->width << ", " << temp2Cloud->height << ", " <<
               temp2Cloud->points.size() << ", " << temp2Cloud->is_dense << "\n";
  std::cerr << "post outCloud   w,h,size,is_dense: " <<
               outCloud->width << ", " << outCloud->height << ", " <<
               outCloud->points.size() << ", " << outCloud->is_dense << "\n";
  std::cerr << "[org_{pt,nan}_ctr] [{pt,nan}_ctr] = ["
            << org_pt_ctr << ", " << org_nan_ctr << "] ["
            << pt_ctr << ", " << nan_ctr << "]\n";
  //print pixel counts
  cv::Mat binaryMask = (mask == val);
  int px_count = cv::countNonZero(binaryMask);
  std::cerr << "mask.val, px_count: [" << static_cast<int>(val) <<
               ", " << px_count << "]\n";
  outCloud->header = cloud->header;
  // ^ repeated in SloamNode.cpp, but need for tree/groundCloud viz.
}

//void Segmentation::maskCloud(const CloudT::Ptr cloud,
//                             cv::Mat mask,
//                             CloudT::Ptr& outCloud,
//                             unsigned char val,
//                             bool organised) {

//    CloudT::Ptr tempCloud = pcl::make_shared<CloudT>();
//    CloudT::Ptr temp2Cloud = pcl::make_shared<CloudT>();
//    //tempCloud->is_dense = true; // assume cloud has no NaN or inf points //breaks?
//    size_t numPoints = mask.rows * mask.cols;
//    assert((mask.rows*mask.cols) == (cloud->width*cloud->height));

//    Point p;
//    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
//    size_t nan_ctr = 0;
//    size_t pt_ctr = 0;

//    // new way: temp2Cloud via org_pc
//    if (organised) {
//      std::vector<std::vector<PointT>> org_pc(mask.cols,
//                                              std::vector<PointT>(mask.rows, p));

//      for (size_t i = 0; i < cloud->points.size(); i++) {
//        auto m = mask.at<uchar>(proj_ys[i], proj_xs[i]);
//        if (m == val) {
//          org_pc[proj_xs[i]][proj_ys[i]] = cloud->points[i];// achtung! xy flip!
//        }
//      }

//      for (size_t i = 0; i < mask.cols; ++i) {
//        for (size_t j = 0; j < mask.rows; ++j) {
//          temp2Cloud->points.push_back(org_pc[i][j]);
//          pt_ctr++;
//        }
//      }

//      temp2Cloud->is_dense = (cloud->points.size() == numPoints);
//      temp2Cloud->width = _img_w;
//      temp2Cloud->height = _img_h;

//      // old way: tempCloud
//      for (int i = 0; i < numPoints; i++) {
//          size_t proj_idx = proj_ys[i] * _img_w + proj_xs[i];
//          unsigned char m = mask.data[proj_idx * sizeof(unsigned char)];

//          if(m == val){
//              tempCloud->points.push_back(cloud->points[i]);
//              ++pt_ctr;
//          } else if(organised){
//              tempCloud->points.push_back(p);
//              ++nan_ctr;
//          }
//      }

//      if (_do_destagger){
//          _destaggerCloud(tempCloud, outCloud);
//          std::cerr << "destaggering\n";
//      } else {
//        pcl::copyPointCloud(*temp2Cloud, *outCloud);
//      }

//    } else {

//      outCloud->width = outCloud->points.size();
//      outCloud->height = 1;
//      outCloud->is_dense = false; //test
//      outCloud->clear();

//// new shit:
////      for (int i = 0; i < cloud->points.size(); i++) {
////        auto m = mask.at<uchar>(proj_ys[i], proj_xs[i]);
////        if (m == val) {
////          outCloud->points.push_back(cloud->points[i]);
////          ++pt_ctr;
////        }
////      }
//    }

//    //print pixel counts
//    cv::Mat binaryMask = (mask == val);
//    int px_count = cv::countNonZero(binaryMask);
//    std::cerr << "mask.val, px_count: [" << static_cast<int>(val) <<
//                 ", " << px_count << "]\n";

//    // test uncomment
////    std::cerr << "outCloud is dense?: " << outCloud->is_dense << "\n";
//    auto outCloud_diff = numPoints - outCloud->points.size();
//    ROS_INFO_STREAM("numPoints: " << numPoints << "; outCloud diff = " << outCloud_diff);
//    ROS_INFO_STREAM("points in outCloud: " << outCloud->points.size());
//}

void Segmentation::_mask(const float* output,
                         const std::vector<size_t>& invalid_idxs,
                         cv::Mat& maskImg){
  size_t channel_offset = _img_w * _img_h;
  unsigned char _n_classes = 3;
  std::vector<unsigned char> max;
  std::vector<int> class_counts(_n_classes, 0);

  for (int pixel_id = 0; pixel_id < channel_offset; pixel_id++){
    int max_idx = pixel_id;
    unsigned char out_idx = 0;
    for (unsigned char i = 1; i < _n_classes; i++) {
      int buffer_idx = channel_offset * i + pixel_id;
      // if(output[max_idx] < output[buffer_idx] && output[buffer_idx] > 0.8){
      if(output[max_idx] < output[buffer_idx]){
        max_idx = buffer_idx;
        out_idx = static_cast<unsigned char>(i);
      }
    }
    if(out_idx == 2) out_idx = 255;
    max.push_back(out_idx);
    class_counts[out_idx]++;
  }

  for(const int idx : invalid_idxs){
    max[idx] = 0;
    class_counts[0]++;
    if(max[idx] == 255) class_counts[255]--;
    else class_counts[max[idx]]--;
  }

  memcpy(maskImg.data, max.data(), max.size()*sizeof(unsigned char));


  std::cerr << "count [0,1,255] = [" << class_counts[0] << ", " <<
               class_counts[1] << ", " << class_counts[255] << "]\n";

}

void Segmentation::_preProcessRange(const cv::Mat& img, cv::Mat& pImg, float maxDist) {
  std::vector<cv::Mat> channels(2);
  cv::split(img, channels);

  pImg = img.clone();

  // Normalize
  double minVal = 0;
  double maxVal = 0;
  cv::minMaxIdx(channels[1], &minVal, &maxVal);
  float rangeMultiplier = 1.0 / 100.0;
  float intensityMultipler = 1.0 / (float)maxVal;
  cv::multiply(pImg, cv::Scalar(rangeMultiplier, intensityMultipler),
               pImg);
}

void Segmentation::_argmax(const float *in, cv::Mat& maskImg){
  std::vector<unsigned char> max;
  size_t numClasses_ = 2;
  size_t outSize = _img_h * _img_w * numClasses_;
  for (unsigned int i = 0; i < outSize; i += numClasses_) {
    unsigned int maxIdx = i;
    unsigned char outIdx = 0;
    for (unsigned int c = 1; c < numClasses_; c++) {
      if (in[maxIdx] < in[i + c]) {
        maxIdx = i + c;
        outIdx = c;
      }
    }
    if(outIdx == 1) outIdx = 255;
    max.push_back(outIdx);
  }
  memcpy(maskImg.data, max.data(), max.size()*sizeof(unsigned char));
}

void Segmentation::runERF(cv::Mat& rImg, cv::Mat& maskImg){
  cv::Mat pImg(rImg.rows, rImg.cols, CV_32FC2, cv::Scalar(0));
  _preProcessRange(rImg, pImg, 30);

  std::vector<float> outputTensorValues(_outputTensorSize);
  std::vector<float> inputTensorValues(_inputTensorSize);

  auto imgSize = pImg.rows * pImg.cols * pImg.channels();
  memcpy(inputTensorValues.data(), pImg.data, imgSize * sizeof(float));
  std::cout << "Tensor size: " << _inputTensorSize << std::endl;
  std::cout << "Tensor size: " << _outputTensorSize << std::endl;
  std::cout << "Data size: " << imgSize << std::endl;

  std::vector<Ort::Value> inputTensors;
  std::vector<Ort::Value> outputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
                           *_memoryInfo, inputTensorValues.data(), _inputTensorSize, _inputDims.data(),
                           _inputDims.size()));

  outputTensors.push_back(Ort::Value::CreateTensor<float>(
                            *_memoryInfo, outputTensorValues.data(), _outputTensorSize,
                            _outputDims.data(), _outputDims.size()));

  _session->Run(Ort::RunOptions{nullptr}, _inputNames.data(),
                inputTensors.data(), 1, _outputNames.data(),
                outputTensors.data(), 1);

  float* outData = outputTensors.front().GetTensorMutableData<float>();
  // int dims[] = {3,64,2048};
  // cv::Mat result = cv::Mat(3, dims, CV_32F, outData);
  // cv::FileStorage file("/opt/bags/inf/res.ext", cv::FileStorage::WRITE);
  // Write to file!
  // file << "matName" << result;
  // std::cout << sizeof(outData) << std::endl;
  _argmax(outData, maskImg);
}

void Segmentation::cloudToCloudVector(const Cloud::Ptr &cloud, std::vector<float> &cloudVector) const{
  for (const auto& point : cloud->points) {
    cloudVector.push_back(point.x);
    cloudVector.push_back(point.y);
    cloudVector.push_back(point.z);
    cloudVector.push_back(point.intensity);
  }
  std::cout << "cloudVector of size: " << cloudVector.size() << std::endl;
  //std::cout << "cloudVector.front(): " << cloudVector.front() << std::endl;
  //std::cout << "cloudVector.back(): " << cloudVector.back() << std::endl;
}

void Segmentation::cloudToNetworkInput(const CloudT::Ptr& cloud,
                                       NetworkInput& netInput){
  std::vector<float> cloudVector;
  cloudToCloudVector(cloud, cloudVector);

  if (_verbose) {
    std::cout << "Projecting data" << std::endl;
    _timer.tic();
  }
  // added rgs, xps, and yps projected indices stored from sortOrder()
  netInput = _doProjection(cloudVector, cloud->width * cloud->height);//, &_rgs);//,
  //                             &_rgs, &_xps, &_yps);

  // taken from sm_prog. create cv::Mat objs
  // TODO: replace w/ _HesaiImages struct, make lightweight
  cv::Size s(_img_w, _img_h);
  cv::Mat range_image = cv::Mat(s, CV_8U);
  cv::Mat range_image_float = cv::Mat(s, CV_32F);
  cv::Mat mask_image(s, CV_8U);
  //    std::cerr << "cv::Mat imgs created, with:\n";
  //    std::cerr << "height = " << s.height << "\n";
  //    std::cerr << "width " << s.width << "\n";

  // fill range images w projected vals.
  //   for (int i = 0; i < _img_h*_img_w; ++i) {
  //       range_image_float.at<float>(proj_ys[i],proj_xs[i]) = _rgs[i];
  //   }
  //   range_image_float *= 255.0/50.5;
  //   range_image_float.convertTo(range_image, CV_8U);

  //    std::vector<float> flattened_inp;
  //    for (int i = 0; i < _img_h*_img_w; ++i) {
  //        size_t proj_idx = proj_ys[i] * _img_w + proj_xs[i];
  //        flattened_inp.push_back(netInput[proj_idx][0]);
  //    }
  // cv::FileStorage file("/opt/bags/inf/inp.ext", cv::FileStorage::WRITE);
  // file << "mat" << result;

  if (_verbose) {
    _timer.toc();
  }
}


void Segmentation::runNetwork(NetworkInput &netInput, cv::Mat &maskImg){
  std::vector<float> outputTensorValues(_outputTensorSize);
  std::vector<float> inputTensorValues(_inputTensorSize);
  std::vector<size_t> invalid_idxs;

  if(_verbose){
    std::cout << "Making tensor" << std::endl;
    _timer.tic();
  }
  _makeTensor(netInput, inputTensorValues, invalid_idxs);

  if(_verbose){
    _timer.toc();
  }

  if(_verbose){
    std::cout << "Running Inference" << std::endl;
    _timer.tic();
  }
  std::vector<Ort::Value> inputTensors;
  std::vector<Ort::Value> outputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
                           *_memoryInfo, inputTensorValues.data(), _inputTensorSize, _inputDims.data(),
                           _inputDims.size()));

  outputTensors.push_back(Ort::Value::CreateTensor<float>(
                            *_memoryInfo, outputTensorValues.data(), _outputTensorSize,
                            _outputDims.data(), _outputDims.size()));

  _session->Run(Ort::RunOptions{nullptr}, _inputNames.data(),
                inputTensors.data(), 1, _outputNames.data(),
                outputTensors.data(), 1);
  if(_verbose){
    _timer.toc();
  }

  if(_verbose){
    std::cout << "Masking" << std::endl;
    _timer.tic();
  }

  float* outData = outputTensors.front().GetTensorMutableData<float>();
  _mask(outData, invalid_idxs, maskImg);

  if(_verbose){
    _timer.toc();
  }
}

void Segmentation::run(const CloudT::Ptr cloud, cv::Mat& maskImg){
  NetworkInput ni;
  ROS_INFO_STREAM("Segmentation::run 1");
  cloudToNetworkInput(cloud, ni);
  ROS_INFO_STREAM("Segmentation::run 2");
  runNetwork(ni, maskImg);
}

void Segmentation::speedTest(const Cloud::Ptr cloud, size_t numTests){
  std::vector<float> cloudVector;
  for (const auto& point : cloud->points) {
    cloudVector.push_back(point.x); cloudVector.push_back(point.y);
    cloudVector.push_back(point.z); cloudVector.push_back(point.intensity);
  }

  std::cout << "PROJECTION" << std::endl;
  auto netInput = _doProjection(cloudVector, cloud->width*cloud->height);

  std::cout << "TO TENSOR" << std::endl;
  std::vector<float> outputTensorValues(_outputTensorSize);
  std::vector<float> inputTensorValues(_inputTensorSize);
  std::vector<size_t> invalid_idxs;
  _makeTensor(netInput, inputTensorValues, invalid_idxs);

  std::cout << "SETUP" << std::endl;
  std::vector<Ort::Value> inputTensors;
  std::vector<Ort::Value> outputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
                           *_memoryInfo, inputTensorValues.data(), _inputTensorSize, _inputDims.data(),
                           _inputDims.size()));

  outputTensors.push_back(Ort::Value::CreateTensor<float>(
                            *_memoryInfo, outputTensorValues.data(), _outputTensorSize,
                            _outputDims.data(), _outputDims.size()));

  // Measure latency
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  for (int i = 0; i < numTests; i++)
  {
    _session->Run(Ort::RunOptions{nullptr}, _inputNames.data(),
                  inputTensors.data(), 1, _outputNames.data(),
                  outputTensors.data(), 1);
  }
  std::chrono::steady_clock::time_point end =
      std::chrono::steady_clock::now();
  std::cout << "Minimum Inference Latency: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
               .count() /
               static_cast<float>(numTests)
            << " ms" << std::endl;
}
} //namespace segmentation
