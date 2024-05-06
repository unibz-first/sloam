#include <trellis.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <chrono>

Instance::Instance() { tree_id_ = 0; }

void Instance::computeClusterDistances(const CloudT::Ptr pc,
                                       const std::vector<pcl::PointIndices> &label_indices) {
  std::vector<Eigen::Vector4f> centroids;
  int size_thresh = 15;
  for (const auto& indices : label_indices) {
    if (indices.indices.size() > size_thresh) {
      CloudT cluster;
      pcl::copyPointCloud(*pc, indices.indices, cluster);
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(cluster, centroid);
      centroids.push_back(centroid);
    }
  }

//  for (size_t i = 0; i < centroids.size() - 1; i++) {
//    for (size_t j = i + 1; j < centroids.size(); j++) {
//      float distance = (centroids[i] - centroids[j]).norm();
//      std::cout << "Distance between clusters " << i << " and " << j << ": " << distance << "\n";
//    }
//  }
  std::cout << "clusters > " << size_thresh << ": " << centroids.size() << "\n";
}

void Instance::findClusters(const CloudT::Ptr pc,
                            pcl::PointCloud<pcl::Label> &euclidean_labels,
                            std::vector<pcl::PointIndices> &label_indices) {
  if (pc->size() == 0) return;
  pcl::EuclideanClusterComparator<
      PointT, pcl::Label>::Ptr euclidean_cluster_comparator =
      pcl::make_shared<pcl::EuclideanClusterComparator<PointT, pcl::Label>>();
  euclidean_cluster_comparator->setInputCloud(pc);
//  euclidean_cluster_comparator->compare();
  std::cout << "pc for findClusters organized?: " << pc->isOrganized() << "\n";
  euclidean_cluster_comparator->setDistanceThreshold(2.0f, false);

  pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label>
      euclidean_segmentation(euclidean_cluster_comparator);
  euclidean_segmentation.setInputCloud(pc);
  euclidean_segmentation.segment(euclidean_labels, label_indices);

  for (size_t i = 0; i < label_indices.size() ; i++){
//    std::cout << "LABEL INDICES SIZE: " <<
//                 label_indices.at(i).indices.size() << std::endl;
    if (label_indices.at(i).indices.size () > 15){
      CloudT cluster;
      pcl::copyPointCloud(*pc, label_indices.at(i).indices, cluster);
      ROS_DEBUG_STREAM(cluster.width << " x " << cluster.height);
      std::stringstream ss;
      ss << "/tmp/sloam_debug/" << "cloud_cluster_" << i << ".pcd";
      pcl::io::savePCDFile(ss.str(), cluster, false);
    }
  }
}

TreeVertex Instance::computeTreeVertex(CloudT::Ptr beam, int label){
  TreeVertex v;
  Slash filteredPoints;
  PointT median;
  Scalar radius;
  bool valid = computeVertexProperties(beam, filteredPoints, median, radius);

  v.treeId = label;
  v.prevVertexSize = 0;
  v.points = filteredPoints;
  v.coords = median;
  // v.isRoot = false;
  v.isValid = valid;
  v.beam = 0;
  v.radius = radius;
  return v;
}

bool Instance::computeVertexProperties(CloudT::Ptr &pc, Slash& filteredPoints,
                                       PointT& median_point, Scalar& radius) {
  // Compute median in each x,y,z
  int num_points = pc->points.size();
  int middle_point = (int)(num_points / 2.0);
  Scalar median_x = 0;
  Scalar median_y = 0;
  Scalar median_z = 0;

  std::sort(pc->points.begin(), pc->points.end(),
            [](const PointT &p1, const PointT &p2) { return p1.x < p2.x; });

  median_x = pc->points[middle_point].x;

  std::sort(pc->points.begin(), pc->points.end(),
            [](const PointT &p1, const PointT &p2) { return p1.y < p2.y; });
  median_y = pc->points[middle_point].y;

  std::sort(pc->points.begin(), pc->points.end(),
            [](const PointT &p1, const PointT &p2) { return p1.z < p2.z; });
  median_z = pc->points[middle_point].z;

  // PointT median_point;
  median_point.x = median_x;
  median_point.y = median_y;
  median_point.z = median_z;

  for (const auto &point : pc->points) {
    if (euclideanDistance(point, median_point) < params_.max_dist_to_centroid) {
      filteredPoints.push_back(point);
    }
  }

  if(filteredPoints.size() > 1){
    PointT pointA = filteredPoints[0];
    PointT pointB = filteredPoints[filteredPoints.size() - 1];
    radius = euclideanDistance(pointA, pointB);
    return true;
  }
  return false;
}

void Instance::findTrees(const CloudT::Ptr pc,
                         pcl::PointCloud<pcl::Label>& euclidean_labels,
                         std::vector<pcl::PointIndices>& label_indices,
                         std::vector<std::vector<TreeVertex>>& landmarks){

    for (size_t i = 0; i < label_indices.size(); i++){
      // TODO: .yaml this 80 shit.
      if (label_indices.at(i).indices.size () > 30){
        std::vector<TreeVertex> tree;
//        std::cout << "******************treeID: "<< label_indices.at(i) << "\n";
        std::cerr << "findTrees TREEID label: " << i << "\n";
        for (int row_idx = pc->height - 1; row_idx >= 0; --row_idx) {
          CloudT::Ptr beam = pcl::make_shared<CloudT>();
          for (size_t col_idx = 0; col_idx < pc->width; ++col_idx) {
            // if the points of label i at [row*2000 + col]
            if(euclidean_labels.points[row_idx * pc->width + col_idx].label == i){
              PointT p = pc->at(col_idx, row_idx);
              beam->points.push_back(p);
//              std::cerr << "Beam col,row: " << col_idx << "," << row_idx << "\n";
            }
          }
          // TODO: .yaml this shit.
          if(beam->points.size() > 3){
            std::cerr << "Beam size, first, last, row : "
                      << beam->points.size() << ", "
                      << beam->points.front() << ", "
                      << beam->points.back() << ", "
                      << row_idx << "*************\n";
//            std::cerr << "Beam col,row: " << col_idx << "," << row_idx << "\n";
            TreeVertex v = computeTreeVertex(beam, i);
            if(v.isValid) tree.push_back(v);
          }
        }
        // TODO: .yaml this shit,
        if(tree.size() > 30){
          std::cerr << "Tree size: " << tree.size() << "\n";
          // TODO: .yaml this shit.
          if(tree.size() > 56) {
            tree.resize(56);
          }
          landmarks.push_back(tree);
        }
      }
    }
}

void Instance::computeGraph(const CloudT::Ptr cloud, const CloudT::Ptr tree_cloud,
                            std::vector<std::vector<TreeVertex>> &landmarks) {
  pcl::PointCloud<pcl::Label> euclidean_labels;
  std::vector<pcl::PointIndices> label_indices;
  findClusters(tree_cloud, euclidean_labels, label_indices);
  std::cerr << "Number of clusters found: " << label_indices.size() << "\n";
  // sanity check distance threshold. delete later.
  computeClusterDistances(tree_cloud, label_indices);
  findTrees(tree_cloud, euclidean_labels, label_indices, landmarks);
  std::cerr << "Num landmarks found: " << landmarks.size() << "\n";

}
