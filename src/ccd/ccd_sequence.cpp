#include "ccd/ccd_sequence.hpp"

#include<opencv2/imgproc.hpp>

#include<stdexcept>

//分析一帧ccd的信息
CcdFrameAnalysis analyzeCcdMask(
    std::size_t file_index,
    const std::int64_t timestamp_ms,
    const cv::Mat& mask,
    const int edge_margin)
    {
         // 1. 检查输入
    if (mask.empty()) {
        throw std::invalid_argument("mask is empty");
    }

    if (mask.type() != CV_8UC1) {
        throw std::invalid_argument(
            "mask type must be CV_8UC1"
        );
    }

    if (edge_margin < 0 ||
        edge_margin * 2 >= mask.rows) {
        throw std::invalid_argument(
            "edge_margin is outside the valid range"
        );
    }

    // 2. 保存当前帧的基本信息
    CcdFrameAnalysis result;

    result.file_index = file_index;
    result.timestamp_ms = timestamp_ms;


    // 3. 计算连通组件
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;

        const int connected_component_count=cv::connectedComponentsWithStats(mask, labels, stats, centroids,8,CV_32S);

        for(int label=1;label<connected_component_count;label++)

        {
            CcdObjectInfo object;

            object.file_index = file_index;
            object.timestamp_ms = timestamp_ms;

            object.label = label;
            object.area=stats.at<int>(label,cv::CC_STAT_AREA);

            object.bounding_rect.x=stats.at<int>(label,cv::CC_STAT_LEFT);
            object.bounding_rect.y=stats.at<int>(label,cv::CC_STAT_TOP);
            object.bounding_rect.width=stats.at<int>(label,cv::CC_STAT_WIDTH);
            object.bounding_rect.height=stats.at<int>(label,cv::CC_STAT_HEIGHT);

            object.center.x=centroids.at<double>(label,0);
            object.center.y=centroids.at<double>(label,1);

          
            //姿态分析
            bool top_near=0,bottom_near=0;
            if(object.bounding_rect.y<=edge_margin) top_near=1;
            if(object.bounding_rect.y+object.bounding_rect.height>=mask.rows-edge_margin) bottom_near=1;

            if(top_near&&bottom_near) throw std::invalid_argument("object is too large");
            else if(top_near) object.position=CcdObjectPosition::TopBoundary;
            else if(bottom_near) object.position=CcdObjectPosition::BottomBoundary;
            else object.position=CcdObjectPosition::Interior;
  result.objects.push_back(object);

        }

        result.labels = labels.clone();//返回labels的副本，因为labels在函数结束后会被销毁

        return result;
    }
