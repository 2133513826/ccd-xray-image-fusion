#include "ccd/ccd_preprocess.hpp"
#include "common/progress.hpp"
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>


CcdBackground buildCcdBackground(const std::vector<CcdPair>& CcdPairs,
    const CcdOptions& p){
   
   if(CcdPairs.empty())
    throw std::runtime_error("No Background samples");

    const auto size= CcdPairs.front().depth_image.size();
    // 背景使用前20帧；不足20帧时使用全部输入帧。
    const std::size_t sample_count =
        std::min<std::size_t>(CcdPairs.size(), 20);
    std::vector<cv::Mat> values(sample_count);
    std::cout << "CCD background samples: " << sample_count
              << "/" << CcdPairs.size() << '\n';
    showProgress("CCD background", 0, size.height);
//将深度图像转换为位64浮点数
    for(size_t i=0;i<sample_count;++i){
       const auto& pair = CcdPairs[i];

        if (pair.depth_image.empty() ||
    pair.intensity_image.empty() ||
    pair.depth_image.size() != size ||
    pair.intensity_image.size() != size ||
    pair.depth_image.channels() != 1 ||
    pair.intensity_image.type() != CV_8UC1) {
    throw std::runtime_error(
        "Invalid background sample: " + pair.id);
}

        pair.depth_image.convertTo(values[i], CV_64F);
    }

    CcdBackground b{
        cv::Mat::zeros(size,CV_64F),
        cv::Mat::zeros(size,CV_64F),
        cv::Mat::zeros(size,CV_8U)
    };

    //中值函数
    auto media=[](std::vector<double>& v){
      auto m=v.begin()+v.size()/2;
      std::nth_element(v.begin(),m,v.end());
      return *m;
    };

    // 复用采样缓存，避免每个像素重新分配内存。
    std::vector<double> ds, is;
    ds.reserve(sample_count);
    is.reserve(sample_count);
    for(int y=0;y<size.height;++y){
        for(int x=0;x<size.width;++x){
            ds.clear();
            is.clear();

            for(size_t i=0;i<sample_count;++i){
                double d=values[i].at<double>(y,x);
                if(std::isfinite(d)&&d!=p.invalid_depth){
                    ds.push_back(d);
                    is.push_back(CcdPairs[i].intensity_image.at<uchar>(y,x));
                }
            }

            // 收集全部有效样本后只计算一次；空样本不求中位数。
            if(!ds.empty() &&
               ds.size()>=static_cast<size_t>(sample_count*p.valid_ratio)){
                b.depth.at<double>(y,x)=media(ds);
                b.intensity.at<double>(y,x)=media(is);
                b.field.at<uchar>(y,x)=255;
            }
        }
        showProgress("CCD background", y + 1, size.height);
    }

     return b;
   
}

cv::Mat detectCcd(const cv::Mat& depth, const cv::Mat& intensity,
    const CcdBackground& b, const CcdOptions& p) {
   
    if(depth.size() != b.depth.size() || intensity.size() != b.intensity.size()|| intensity.type() != CV_8UC1|| depth.channels() != 1) {
        throw std::runtime_error("Depth and intensity images must have the same size as the background.");
    }

    cv::Mat d,mask= cv::Mat::zeros(depth.size(), CV_8U);
    depth.convertTo(d,CV_64F);
//遍历每个像素点，判断是否为目标,生成掩膜
    for(int y=0;y<d.rows;y++)
      for(int x=0;x<d.cols;x++)
      {
        if(b.field.at<uchar>(y,x)==0) continue;
        double v=d.at<double>(y,x);
        bool depth_valid=std::isfinite(v)&&v!=p.invalid_depth&&std::abs(v-b.depth.at<double>(y,x))>p.depth_diff;
        bool intensity_valid=std::abs(intensity.at<uchar>(y,x)-b.intensity.at<double>(y,x))>p.intensity_diff;
        if(depth_valid||intensity_valid){
            mask.at<uchar>(y,x)=255;
        }
      }
//形态学处理，去除噪声
    auto kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,{p.kernel,p.kernel});

    cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,kernel);
    cv::bitwise_and(mask,b.field,mask);

    cv::Mat labels,stats,centers;
    cv::Mat result= cv::Mat::zeros(mask.size(), CV_8U);
//连通组件分析，去除小区域
    int n=cv::connectedComponentsWithStats(mask,labels,stats,centers);
    for(int i=1;i<n;i++){
    if(stats.at<int>(i,cv::CC_STAT_AREA)>=p.min_area){
        result.setTo(255,labels==i);
    }
}

    return result;
     
}
