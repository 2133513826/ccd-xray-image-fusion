
#include<opencv2/core.hpp>
#include"ccd/ccd_preprocess.hpp"
#include"ccd/ccd_crop.hpp"
#include "common/progress.hpp"

//整合单帧ccd处理的功能，输出存储所有未拼接物体的vector
std::vector<CcdObjectCrop> ccd_allmediaprocess(
    const std::vector<CcdPair>& CcdPairs,
    const CcdBackground& b,
    const CcdOptions& p
)
{
    std::vector<CcdObjectCrop> CcdObjectCrops;

    //遍历每一帧ccd，进行单帧ccd处理

    std::size_t file_index=0;
    std::int64_t timestamp_ms=0;
    int p_e=p.edge_margin;
    int p_pdp=p.padding_pixels;
    showProgress("CCD frames", 0, CcdPairs.size());

    for (auto& CcdPair : CcdPairs)
    {
        file_index++;
        // id格式：20251127155634931，前14位是日期时间，末3位是毫秒。
std::tm time{};
time.tm_isdst = -1;
std::istringstream input(CcdPair.id.substr(0, 14));
input >> std::get_time(&time, "%Y%m%d%H%M%S");
if (input.fail()) throw std::runtime_error("Invalid CCD timestamp");
const std::time_t seconds = std::mktime(&time);
if (seconds == -1) throw std::runtime_error("Invalid CCD date");
timestamp_ms = static_cast<std::int64_t>(seconds) * 1000
             + std::stoi(CcdPair.id.substr(14, 3));
        //timestamp_ms=static_cast<std::int64_t>(CcdPair.id);
        //生成掩膜
        cv::Mat TemMask=detectCcd(CcdPair.depth_image,CcdPair.intensity_image,b,p);
        //存储信息
        
        CcdFrameAnalysis CcdFrameAnalysis=analyzeCcdMask(file_index,timestamp_ms,TemMask,p_e);

        std::vector<CcdObjectCrop> CurrentCcdObjectCrop=cropCcdObjects(CcdPair,CcdFrameAnalysis,TemMask,p_pdp);

        //存储进总表内

        CcdObjectCrops.insert
        (
            CcdObjectCrops.end(),
            CurrentCcdObjectCrop.begin(),
            CurrentCcdObjectCrop.end()
        );
        showProgress("CCD frames", file_index, CcdPairs.size());

    }

    return CcdObjectCrops;
}


