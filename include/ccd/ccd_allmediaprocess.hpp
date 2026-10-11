#pragma once

#include<opencv2/core.hpp>
#include"ccd/ccd_preprocess.hpp"
#include"ccd/ccd_crop.hpp"

//整合单帧ccd处理的功能，输出存储所有未拼接物体的vector
std::vector<CcdObjectCrop> ccd_allmediaprocess(
    const std::vector<CcdPair>& CcdPairs,
    const CcdBackground& b,
    const CcdOptions& p
);