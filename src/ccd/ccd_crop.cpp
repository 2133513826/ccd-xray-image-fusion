#include "ccd/ccd_crop.hpp"

#include<opencv2/imgproc.hpp>

#include<stdexcept>


namespace{
cv::Rect makeSafeCropBox(
    const cv::Rect& object_box,
    int padding_pixels,
    const cv::Size& image_size
){
     if (padding_pixels < 0) {
        throw std::invalid_argument(
            "padding_pixels cannot be negative"
        );
    }


    // 向四周扩大指定像素
    cv::Rect expanded_box(
        object_box.x - padding_pixels,
        object_box.y - padding_pixels,
        object_box.width + 2 * padding_pixels,
        object_box.height + 2 * padding_pixels
    );

    // 整张图像的合法范围
    cv::Rect image_box(
        0,
        0,
        image_size.width,
        image_size.height
    );
    // 取交集   
    return expanded_box & image_box;

}
}


// 裁剪
std::vector<CcdObjectCrop> cropCcdObjects(
    const CcdPair& pair,
    const CcdFrameAnalysis& frame,
    const cv::Mat& mask,
    int padding_pixels
) {
    // 检查深度图和强度图,mask，labels
    if (pair.depth_image.empty()) {
         throw std::invalid_argument(
            "depth image is empty"
        );
    }

    if (pair.intensity_image.empty()) {
        throw std::invalid_argument(
            "intensity image is empty"
        );
    }

    if (pair.depth_image.size() != pair.intensity_image.size()) {
        throw std::invalid_argument(
            "depth and intensity must have the same size"
        );
    }

    if (mask.empty()) {
        throw std::invalid_argument(
            "mask is empty"
        );
    }

    if (mask.type() != CV_8UC1) {
        throw std::invalid_argument(
            "mask must have type CV_8UC1"
        );
    }

    if (mask.size() != pair.depth_image.size()) {
        throw std::invalid_argument(
            "mask and images must have the same size"
        );
    }
     if (frame.labels.empty()) {
        throw std::invalid_argument(
            "component labels are empty"
        );
    }

    if (frame.labels.type() != CV_32SC1) {
        throw std::invalid_argument(
            "component labels must have type CV_32SC1"
        );
    }

    if (frame.labels.size() != pair.depth_image.size()) {
        throw std::invalid_argument(
            "component labels and images must have the same size"
        );
    }

    std::vector<CcdObjectCrop> results;
    results.reserve(frame.objects.size());

    //遍历每一个连通区域

    for (const auto& object : frame.objects) {
        // 只允许使用有效的前景标签
        if (object.label <= 0) {
            continue;
        }
    

    const cv::Rect crop_box = makeSafeCropBox(
            object.bounding_rect,padding_pixels,pair.depth_image.size()
        );
    if (crop_box.empty()) {
            continue;
        }

    CcdObjectCrop crop;

    crop.object_info=object;

    
    crop.crop_box = crop_box;

    crop.depth = pair.depth_image(crop_box).clone();
    crop.intensity = pair.intensity_image(crop_box).clone();

    const cv::Mat labels_crop=frame.labels(crop_box);
    cv::Mat mask_crop =cv::Mat::zeros(labels_crop.size(),CV_8UC1);

    mask_crop.setTo(255,labels_crop == object.label);

     const cv::Mat mask_roi = mask(crop_box);

    cv::bitwise_and(mask_roi,mask_crop,crop.mask);

    results.push_back(std::move(crop));

    }

    return results;

}