#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

struct BBox 
{
    float x1, y1, x2, y2;
    float score;
    std::vector<float> offsets;    // 4 个偏移量 (tx1, ty1, tx2, ty2)
    std::vector<float> landmarks;  // 10 个关键点坐标 (x1..x5, y1..y5)
};

class MTCNN 
{
public:
    /**
     * @param pnet_path P-Net ONNX 模型路径（宽字符，Windows 下可用 L"..."）
     * @param rnet_path R-Net ONNX 模型路径
     * @param onet_path O-Net ONNX 模型路径
     */
    MTCNN(const wchar_t* pnet_path, const wchar_t* rnet_path, const wchar_t* onet_path);

    /**
     * 执行人脸检测
     * @param image  输入图像（RGB 格式，CV_8UC3）
     * @param min_face_size  最小人脸尺寸（像素）
     * @param thresholds     三个阶段的人脸概率阈值
     * @param nms_thresholds 三个阶段的 NMS 阈值
     * @return 检测到的人脸边界框及关键点
     */
    std::vector<BBox> Detect(const cv::Mat& image,
        float min_face_size = 20.0f,
        const std::vector<float>& thresholds = { 0.6f, 0.7f, 0.8f },
        const std::vector<float>& nms_thresholds = { 0.7f, 0.7f, 0.7f });

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> pnet_;
    std::unique_ptr<Ort::Session> rnet_;
    std::unique_ptr<Ort::Session> onet_;

    cv::Mat img_;     // 原始图像副本
    int img_w_, img_h_;

    // ---------- 工具函数 ----------
    static float IoU(const BBox& a, const BBox& b);
    static float IoM(const BBox& a, const BBox& b);
    static void NMS(std::vector<BBox>& boxes, float threshold, const std::string& mode = "union");
    static void CalibrateBox(BBox& box);
    static void ConvertToSquare(BBox& box);
    static void ClipBox(BBox& box, int img_w, int img_h);

    // ---------- 核心：无畸变图像块提取（对齐 Python 的 get_image_boxes + correct_bboxes） ----------
    cv::Mat GetImagePatch(const cv::Mat& img, const BBox& box, int patch_size);

    // ---------- 三个阶段 ----------
    std::vector<BBox> Stage1(float min_face_size, float threshold, float nms_threshold);
    std::vector<BBox> Stage2(const std::vector<BBox>& in_boxes, float threshold, float nms_threshold);
    std::vector<BBox> Stage3(const std::vector<BBox>& in_boxes, float threshold, float nms_threshold);
};