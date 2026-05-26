#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

#include "mtcnn.h"


float MTCNN::IoU(const BBox& a, const BBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2) return 0.0f;
    float inter_area = (inter_x2 - inter_x1 + 1.0f) * (inter_y2 - inter_y1 + 1.0f);
    float area_a = (a.x2 - a.x1 + 1.0f) * (a.y2 - a.y1 + 1.0f);
    float area_b = (b.x2 - b.x1 + 1.0f) * (b.y2 - b.y1 + 1.0f);
    return inter_area / (area_a + area_b - inter_area);
}

float MTCNN::IoM(const BBox& a, const BBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2) return 0.0f;
    float inter_area = (inter_x2 - inter_x1 + 1.0f) * (inter_y2 - inter_y1 + 1.0f);
    float area_a = (a.x2 - a.x1 + 1.0f) * (a.y2 - a.y1 + 1.0f);
    float area_b = (b.x2 - b.x1 + 1.0f) * (b.y2 - b.y1 + 1.0f);
    return inter_area / std::min(area_a, area_b);
}

void MTCNN::NMS(std::vector<BBox>& boxes, float threshold, const std::string& mode) {
    if (boxes.empty()) return;
    // 按得分降序
    std::sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) {
        return a.score > b.score;
        });
    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            float overlap = (mode == "min") ? IoM(boxes[i], boxes[j]) : IoU(boxes[i], boxes[j]);
            if (overlap > threshold) {
                suppressed[j] = true;
            }
        }
    }
    std::vector<BBox> keep;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!suppressed[i]) keep.push_back(boxes[i]);
    }
    boxes = keep;
}

void MTCNN::CalibrateBox(BBox& box) {
    float w = box.x2 - box.x1 + 1.0f;
    float h = box.y2 - box.y1 + 1.0f;
    box.x1 += w * box.offsets[0];
    box.y1 += h * box.offsets[1];
    box.x2 += w * box.offsets[2];
    box.y2 += h * box.offsets[3];
}

void MTCNN::ConvertToSquare(BBox& box) {
    float w = box.x2 - box.x1 + 1.0f;
    float h = box.y2 - box.y1 + 1.0f;
    float max_side = std::max(w, h);
    float cx = box.x1 + w * 0.5f;
    float cy = box.y1 + h * 0.5f;
    box.x1 = cx - max_side * 0.5f;
    box.y1 = cy - max_side * 0.5f;
    box.x2 = box.x1 + max_side - 1.0f;
    box.y2 = box.y1 + max_side - 1.0f;
}

void MTCNN::ClipBox(BBox& box, int img_w, int img_h) {
    box.x1 = std::max(0.0f, std::min(box.x1, (float)img_w - 1.0f));
    box.y1 = std::max(0.0f, std::min(box.y1, (float)img_h - 1.0f));
    box.x2 = std::max(0.0f, std::min(box.x2, (float)img_w - 1.0f));
    box.y2 = std::max(0.0f, std::min(box.y2, (float)img_h - 1.0f));
}

// ---------- 关键修复：对齐 Python 的 get_image_boxes ----------
cv::Mat MTCNN::GetImagePatch(const cv::Mat& img, const BBox& box, int patch_size) {
    // 计算整数坐标与宽高
    int x1 = static_cast<int>(box.x1);
    int y1 = static_cast<int>(box.y1);
    int x2 = static_cast<int>(box.x2);
    int y2 = static_cast<int>(box.y2);
    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;

    // 初始偏移量（裁剪画布内的有效区域起点）
    int dx = 0, dy = 0;
    int edx = w - 1, edy = h - 1;
    // 原图上的有效裁剪坐标
    int x = x1, y = y1, ex = x2, ey = y2;

    // 修正右边界溢出
    if (ex > img.cols - 1) {
        edx = w + img.cols - 2 - ex;
        ex = img.cols - 1;
    }
    // 修正下边界溢出
    if (ey > img.rows - 1) {
        edy = h + img.rows - 2 - ey;
        ey = img.rows - 1;
    }
    // 修正左边界溢出
    if (x < 0) {
        dx = -x;
        x = 0;
    }
    // 修正上边界溢出
    if (y < 0) {
        dy = -y;
        y = 0;
    }

    // 创建全零画布（黑色填充）
    cv::Mat patch(h, w, CV_8UC3, cv::Scalar(0, 0, 0));

    // 如果原图有效区域非空，拷贝过去
    if (ex >= x && ey >= y) {
        cv::Rect roi_orig(x, y, ex - x + 1, ey - y + 1);
        cv::Rect roi_patch(dx, dy, edx - dx + 1, edy - dy + 1);
        img(roi_orig).copyTo(patch(roi_patch));
    }

    // 缩放到指定尺寸
    cv::Mat resized;
    cv::resize(patch, resized, cv::Size(patch_size, patch_size), 0, 0, cv::INTER_LINEAR);
    return resized;
}

// ==================== 构造函数 ====================
MTCNN::MTCNN(const wchar_t* pnet_path, const wchar_t* rnet_path, const wchar_t* onet_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "MTCNN") {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // 如果需要 CUDA，取消下面一行的注释
    // OrtSessionOptionsAppendExecutionProvider_CUDA(opts, 0);
    pnet_ = std::make_unique<Ort::Session>(env_, pnet_path, opts);
    rnet_ = std::make_unique<Ort::Session>(env_, rnet_path, opts);
    onet_ = std::make_unique<Ort::Session>(env_, onet_path, opts);
}

// ==================== Stage 1: P-Net ====================
std::vector<BBox> MTCNN::Stage1(float min_face_size, float threshold, float nms_threshold) {
    const float factor = 0.707f;       // sqrt(0.5)
    const int min_det_size = 12;
    float m = (float)min_det_size / min_face_size;
    int min_len = static_cast<int>(std::min(img_w_, img_h_) * m);

    // 构建图像金字塔尺度
    std::vector<float> scales;
    while (min_len > min_det_size) {
        scales.push_back(m * std::pow(factor, (int)scales.size()));
        min_len = static_cast<int>(min_len * factor);
    }

    std::vector<BBox> total_boxes;
    for (float scale : scales) {
        int sw = static_cast<int>(std::ceil(img_w_ * scale));
        int sh = static_cast<int>(std::ceil(img_h_ * scale));
        cv::Mat resized;
        cv::resize(img_, resized, cv::Size(sw, sh));
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3);
        // 预处理：(img - 127.5) * 0.0078125
        float_img = (float_img - 127.5f) * 0.0078125f;

        // 转为 CHW 并添加 batch 维度
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);
        std::vector<float> input_data(1 * 3 * sh * sw);
        for (int c = 0; c < 3; ++c) {
            memcpy(input_data.data() + c * sh * sw, channels[c].data, sh * sw * sizeof(float));
        }

        std::vector<int64_t> input_shape = { 1, 3, sh, sw };
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = { "input" };
        const char* output_names[] = { "bbox_offsets", "face_probs" };
        auto outputs = pnet_->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 2);

        const float* offsets_data = outputs[0].GetTensorData<float>(); // [1,4,h',w']
        const float* probs_data = outputs[1].GetTensorData<float>(); // [1,2,h',w']
        auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int h_out = static_cast<int>(out_shape[2]);
        int w_out = static_cast<int>(out_shape[3]);

        std::vector<BBox> scale_boxes;
        for (int i = 0; i < h_out; ++i) {
            for (int j = 0; j < w_out; ++j) {
                // 人脸概率通道索引 1
                float score = probs_data[1 * h_out * w_out + i * w_out + j];
                if (score < threshold) continue;

                // 计算原图坐标（公式与 Python 一致）
                float x1 = (2.0f * j + 1.0f) / scale;
                float y1 = (2.0f * i + 1.0f) / scale;
                float x2 = (2.0f * j + 1.0f + 12.0f) / scale;
                float y2 = (2.0f * i + 1.0f + 12.0f) / scale;

                BBox box;
                box.x1 = x1; box.y1 = y1; box.x2 = x2; box.y2 = y2;
                box.score = score;
                box.offsets.resize(4);
                for (int k = 0; k < 4; ++k) {
                    box.offsets[k] = offsets_data[k * h_out * w_out + i * w_out + j];
                }
                scale_boxes.push_back(box);
            }
        }

        // 关键修复：尺度内 NMS 使用固定阈值 0.5（对齐 Python）
        NMS(scale_boxes, 0.5f);
        total_boxes.insert(total_boxes.end(), scale_boxes.begin(), scale_boxes.end());
    }

    // 全局 NMS
    NMS(total_boxes, nms_threshold);

    // 校准、转正方形、取整、裁剪
    for (auto& box : total_boxes) {
        CalibrateBox(box);
        ConvertToSquare(box);
        box.x1 = std::round(box.x1);
        box.y1 = std::round(box.y1);
        box.x2 = std::round(box.x2);
        box.y2 = std::round(box.y2);
        ClipBox(box, img_w_, img_h_);
    }
    return total_boxes;
}

// ==================== Stage 2: R-Net ====================
std::vector<BBox> MTCNN::Stage2(const std::vector<BBox>& in_boxes, float threshold, float nms_threshold) {
    if (in_boxes.empty()) return {};
    int num = static_cast<int>(in_boxes.size());
    std::vector<float> input_data(num * 3 * 24 * 24);

    for (int n = 0; n < num; ++n) {
        // 使用无畸变图像块提取
        cv::Mat patch = GetImagePatch(img_, in_boxes[n], 24);
        patch.convertTo(patch, CV_32FC3);
        patch = (patch - 127.5f) * 0.0078125f;

        std::vector<cv::Mat> channels(3);
        cv::split(patch, channels);
        for (int c = 0; c < 3; ++c) {
            memcpy(input_data.data() + n * (3 * 24 * 24) + c * 24 * 24,
                channels[c].data, 24 * 24 * sizeof(float));
        }
    }

    std::vector<int64_t> input_shape = { num, 3, 24, 24 };
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = { "input" };
    const char* output_names[] = { "bbox_offsets", "face_probs" };
    auto outputs = rnet_->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 2);

    const float* offsets_data = outputs[0].GetTensorData<float>(); // [num,4]
    const float* probs_data = outputs[1].GetTensorData<float>(); // [num,2]

    std::vector<BBox> out_boxes;
    for (int n = 0; n < num; ++n) {
        float score = probs_data[n * 2 + 1];
        if (score < threshold) continue;
        BBox box = in_boxes[n];
        box.score = score;
        box.offsets = { offsets_data[n * 4], offsets_data[n * 4 + 1], offsets_data[n * 4 + 2], offsets_data[n * 4 + 3] };
        out_boxes.push_back(box);
    }

    NMS(out_boxes, nms_threshold);
    for (auto& box : out_boxes) {
        CalibrateBox(box);
        ConvertToSquare(box);
        box.x1 = std::round(box.x1);
        box.y1 = std::round(box.y1);
        box.x2 = std::round(box.x2);
        box.y2 = std::round(box.y2);
        ClipBox(box, img_w_, img_h_);
    }
    return out_boxes;
}

// ==================== Stage 3: O-Net ====================
std::vector<BBox> MTCNN::Stage3(const std::vector<BBox>& in_boxes, float threshold, float nms_threshold) {
    if (in_boxes.empty()) return {};
    int num = static_cast<int>(in_boxes.size());
    std::vector<float> input_data(num * 3 * 48 * 48);

    for (int n = 0; n < num; ++n) {
        cv::Mat patch = GetImagePatch(img_, in_boxes[n], 48);
        patch.convertTo(patch, CV_32FC3);
        patch = (patch - 127.5f) * 0.0078125f;

        std::vector<cv::Mat> channels(3);
        cv::split(patch, channels);
        for (int c = 0; c < 3; ++c) {
            memcpy(input_data.data() + n * (3 * 48 * 48) + c * 48 * 48,
                channels[c].data, 48 * 48 * sizeof(float));
        }
    }

    std::vector<int64_t> input_shape = { num, 3, 48, 48 };
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = { "input" };
    const char* output_names[] = { "landmarks", "bbox_offsets", "face_probs" };
    auto outputs = onet_->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 3);

    const float* landmarks_data = outputs[0].GetTensorData<float>(); // [num,10]
    const float* offsets_data = outputs[1].GetTensorData<float>(); // [num,4]
    const float* probs_data = outputs[2].GetTensorData<float>(); // [num,2]

    std::vector<BBox> out_boxes;
    for (int n = 0; n < num; ++n) {
        float score = probs_data[n * 2 + 1];
        if (score < threshold) continue;
        BBox box = in_boxes[n];
        box.score = score;
        box.offsets = { offsets_data[n * 4], offsets_data[n * 4 + 1], offsets_data[n * 4 + 2], offsets_data[n * 4 + 3] };
        box.landmarks.resize(10);
        for (int k = 0; k < 10; ++k) {
            box.landmarks[k] = landmarks_data[n * 10 + k];
        }
        out_boxes.push_back(box);
    }

    // 关键点坐标转换：使用校准前的框宽高（对齐 Python）
    for (auto& box : out_boxes) {
        float w0 = box.x2 - box.x1 + 1.0f;
        float h0 = box.y2 - box.y1 + 1.0f;
        for (int k = 0; k < 5; ++k) {
            box.landmarks[k] = box.x1 + w0 * box.landmarks[k];
            box.landmarks[k + 5] = box.y1 + h0 * box.landmarks[k + 5];
        }
        CalibrateBox(box);
        box.x1 = std::round(box.x1);
        box.y1 = std::round(box.y1);
        box.x2 = std::round(box.x2);
        box.y2 = std::round(box.y2);
        ClipBox(box, img_w_, img_h_);
    }

    NMS(out_boxes, nms_threshold, "min");
    return out_boxes;
}

// ==================== 公开检测接口 ====================
std::vector<BBox> MTCNN::Detect(const cv::Mat& image,
    float min_face_size,
    const std::vector<float>& thresholds,
    const std::vector<float>& nms_thresholds) {
    img_ = image.clone();
    img_w_ = img_.cols;
    img_h_ = img_.rows;

    auto candidates = Stage1(min_face_size, thresholds[0], nms_thresholds[0]);
    candidates = Stage2(candidates, thresholds[1], nms_thresholds[1]);
    candidates = Stage3(candidates, thresholds[2], nms_thresholds[2]);
    return candidates;
}