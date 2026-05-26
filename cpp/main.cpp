#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "mtcnn.h"



void detect_img()
{
    // 初始化检测器（请根据实际路径修改）
    MTCNN detector(L"./onnx/pnet.onnx", L"./onnx/rnet.onnx", L"./onnx/onet.onnx");

    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return ;
    }
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    std::vector<BBox> faces = detector.Detect(rgb, 20.0f, { 0.6f, 0.7f, 0.8f }, { 0.7f, 0.7f, 0.7f });
    // 绘制结果
    for (const auto& face : faces) 
    {
        cv::rectangle(image, cv::Point((int)face.x1, (int)face.y1), cv::Point((int)face.x2, (int)face.y2), cv::Scalar(0, 255, 0), 2);
        for (int k = 0; k < 5; ++k) 
        {
            cv::circle(image, cv::Point((int)face.landmarks[k], (int)face.landmarks[k + 5]), 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    cv::imwrite("./output.jpg", image);
    std::cout << "Detected " << faces.size() << " faces." << std::endl;
}

void detect_camera()
{
    // 初始化检测器（请根据实际路径修改）
    MTCNN detector(L"./onnx/pnet.onnx", L"./onnx/rnet.onnx", L"./onnx/onet.onnx");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return ;
    }

    cv::Mat frame, rgb;
    std::vector<BBox> faces;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        // 执行检测
        faces = detector.Detect(rgb, 40.0f, { 0.6f, 0.7f, 0.8f }, { 0.7f, 0.7f, 0.7f });

        // 绘制结果
        for (const auto& face : faces) {
            cv::rectangle(frame, cv::Point((int)face.x1, (int)face.y1),
                cv::Point((int)face.x2, (int)face.y2), cv::Scalar(0, 255, 0), 2);
            for (int k = 0; k < 5; ++k) {
                cv::circle(frame, cv::Point((int)face.landmarks[k], (int)face.landmarks[k + 5]),
                    2, cv::Scalar(255, 0, 0), -1);
            }
        }

        cv::imshow("MTCNN", frame);
        if (cv::waitKey(1) == 27) break; // ESC 退出
    }

    cap.release();
    cv::destroyAllWindows();
}


int main() 
{
    //detect_img();
    detect_camera();
    return 0;
}