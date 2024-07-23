//#pragma once
//
//#define RET_OK nullptr
//
//#ifdef _WIN32
//#include <Windows.h>
//#include <direct.h>
//#include <io.h>
//#endif
//
//#include <string>
//#include <vector>
//#include <cstdio>
//#include <opencv2/opencv.hpp>
//#include "onnxruntime_cxx_api.h"
//
//#ifdef USE_CUDA
//#include <cuda_fp16.h>
//#endif
//
//typedef struct _DL_INIT_PARAM
//{
//    std::string modelPath;
//    std::vector<int> imgSize = { 640, 640 };
//    float rectConfidenceThreshold = 0.6;
//    float iouThreshold = 0.5;
//    bool cudaEnable = false;
//    int logSeverityLevel = 3;
//    int intraOpNumThreads = 1;
//} DL_INIT_PARAM;
//
//typedef struct _DL_RESULT
//{
//    int classId;
//    float confidence;
//    cv::Rect box;
//} DL_RESULT;
//
//class YOLO_V8
//{
//public:
//    YOLO_V8();
//    ~YOLO_V8();
//
//public:
//    char* CreateSession(DL_INIT_PARAM& iParams);
//    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);
//
//private:
//    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
//    char* TensorProcess(cv::Mat& iImg, std::vector<float>& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult);
//
//    Ort::Env env;
//    Ort::Session* session;
//    Ort::RunOptions options;
//    std::vector<const char*> inputNodeNames;
//    std::vector<const char*> outputNodeNames;
//    std::vector<int> imgSize;
//    float rectConfidenceThreshold;
//    float iouThreshold;
//    bool cudaEnable;
//};
//
//YOLO_V8::YOLO_V8() : session(nullptr), cudaEnable(false) {}
//
//YOLO_V8::~YOLO_V8()
//{
//    delete session;
//}
//
//char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
//{
//    if (iImg.channels() == 3)
//    {
//        oImg = iImg.clone();
//        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
//    }
//    else
//    {
//        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
//    }
//
//    if (iImg.cols >= iImg.rows)
//    {
//        float resizeScale = iImg.cols / (float)iImgSize[0];
//        cv::resize(oImg, oImg, cv::Size(iImgSize[0], int(iImg.rows / resizeScale)));
//    }
//    else
//    {
//        float resizeScale = iImg.rows / (float)iImgSize[1];
//        cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScale), iImgSize[1]));
//    }
//
//    cv::Mat tempImg = cv::Mat::zeros(iImgSize[0], iImgSize[1], CV_8UC3);
//    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
//    oImg = tempImg;
//
//    return RET_OK;
//}
//
//char* YOLO_V8::TensorProcess(cv::Mat& iImg, std::vector<float>& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult)
//{
//    inputNodeDims = { 1, iImg.channels(), iImg.rows, iImg.cols };
//    size_t inputTensorSize = iImg.total() * iImg.elemSize();
//    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, blob.data(), inputTensorSize, inputNodeDims.data(), inputNodeDims.size());
//
//    std::vector<int64_t> outputNodeDims;
//    std::vector<Ort::Value> outputTensors;
//
//    try
//    {
//        outputTensors = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);
//    }
//    catch (const std::exception& e)
//    {
//        std::cerr << "Error during inference: " << e.what() << std::endl;
//        return "[YOLO_V8]:RunSession failed.";
//    }
//
//    float* out = outputTensors[0].GetTensorMutableData<float>();
//    outputNodeDims = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
//
//    float ratioh = static_cast<float>(iImg.rows) / imgSize[1];
//    float ratiow = static_cast<float>(iImg.cols) / imgSize[0];
//
//    int numResults = outputNodeDims[1];
//    for (int i = 0; i < numResults; i++)
//    {
//        float confidence = out[4];
//        if (confidence >= rectConfidenceThreshold)
//        {
//            float classScore = out[5];
//            int classId = static_cast<int>(classScore);
//
//            float x_center = out[0];
//            float y_center = out[1];
//            float width = out[2];
//            float height = out[3];
//
//            cv::Rect rect;
//            rect.x = static_cast<int>((x_center - width / 2) * ratiow);
//            rect.y = static_cast<int>((y_center - height / 2) * ratioh);
//            rect.width = static_cast<int>(width * ratiow);
//            rect.height = static_cast<int>(height * ratioh);
//
//            DL_RESULT res;
//            res.classId = classId;
//            res.confidence = confidence;
//            res.box = rect;
//            oResult.push_back(res);
//        }
//        out += 6; // Move to the next set of results
//    }
//
//    return RET_OK;
//}
//
//char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams)
//{
//    try
//    {
//        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
//        iouThreshold = iParams.iouThreshold;
//        imgSize = iParams.imgSize;
//        cudaEnable = iParams.cudaEnable;
//        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
//        Ort::SessionOptions sessionOption;
//        if (cudaEnable)
//        {
//            OrtCUDAProviderOptions cudaOption;
//            cudaOption.device_id = 0;
//            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
//        }
//        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
//        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
//        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);
//
//#ifdef _WIN32
//        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
//        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
//        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
//        wide_cstr[ModelPathSize] = L'\0';
//        const wchar_t* modelPath = wide_cstr;
//#else
//        const char* modelPath = iParams.modelPath.c_str();
//#endif // _WIN32
//
//        session = new Ort::Session(env, modelPath, sessionOption);
//        Ort::AllocatorWithDefaultOptions allocator;
//        size_t inputNodesNum = session->GetInputCount();
//        for (size_t i = 0; i < inputNodesNum; i++)
//        {
//            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
//            char* temp_buf = new char[50];
//            strcpy(temp_buf, input_node_name.get());
//            inputNodeNames.push_back(temp_buf);
//        }
//        size_t OutputNodesNum = session->GetOutputCount();
//        for (size_t i = 0; i < OutputNodesNum; i++)
//        {
//            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
//            char* temp_buf = new char[10];
//            strcpy(temp_buf, output_node_name.get());
//            outputNodeNames.push_back(temp_buf);
//        }
//        options = Ort::RunOptions{ nullptr };
//        return RET_OK;
//    }
//    catch (const std::exception& e)
//    {
//        std::cerr << "[YOLO_V8]:Create session failed. " << e.what() << std::endl;
//        return "[YOLO_V8]:Create session failed.";
//    }
//}
//
//char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult)
//{
//    cv::Mat resizedImg;
//    PreProcess(iImg, imgSize, resizedImg);
//
//    std::vector<float> blob(resizedImg.total() * resizedImg.channels());
//    int channels = resizedImg.channels();
//    int imgHeight = resizedImg.rows;
//    int imgWidth = resizedImg.cols;
//
//    for (int c = 0; c < channels; c++)
//    {
//        for (int h = 0; h < imgHeight; h++)
//        {
//            for (int w = 0; w < imgWidth; w++)
//            {
//                blob[c * imgWidth * imgHeight + h * imgWidth + w] = resizedImg.at<cv::Vec3b>(h, w)[c] / 255.0f;
//            }
//        }
//    }
//
//    std::vector<int64_t> inputNodeDims;
//    return TensorProcess(resizedImg, blob, inputNodeDims, oResult);
//}
//
//void Detector(YOLO_V8* p)
//{
//    std::vector<cv::String> images;
//    cv::glob("D:\\OPENCV\\Images\\13.jpg", images);
//
//    for (const auto& imagePath : images)
//    {
//        std::vector<DL_RESULT> result;
//        cv::Mat image = cv::imread(imagePath);
//
//        p->RunSession(image, result);
//
//        for (const auto& det : result)
//        {
//            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
//            std::string label = std::to_string(det.classId) + ":" + std::to_string(det.confidence);
//            cv::putText(image, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//        }
//
//        cv::imshow("Detection", image);
//        cv::waitKey(0);
//    }
//}
//
//int main()
//{
//    YOLO_V8* p = new YOLO_V8();
//    DL_INIT_PARAM initParams;
//    initParams.modelPath = "D:\\OPENCV\\yolov8n\\yolov8m.onnx";
//    initParams.imgSize = { 640, 640 };
//    initParams.rectConfidenceThreshold = 0.6;
//    initParams.iouThreshold = 0.5;
//    initParams.cudaEnable = true;
//
//    char* ret = p->CreateSession(initParams);
//    if (ret != RET_OK)
//    {
//        std::cerr << "Failed to create session: " << ret << std::endl;
//        delete p;
//        return -1;
//    }
//
//    Detector(p);
//
//    delete p;
//    return 0;
//}




//
//
//#pragma once
//
//#define RET_OK "OK"
//
//#ifdef _WIN32
//#include <Windows.h>
//#include <direct.h>
//#include <io.h>
//#endif
//
//#include <string>
//#include <vector>
//#include <cstdio>
//#include <opencv2/opencv.hpp>
//#include "onnxruntime_cxx_api.h"
//
//#ifdef USE_CUDA
//#include <cuda_fp16.h>
//#endif
//
//typedef struct _DL_INIT_PARAM
//{
//    std::string modelPath;
//    std::vector<int> imgSize = { 640, 640 };
//    float rectConfidenceThreshold = 0.6;
//    float iouThreshold = 0.5;
//    bool cudaEnable = false;
//    int logSeverityLevel = 3;
//    int intraOpNumThreads = 1;
//} DL_INIT_PARAM;
//
//typedef struct _DL_RESULT
//{
//    int classId;
//    float confidence;
//    cv::Rect box;
//} DL_RESULT;
//
//class YOLO_V8
//{
//public:
//    YOLO_V8();
//    ~YOLO_V8();
//
//public:
//    std::string CreateSession(const DL_INIT_PARAM& iParams);
//    std::string RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);
//
//private:
//    std::string PreProcess(cv::Mat& iImg, const std::vector<int>& iImgSize, cv::Mat& oImg);
//    std::string TensorProcess(cv::Mat& iImg, std::vector<float>& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult);
//
//    Ort::Env env;
//    Ort::Session* session;
//    Ort::RunOptions options;
//    std::vector<const char*> inputNodeNames;
//    std::vector<const char*> outputNodeNames;
//    std::vector<int> imgSize;
//    float rectConfidenceThreshold;
//    float iouThreshold;
//    bool cudaEnable;
//};
//
//YOLO_V8::YOLO_V8() : session(nullptr), cudaEnable(false) {}
//
//YOLO_V8::~YOLO_V8()
//{
//    delete session;
//}
//
//std::string YOLO_V8::PreProcess(cv::Mat& iImg, const std::vector<int>& iImgSize, cv::Mat& oImg)
//{
//    if (iImg.channels() == 3)
//    {
//        oImg = iImg.clone();
//        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
//    }
//    else
//    {
//        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
//    }
//
//    if (iImg.cols >= iImg.rows)
//    {
//        float resizeScale = iImg.cols / static_cast<float>(iImgSize[0]);
//        cv::resize(oImg, oImg, cv::Size(iImgSize[0], static_cast<int>(iImg.rows / resizeScale)));
//    }
//    else
//    {
//        float resizeScale = iImg.rows / static_cast<float>(iImgSize[1]);
//        cv::resize(oImg, oImg, cv::Size(static_cast<int>(iImg.cols / resizeScale), iImgSize[1]));
//    }
//
//    cv::Mat tempImg = cv::Mat::zeros(iImgSize[0], iImgSize[1], CV_8UC3);
//    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
//    oImg = tempImg;
//
//    return RET_OK;
//}
//
//std::string YOLO_V8::TensorProcess(cv::Mat& iImg, std::vector<float>& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult)
//{
//    inputNodeDims = { 1, iImg.channels(), iImg.rows, iImg.cols };
//    size_t inputTensorSize = iImg.total() * iImg.elemSize();
//    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, blob.data(), inputTensorSize, inputNodeDims.data(), inputNodeDims.size());
//
//    std::vector<int64_t> outputNodeDims;
//    std::vector<Ort::Value> outputTensors;
//
//    try
//    {
//        outputTensors = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);
//    }
//    catch (const std::exception& e)
//    {
//        return std::string("[YOLO_V8]:RunSession failed. ") + e.what();
//    }
//
//    float* out = outputTensors[0].GetTensorMutableData<float>();
//    outputNodeDims = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
//
//    float ratioh = static_cast<float>(iImg.rows) / imgSize[1];
//    float ratiow = static_cast<float>(iImg.cols) / imgSize[0];
//
//    int numResults = outputNodeDims[1];
//    for (int i = 0; i < numResults; i++)
//    {
//        float confidence = out[4];
//        if (confidence >= rectConfidenceThreshold)
//        {
//            float classScore = out[5];
//            int classId = static_cast<int>(classScore);
//
//            float x_center = out[0];
//            float y_center = out[1];
//            float width = out[2];
//            float height = out[3];
//
//            cv::Rect rect;
//            rect.x = static_cast<int>((x_center - width / 2) * ratiow);
//            rect.y = static_cast<int>((y_center - height / 2) * ratioh);
//            rect.width = static_cast<int>(width * ratiow);
//            rect.height = static_cast<int>(height * ratioh);
//
//            DL_RESULT res;
//            res.classId = classId;
//            res.confidence = confidence;
//            res.box = rect;
//            oResult.push_back(res);
//        }
//        out += 6; // Move to the next set of results
//    }
//
//    return RET_OK;
//}
//
//std::string YOLO_V8::CreateSession(const DL_INIT_PARAM& iParams)
//{
//    try
//    {
//        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
//        iouThreshold = iParams.iouThreshold;
//        imgSize = iParams.imgSize;
//        cudaEnable = iParams.cudaEnable;
//        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
//        Ort::SessionOptions sessionOption;
//        if (cudaEnable)
//        {
//            OrtCUDAProviderOptions cudaOption;
//            cudaOption.device_id = 0;
//            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
//        }
//        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
//        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
//        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);
//
//#ifdef _WIN32
//        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
//        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
//        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
//        wide_cstr[ModelPathSize] = L'\0';
//        const wchar_t* modelPath = wide_cstr;
//#else
//        const char* modelPath = iParams.modelPath.c_str();
//#endif // _WIN32
//
//        session = new Ort::Session(env, modelPath, sessionOption);
//        Ort::AllocatorWithDefaultOptions allocator;
//        size_t inputNodesNum = session->GetInputCount();
//        for (size_t i = 0; i < inputNodesNum; i++)
//        {
//            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
//            char* temp_buf = new char[50];
//            strcpy_s(temp_buf, 50, input_node_name.get());
//            inputNodeNames.push_back(temp_buf);
//        }
//        size_t OutputNodesNum = session->GetOutputCount();
//        for (size_t i = 0; i < OutputNodesNum; i++)
//        {
//            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
//            char* temp_buf = new char[10];
//            strcpy_s(temp_buf, 10, output_node_name.get());
//            outputNodeNames.push_back(temp_buf);
//        }
//        options = Ort::RunOptions{ nullptr };
//        return RET_OK;
//    }
//    catch (const std::exception& e)
//    {
//        return std::string("[YOLO_V8]:Create session failed. ") + e.what();
//    }
//}
//
//std::string YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult)
//{
//    cv::Mat resizedImg;
//    std::string preProcessResult = PreProcess(iImg, imgSize, resizedImg);
//    if (preProcessResult != RET_OK)
//    {
//        return preProcessResult;
//    }
//
//    std::vector<float> blob(resizedImg.total() * resizedImg.channels());
//    int channels = resizedImg.channels();
//    int imgHeight = resizedImg.rows;
//    int imgWidth = resizedImg.cols;
//
//    for (int c = 0; c < channels; c++)
//    {
//        for (int h = 0; h < imgHeight; h++)
//        {
//            for (int w = 0; w < imgWidth; w++)
//            {
//                blob[c * imgHeight * imgWidth + h * imgWidth + w] = resizedImg.at<cv::Vec3b>(h, w)[c] / 255.0f;
//            }
//        }
//    }
//
//    std::vector<int64_t> inputNodeDims;
//    std::string tensorProcessResult = TensorProcess(resizedImg, blob, inputNodeDims, oResult);
//    if (tensorProcessResult != RET_OK)
//    {
//        return tensorProcessResult;
//    }
//
//    return RET_OK;
//}
//
//void Detector(YOLO_V8* p)
//{
//    std::vector<cv::String> images;
//    cv::glob("D:\\OPENCV\\Images\\*.jpg", images);
//
//    for (const auto& imagePath : images)
//    {
//        std::vector<DL_RESULT> result;
//        cv::Mat image = cv::imread(imagePath);
//
//        std::string runSessionResult = p->RunSession(image, result);
//        if (runSessionResult != RET_OK)
//        {
//            std::cerr << "Error during RunSession: " << runSessionResult << std::endl;
//            continue;
//        }
//
//        for (const auto& det : result)
//        {
//            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
//            std::string label = std::to_string(det.classId) + ":" + std::to_string(det.confidence);
//            cv::putText(image, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//        }
//
//        cv::imshow("Detection", image);
//        cv::waitKey(0);
//    }
//}
//
//int main()
//{
//    YOLO_V8* p = new YOLO_V8();
//    DL_INIT_PARAM initParams;
//    initParams.modelPath = "D:\\OPENCV\\yolov8n\\yolov8m.onnx";
//    initParams.imgSize = { 640, 640 };
//    initParams.rectConfidenceThreshold = 0.6;
//    initParams.iouThreshold = 0.5;
//    initParams.cudaEnable = true;
//
//    std::string createSessionResult = p->CreateSession(initParams);
//    if (createSessionResult != RET_OK)
//    {
//        std::cerr << "Failed to create session: " << createSessionResult << std::endl;
//        delete p;
//        return -1;
//    }
//
//    Detector(p);
//
//    delete p;
//    return 0;
//}
//
