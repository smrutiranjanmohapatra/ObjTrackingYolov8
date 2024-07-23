//#include "inference.h"
//#include <regex>
//
//#define benchmark
//#define min(a,b)            (((a) < (b)) ? (a) : (b))
//YOLO_V8::YOLO_V8() {
//
//}
//
//
//YOLO_V8::~YOLO_V8() {
//    delete session;
//}
//
//#ifdef USE_CUDA
//namespace Ort
//{
//    template<>
//    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
//}
//#endif
//
//
//template<typename T>
//char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
//    int channels = iImg.channels();
//    int imgHeight = iImg.rows;
//    int imgWidth = iImg.cols;
//
//    for (int c = 0; c < channels; c++)
//    {
//        for (int h = 0; h < imgHeight; h++)
//        {
//            for (int w = 0; w < imgWidth; w++)
//            {
//                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
//                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
//            }
//        }
//    }
//    return RET_OK;
//}
//
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
//    switch (modelType)
//    {
//    case YOLO_DETECT_V8:
//    case YOLO_POSE:
//    case YOLO_DETECT_V8_HALF:
//    case YOLO_POSE_V8_HALF://LetterBox
//    {
//        if (iImg.cols >= iImg.rows)
//        {
//            resizeScales = iImg.cols / (float)iImgSize.at(0);
//            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
//        }
//        else
//        {
//            resizeScales = iImg.rows / (float)iImgSize.at(0);
//            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
//        }
//        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
//        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
//        oImg = tempImg;
//        break;
//    }
//    case YOLO_CLS://CenterCrop
//    {
//        int h = iImg.rows;
//        int w = iImg.cols;
//        int m = min(h, w);
//        int top = (h - m) / 2;
//        int left = (w - m) / 2;
//        cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
//        break;
//    }
//    }
//    return RET_OK;
//}
//
//
//char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
//    char* Ret = RET_OK;
//    std::regex pattern("[\u4e00-\u9fa5]");
//    bool result = std::regex_search(iParams.modelPath, pattern);
//    if (result)
//    {
//        Ret = "[YOLO_V8]:Your model path is error.Change your model path without chinese characters.";
//        std::cout << Ret << std::endl;
//        return Ret;
//    }
//    try
//    {
//        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
//        iouThreshold = iParams.iouThreshold;
//        imgSize = iParams.imgSize;
//        modelType = iParams.modelType;
//        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
//        Ort::SessionOptions sessionOption;
//        if (iParams.cudaEnable)
//        {
//            cudaEnable = iParams.cudaEnable;
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
//        WarmUpSession();
//        return RET_OK;
//    }
//    catch (const std::exception& e)
//    {
//        const char* str1 = "[YOLO_V8]:";
//        const char* str2 = e.what();
//        std::string result = std::string(str1) + std::string(str2);
//        char* merged = new char[result.length() + 1];
//        std::strcpy(merged, result.c_str());
//        std::cout << merged << std::endl;
//        delete[] merged;
//        return "[YOLO_V8]:Create session failed.";
//    }
//
//}
//
//
//char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
//#ifdef benchmark
//    clock_t starttime_1 = clock();
//#endif // benchmark
//
//    char* Ret = RET_OK;
//    cv::Mat processedImg;
//    PreProcess(iImg, imgSize, processedImg);
//    if (modelType < 4)
//    {
//        float* blob = new float[processedImg.total() * 3];
//        BlobFromImage(processedImg, blob);
//        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
//        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
//    }
//    else
//    {
//#ifdef USE_CUDA
//        half* blob = new half[processedImg.total() * 3];
//        BlobFromImage(processedImg, blob);
//        std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
//        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
//#endif
//    }
//
//    return Ret;
//}
//
//
//template<typename N>
//char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
//    std::vector<DL_RESULT>& oResult) {
//    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
//        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
//        inputNodeDims.data(), inputNodeDims.size());
//#ifdef benchmark
//    clock_t starttime_2 = clock();
//#endif // benchmark
//    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
//        outputNodeNames.size());
//#ifdef benchmark
//    clock_t starttime_3 = clock();
//#endif // benchmark
//
//    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
//    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
//    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
//    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
//    delete[] blob;
//    switch (modelType)
//    {
//    case YOLO_DETECT_V8:
//    case YOLO_DETECT_V8_HALF:
//    {
//        int strideNum = outputNodeDims[1];//8400
//        int signalResultNum = outputNodeDims[2];//84
//        std::vector<int> class_ids;
//        std::vector<float> confidences;
//        std::vector<cv::Rect> boxes;
//        cv::Mat rawData;
//        if (modelType == YOLO_DETECT_V8)
//        {
//            // FP32
//            rawData = cv::Mat(strideNum, signalResultNum, CV_32F, output);
//        }
//        else
//        {
//            // FP16
//            rawData = cv::Mat(strideNum, signalResultNum, CV_16F, output);
//            rawData.convertTo(rawData, CV_32F);
//        }
//        //Note:
//        //ultralytics add transpose operator to the output of yolov8 model.which make yolov8/v5/v7 has same shape
//        //https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
//        //rowData = rowData.t();
//
//        float* data = (float*)rawData.data;
//
//        for (int i = 0; i < strideNum; ++i)
//        {
//            float* classesScores = data + 4;
//            cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
//            cv::Point class_id;
//            double maxClassScore;
//            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
//            if (maxClassScore > rectConfidenceThreshold)
//            {
//                confidences.push_back(maxClassScore);
//                class_ids.push_back(class_id.x);
//                float x = data[0];
//                float y = data[1];
//                float w = data[2];
//                float h = data[3];
//
//                int left = int((x - 0.5 * w) * resizeScales);
//                int top = int((y - 0.5 * h) * resizeScales);
//
//                int width = int(w * resizeScales);
//                int height = int(h * resizeScales);
//
//                boxes.push_back(cv::Rect(left, top, width, height));
//            }
//            data += signalResultNum;
//        }
//        std::vector<int> nmsResult;
//        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
//        for (int i = 0; i < nmsResult.size(); ++i)
//        {
//            int idx = nmsResult[i];
//            DL_RESULT result;
//            result.classId = class_ids[idx];
//            result.confidence = confidences[idx];
//            result.box = boxes[idx];
//            oResult.push_back(result);
//        }
//
//#ifdef benchmark
//        clock_t starttime_4 = clock();
//        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
//        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
//        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
//        if (cudaEnable)
//        {
//            std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
//        }
//        else
//        {
//            std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
//        }
//#endif // benchmark
//
//        break;
//    }
//    case YOLO_CLS:
//    case YOLO_CLS_HALF:
//    {
//        cv::Mat rawData;
//        if (modelType == YOLO_CLS) {
//            // FP32
//            rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
//        }
//        else {
//            // FP16
//            rawData = cv::Mat(1, this->classes.size(), CV_16F, output);
//            rawData.convertTo(rawData, CV_32F);
//        }
//        float* data = (float*)rawData.data;
//
//        DL_RESULT result;
//        for (int i = 0; i < this->classes.size(); i++)
//        {
//            result.classId = i;
//            result.confidence = data[i];
//            oResult.push_back(result);
//        }
//        break;
//    }
//    default:
//        std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
//    }
//    return RET_OK;
//
//}
//
//
//char* YOLO_V8::WarmUpSession() {
//    clock_t starttime_1 = clock();
//    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
//    cv::Mat processedImg;
//    PreProcess(iImg, imgSize, processedImg);
//    if (modelType < 4)
//    {
//        float* blob = new float[iImg.total() * 3];
//        BlobFromImage(processedImg, blob);
//        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
//        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
//            YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
//        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
//            outputNodeNames.size());
//        delete[] blob;
//        clock_t starttime_4 = clock();
//        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
//        if (cudaEnable)
//        {
//            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
//        }
//    }
//    else
//    {
//#ifdef USE_CUDA
//        half* blob = new half[iImg.total() * 3];
//        BlobFromImage(processedImg, blob);
//        std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
//        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
//        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
//        delete[] blob;
//        clock_t starttime_4 = clock();
//        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
//        if (cudaEnable)
//        {
//            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
//        }
//#endif
//    }
//    return RET_OK;
//}
//
//
//
//
//
//
//
//#include <iostream>
//#include <iomanip>
//#include "inference.h"
//#include <filesystem>
//#include <fstream>
//#include <random>
//
//void Detector(YOLO_V8*& p) {
//    std::filesystem::path current_path = std::filesystem::current_path();
//    std::filesystem::path imgs_path = current_path / "images";
//    for (auto& i : std::filesystem::directory_iterator(imgs_path))
//    {
//        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
//        {
//            std::string img_path = i.path().string();
//            cv::Mat img = cv::imread(img_path);
//            std::vector<DL_RESULT> res;
//            p->RunSession(img, res);
//
//            for (auto& re : res)
//            {
//                cv::RNG rng(cv::getTickCount());
//                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
//
//                cv::rectangle(img, re.box, color, 3);
//
//                float confidence = floor(100 * re.confidence) / 100;
//                std::cout << std::fixed << std::setprecision(2);
//                std::string label = p->classes[re.classId] + " " +
//                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);
//
//                cv::rectangle(
//                    img,
//                    cv::Point(re.box.x, re.box.y - 25),
//                    cv::Point(re.box.x + label.length() * 15, re.box.y),
//                    color,
//                    cv::FILLED
//                );
//
//                cv::putText(
//                    img,
//                    label,
//                    cv::Point(re.box.x, re.box.y - 5),
//                    cv::FONT_HERSHEY_SIMPLEX,
//                    0.75,
//                    cv::Scalar(0, 0, 0),
//                    2
//                );
//
//
//            }
//            std::cout << "Press any key to exit" << std::endl;
//            cv::imshow("Result of Detection", img);
//            cv::waitKey(0);
//            cv::destroyAllWindows();
//        }
//    }
//}
//
//
//void Classifier(YOLO_V8*& p)
//{
//    std::filesystem::path current_path = std::filesystem::current_path();
//    std::filesystem::path imgs_path = current_path;// / "images"
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_int_distribution<int> dis(0, 255);
//    for (auto& i : std::filesystem::directory_iterator(imgs_path))
//    {
//        if (i.path().extension() == ".jpg" || i.path().extension() == ".png")
//        {
//            std::string img_path = i.path().string();
//            //std::cout << img_path << std::endl;
//            cv::Mat img = cv::imread(img_path);
//            std::vector<DL_RESULT> res;
//            char* ret = p->RunSession(img, res);
//
//            float positionY = 50;
//            for (int i = 0; i < res.size(); i++)
//            {
//                int r = dis(gen);
//                int g = dis(gen);
//                int b = dis(gen);
//                cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
//                cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
//                positionY += 50;
//            }
//
//            cv::imshow("TEST_CLS", img);
//            cv::waitKey(0);
//            cv::destroyAllWindows();
//            //cv::imwrite("E:\\output\\" + std::to_string(k) + ".png", img);
//        }
//
//    }
//}
//
//
//
//int ReadCocoYaml(YOLO_V8*& p) {
//    // Open the YAML file
//    std::ifstream file("coco.yaml");
//    if (!file.is_open())
//    {
//        std::cerr << "Failed to open file" << std::endl;
//        return 1;
//    }
//
//    // Read the file line by line
//    std::string line;
//    std::vector<std::string> lines;
//    while (std::getline(file, line))
//    {
//        lines.push_back(line);
//    }
//
//    // Find the start and end of the names section
//    std::size_t start = 0;
//    std::size_t end = 0;
//    for (std::size_t i = 0; i < lines.size(); i++)
//    {
//        if (lines[i].find("names:") != std::string::npos)
//        {
//            start = i + 1;
//        }
//        else if (start > 0 && lines[i].find(':') == std::string::npos)
//        {
//            end = i;
//            break;
//        }
//    }
//
//    // Extract the names
//    std::vector<std::string> names;
//    for (std::size_t i = start; i < end; i++)
//    {
//        std::stringstream ss(lines[i]);
//        std::string name;
//        std::getline(ss, name, ':'); // Extract the number before the delimiter
//        std::getline(ss, name); // Extract the string after the delimiter
//        names.push_back(name);
//    }
//
//    p->classes = names;
//    return 0;
//}
//
//
//void DetectTest()
//{
//    YOLO_V8* yoloDetector = new YOLO_V8;
//    ReadCocoYaml(yoloDetector);
//    DL_INIT_PARAM params;
//    params.rectConfidenceThreshold = 0.1;
//    params.iouThreshold = 0.5;
//    params.modelPath = "yolov8n.onnx";
//    params.imgSize = { 640, 640 };
//#ifdef USE_CUDA
//    params.cudaEnable = true;
//
//    // GPU FP32 inference
//    params.modelType = YOLO_DETECT_V8;
//    // GPU FP16 inference
//    //Note: change fp16 onnx model
//    //params.modelType = YOLO_DETECT_V8_HALF;
//
//#else
//    // CPU inference
//    params.modelType = YOLO_DETECT_V8;
//    params.cudaEnable = false;
//
//#endif
//    yoloDetector->CreateSession(params);
//    Detector(yoloDetector);
//}
//
//
//void ClsTest()
//{
//    YOLO_V8* yoloDetector = new YOLO_V8;
//    std::string model_path = "cls.onnx";
//    ReadCocoYaml(yoloDetector);
//    DL_INIT_PARAM params{ model_path, YOLO_CLS, {224, 224} };
//    yoloDetector->CreateSession(params);
//    Classifier(yoloDetector);
//}
//
//
//int main()
//{
//    //DetectTest();
//    ClsTest();
//}