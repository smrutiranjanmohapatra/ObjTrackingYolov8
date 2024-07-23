////#include <opencv2/opencv.hpp>
////#include <opencv2/tracking.hpp>
////#include <opencv2/dnn.hpp>
////#include <fstream>
////#include <iostream>
////#include <vector>
////
////using namespace cv;
////using namespace cv::dnn;
////using namespace std;
////
////class ObjectDetection {
////private:
////    Net net;
////    vector<string> classes;
////    vector<Scalar> colors;
////    float nmsThreshold;
////    float confThreshold;
////    int image_size;
////
////public:
////    ObjectDetection(const string& model_path, bool use_cuda) {
////        cout << "Loading Object Detection" << endl;
////        cout << "Running OpenCV DNN with YOLOv5s" << endl;
////
////        nmsThreshold = 0.4;
////        confThreshold = 0.4;
////        image_size = 640;
////
////        // Load Network
////        net = readNet(model_path);
////
////        // Enable GPU CUDA or CPU
////        if (use_cuda) {
////            net.setPreferableBackend(DNN_BACKEND_CUDA);
////            net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
////        }
////        else {
////            net.setPreferableBackend(DNN_BACKEND_OPENCV);
////            net.setPreferableTarget(DNN_TARGET_CPU);
////        }
////
////        // Load class names
////        loadClassNames("D:\\OPENCV\\dnn_model\\classes.txt");
////
////        // Generate random colors for each class
////        RNG rng(0xFFFFFFFF);
////        for (size_t i = 0; i < classes.size(); i++) {
////            Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
////            colors.push_back(color);
////        }
////    }
////
////    vector<string> loadClassNames(const string& classes_path) {
////        ifstream ifs(classes_path.c_str());
////        string line;
////        while (getline(ifs, line)) {
////            classes.push_back(line);
////        }
////        return classes;
////    }
////
////    void detect(Mat& frame, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) {
////        Mat blob;
////        blobFromImage(frame, blob, 1 / 255.0, Size(image_size, image_size), Scalar(), true, false);
////        net.setInput(blob);
////
////        // Run the forward pass to get output from the output layers
////        vector<Mat> outs;
////        net.forward(outs, net.getUnconnectedOutLayersNames());
////
////        postprocess(frame, outs, classIds, confidences, boxes);
////    }
////
////    void postprocess(Mat& frame, const vector<Mat>& outs, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) {
////        float x_factor = frame.cols / (float)image_size;
////        float y_factor = frame.rows / (float)image_size;
////
////        for (const auto& out : outs) {
////            float* data = (float*)out.data;
////            for (int j = 0; j < out.rows; ++j, data += out.cols) {
////                float confidence = data[4];
////                if (confidence > confThreshold) {
////                    Mat scores = out.row(j).colRange(5, out.cols);
////                    Point classIdPoint;
////                    double max_class_score;
////                    minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
////                    if (max_class_score > nmsThreshold) {
////                        int centerX = static_cast<int>(data[0] * x_factor);
////                        int centerY = static_cast<int>(data[1] * y_factor);
////                        int width = static_cast<int>(data[2] * x_factor);
////                        int height = static_cast<int>(data[3] * y_factor);
////                        int left = centerX - width / 2;
////                        int top = centerY - height / 2;
////
////                        classIds.push_back(classIdPoint.x);
////                        confidences.push_back(confidence);
////                        boxes.push_back(Rect(left, top, width, height));
////                    }
////                }
////            }
////        }
////
////        // Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences
////        vector<int> indices;
////        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
////        for (size_t i = 0; i < indices.size(); ++i) {
////            int idx = indices[i];
////            Rect box = boxes[idx];
////            drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
////        }
////    }
////
////    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
////        // Draw a bounding box.
////        rectangle(frame, Point(left, top), Point(right, bottom), colors[classId], 3);
////
////        // Get the label for the class name and its confidence
////        string label = format("%.2f", conf);
////        if (!classes.empty()) {
////            CV_Assert(classId < (int)classes.size());
////            label = classes[classId] + ":" + label;
////        }
////
////        // Display the label at the top of the bounding box
////        int baseLine;
////        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
////        top = max(top, labelSize.height);
////        rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar::all(255), FILLED);
////        putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(), 1);
////    }
////};
////
////int main() {
////    // Initialize the object detection with model path
////    ObjectDetection objectDetection("D:\\OPENCV\\yolov10\\yolov5s.onnx", true);
////
////    // Load a video file
////    VideoCapture cap("D:\\OPENCV\\Images\\24.mp4");
////    if (!cap.isOpened()) {
////        cerr << "Error opening video file" << endl;
////        return -1;
////    }
////
////    Mat frame;
////    cap.read(frame);
////    resize(frame, frame, Size(800, 800));
////    // Detect objects in the first frame
////    vector<int> classIds;
////    vector<float> confidences;
////    vector<Rect> boxes;
////    objectDetection.detect(frame, classIds, confidences, boxes);
////
////    // Initialize trackers for each detected object
////    vector<Ptr<Tracker>> trackers;
////    for (const auto& box : boxes) {
////        Ptr<Tracker> tracker = TrackerCSRT::create();
////        tracker->init(frame, box);
////        trackers.push_back(tracker);
////    }
////
////    while (cap.read(frame)) {
////        if (frame.empty()) {
////            break;
////        }
////        resize(frame, frame, Size(800, 800));
////        // Update trackers
////        for (size_t i = 0; i < trackers.size(); ++i) {
////            Rect box;
////            bool ok = trackers[i]->update(frame, box);
////            if (ok) {
////                rectangle(frame, box, Scalar(0, 255, 0), 2, LINE_AA);
////            }
////            else {
////                cout << "Tracking failure detected for object " << i + 1 << endl;
////            }
////        }
////
////        // Display the frame with detections
////        imshow("Detections", frame);
////
////        // Exit if ESC key is pressed
////        int key = waitKey(1);
////        if (key == 27) break;
////    }
////
////    cap.release();
////    destroyAllWindows();
////    return 0;
////}
//#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
//#include <opencv2/dnn.hpp>
//#include <fstream>
//#include <iostream>
//#include <vector>
//
//using namespace cv;
//using namespace cv::dnn;
//using namespace std;
//
//class ObjectDetection {
//private:
//    Net net;
//    vector<string> classes;
//    vector<Scalar> colors;
//    float nmsThreshold;
//    float confThreshold;
//    int image_size;
//
//public:
//    ObjectDetection(const string& model_path, bool use_cuda) {
//        cout << "Loading Object Detection" << endl;
//        cout << "Running OpenCV DNN with YOLOv5s" << endl;
//
//        nmsThreshold = 0.4;
//        confThreshold = 0.4;
//        image_size = 640;
//
//        // Load Network
//        net = readNet(model_path);
//
//        // Enable GPU CUDA or CPU
//        if (use_cuda) {
//            net.setPreferableBackend(DNN_BACKEND_CUDA);
//            net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
//        }
//        else {
//            net.setPreferableBackend(DNN_BACKEND_OPENCV);
//            net.setPreferableTarget(DNN_TARGET_CPU);
//        }
//
//        // Load class names
//        loadClassNames("D:\\OPENCV\\dnn_model\\classes.txt");
//
//        // Generate random colors for each class
//        RNG rng(0xFFFFFFFF);
//        for (size_t i = 0; i < classes.size(); i++) {
//            Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
//            colors.push_back(color);
//        }
//    }
//
//    vector<string> loadClassNames(const string& classes_path) {
//        ifstream ifs(classes_path.c_str());
//        string line;
//        while (getline(ifs, line)) {
//            classes.push_back(line);
//        }
//        return classes;
//    }
//
//    void detect(Mat& frame, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) {
//        Mat blob;
//        blobFromImage(frame, blob, 1 / 255.0, Size(image_size, image_size), Scalar(), true, false);
//        net.setInput(blob);
//
//        // Run the forward pass to get output from the output layers
//        vector<Mat> outs;
//        net.forward(outs, net.getUnconnectedOutLayersNames());
//
//        postprocess(frame, outs, classIds, confidences, boxes);
//    }
//
//    void postprocess(Mat& frame, const vector<Mat>& outs, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) {
//        float x_factor = frame.cols / (float)image_size;
//        float y_factor = frame.rows / (float)image_size;
//
//        for (const auto& out : outs) {
//            float* data = (float*)out.data;
//            for (int j = 0; j < out.rows; ++j, data += out.cols) {
//                float confidence = data[4];
//                if (confidence > confThreshold) {
//                    Mat scores = out.row(j).colRange(5, out.cols);
//                    Point classIdPoint;
//                    double max_class_score;
//                    minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
//                    if (max_class_score > nmsThreshold) {
//                        int centerX = static_cast<int>(data[0] * x_factor);
//                        int centerY = static_cast<int>(data[1] * y_factor);
//                        int width = static_cast<int>(data[2] * x_factor);
//                        int height = static_cast<int>(data[3] * y_factor);
//                        int left = centerX - width / 2;
//                        int top = centerY - height / 2;
//
//                        classIds.push_back(classIdPoint.x);
//                        confidences.push_back(confidence);
//                        boxes.push_back(Rect(left, top, width, height));
//                    }
//                }
//            }
//        }
//
//        // Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences
//        vector<int> indices;
//        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
//        for (size_t i = 0; i < indices.size(); ++i) {
//            int idx = indices[i];
//            Rect box = boxes[idx];
//            drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
//        }
//    }
//
//    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
//        // Draw a bounding box.
//        rectangle(frame, Point(left, top), Point(right, bottom), colors[classId], 3);
//
//        // Get the label for the class name and its confidence
//        string label = format("%.2f", conf);
//        if (!classes.empty()) {
//            CV_Assert(classId < (int)classes.size());
//            label = classes[classId] + ":" + label;
//        }
//
//        // Display the label at the top of the bounding box
//        int baseLine;
//        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//        top = max(top, labelSize.height);
//        rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar::all(255), FILLED);
//        putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(), 1);
//    }
//};
//
//int main() {
//    // Initialize the object detection with model path
//    ObjectDetection objectDetection("D:\\OPENCV\\yolov10\\yolov5s.onnx", true);
//
//    // Load a video file
//    VideoCapture cap("D:\\OPENCV\\Images\\24.mp4");
//    if (!cap.isOpened()) {
//        cerr << "Error opening video file" << endl;
//        return -1;
//    }
//
//    Mat frame;
//    cap.read(frame);
//    resize(frame, frame, Size(800, 800));
//
//    // Detect objects in the first frame
//    vector<int> classIds;
//    vector<float> confidences;
//    vector<Rect> boxes;
//    objectDetection.detect(frame, classIds, confidences, boxes);
//
//    // Check if any detections were made
//    if (classIds.empty()) {
//        cerr << "No objects detected in the initial frame" << endl;
//        return -1;
//    }
//
//    // Initialize trackers for each detected object
//    vector<Ptr<Tracker>> trackers;
//    for (const auto& box : boxes) {
//        Ptr<Tracker> tracker = TrackerCSRT::create();
//        tracker->init(frame, box);
//        trackers.push_back(tracker);
//    }
//
//    while (cap.read(frame)) {
//        if (frame.empty()) {
//            break;
//        }
//        resize(frame, frame, Size(800, 800));
//
//        // Update trackers
//        for (size_t i = 0; i < trackers.size(); ++i) {
//            Rect box;
//            bool ok = trackers[i]->update(frame, box);
//            if (ok) {
//                rectangle(frame, box, Scalar(0, 255, 0), 2, LINE_AA);
//            }
//            else {
//                cout << "Tracking failure detected for object " << i + 1 << endl;
//            }
//        }
//
//        // Display the frame with detections
//        imshow("Detections", frame);
//
//        // Exit if ESC key is pressed
//        int key = waitKey(1);
//        if (key == 27) break;
//    }
//
//    cap.release();
//    destroyAllWindows();
//    return 0;
//}


//#include <fstream>
//#include<mutex>
//#include <opencv2/opencv.hpp>
//#include<iostream>
//
//
//using namespace std;
//using namespace cv;
//using namespace dnn;
//
//Mutex mx;
//bool is_cuda = 0;
//queue<Mat>video;
//vector<string> class_list;
//vector<string> load_class_list()  //reads class labels from a file(classes.txt) and stores them in class_list.
//{
//    ifstream ifs("D:\\OPENCV\\yolov5\\classes.txt");
//    string line;
//    while (getline(ifs, line))
//    {
//        class_list.push_back(line);
//    }
//    return class_list;
//}
//
//void load_net(Net& net, bool is_cuda) //loads the YOLOv5 model from an ONNX file (yolov5s.onnx) and sets its backend (CUDA or OpenCV) based on the is_cuda flag.
//{
//    // auto result = readNet("D:\\OpenCV_Person_Deteection\\Python\\yolov5-opencv-cpp-python\\config_files\\yolov5s.onnx");
//    auto result = readNet("D:\\OPENCV\\yolov5\\yolov5\\yolov5l.onnx");
//    if (is_cuda)
//    {
//        cout << "Attempt to use CUDA\n";
//        result.setPreferableBackend(DNN_BACKEND_CUDA);
//        result.setPreferableTarget(DNN_TARGET_CUDA_FP16);
//    }
//    else
//    {
//        cout << "Running on CPU\n";
//        result.setPreferableBackend(DNN_BACKEND_OPENCV);
//        result.setPreferableTarget(DNN_TARGET_CPU);
//    }
//    net = result;
//}
//
//const vector<Scalar> colors = { Scalar(0,255, 0) };
//
//const float INPUT_WIDTH = 640.0;
//const float INPUT_HEIGHT = 640.0;
//const float SCORE_THRESHOLD = 0.5;
//const float NMS_THRESHOLD = 0.6;
//const float CONFIDENCE_THRESHOLD = 0.5;
//
//struct Detection
//{
//    int class_id;
//    float confidence;
//    Rect box;
//};
//
//Mat format_yolov5(const Mat& source) //format_yolov5() function resizes the input image to square dimensions (max x max) required by the YOLOv5 model and returns the resized image.
//{
//    int col = source.cols;
//    int row = source.rows;
//    int max = MAX(col, row);
//    Mat result = Mat::zeros(max, max, CV_8UC3);
//    source.copyTo(result(Rect(0, 0, col, row)));
//    return result;
//}
//void detect()
//{
//    Mat blob;
//    Net net;
//    Mat frame1;
//    load_net(net, is_cuda);
//    while (!video.empty())
//    {
//        auto start = std::chrono::high_resolution_clock::now();
//        mx.lock();
//        frame1 = video.front();
//        video.pop();
//        mx.unlock();
//        if (frame1.rows == 0 && frame1.cols == 0)
//            break;
//        auto input_image = format_yolov5(frame1);
//
//        blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false); //convert input image to a blob suitable for the network
//        net.setInput(blob);  //set prepared blob as input to the network
//
//        vector<Mat> outputs;  //forward pass through the network to get output detections
//        net.forward(outputs, net.getUnconnectedOutLayersNames());
//
//        //Calculate scaling factors to map box coordinates back to original image size
//        float x_factor = input_image.cols / INPUT_WIDTH;
//        float y_factor = input_image.rows / INPUT_HEIGHT;
//
//        // Get pointer to the data of the first output layer
//        float* data = (float*)outputs[0].data;
//        // Constants for YOLOv5 model
//        const int dimensions = 85;
//        const int rows = 25200;
//
//        // Initialize containers for class IDs, confidences, and bounding boxes
//        vector<int> class_ids;
//        vector<float> confidences;
//        vector<cv::Rect> boxes;
//
//        //Iterate through each detection output
//        for (int i = 0; i < rows; ++i)
//        {
//            // Extract confidence score for the detection
//            float confidence = data[4];
//
//            // Check if confidence meets the threshold
//            if (confidence >= CONFIDENCE_THRESHOLD) {
//
//                float* classes_scores = data + 5;//Extract class score
//                Mat scores(1, class_list.size(), CV_32FC1, classes_scores);//create a matrix from class scores
//                // Find maximum class score and its index
//                Point class_id;
//                double max_class_score;
//                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
//
//                if (max_class_score > SCORE_THRESHOLD)
//                {
//
//                    confidences.push_back(confidence);//store confidence score
//
//                    class_ids.push_back(class_id.x);//store class id
//
//                    float x = data[0];
//                    float y = data[1];
//                    float w = data[2];
//                    float h = data[3];
//
//                    //Calculate bounding box coordinates scaled to the original image size
//                    int left = int((x - 0.5 * w) * x_factor);
//                    int top = int((y - 0.5 * h) * y_factor);
//                    int width = int(w * x_factor);
//                    int height = int(h * y_factor);
//                    boxes.push_back(Rect(left, top, width, height));//store the bounding box
//                }
//
//            }
//
//            data += dimensions; //move to next detection
//
//        }
//
//        vector<Detection> output;
//        vector<int> nms_result;
//        NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
//
//        //store the final detection in output vector
//        for (int i = 0; i < nms_result.size(); i++) {
//            int idx = nms_result[i];
//            Detection result;
//            result.class_id = class_ids[idx];
//            result.confidence = confidences[idx];
//            result.box = boxes[idx];
//            output.push_back(result);
//        }
//        int detections = output.size();
//
//        for (int i = 0; i < detections; ++i)
//        {
//            auto detection = output[i];
//            auto box = detection.box;
//            auto classId = detection.class_id;
//            const auto color = colors[classId % colors.size()];
//            rectangle(frame1, box, color, 2);
//
//            //cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
//           // cv::putText(frame1, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
//        }
//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double, std::milli> elapsedTime = end - start;
//        std::cout << endl << "------------ time:----------- " << elapsedTime.count() << "ms" << endl;
//
//
//        imshow("output", frame1);
//        char c = waitKey(1);
//        if (c == 27)
//            break;
//        else if (c == 32)
//            waitKey(0);
//    }
//    detect();
//}
//void capture()
//{
//    // VideoCapture capture("rtsp://admin:Oditek123@@10.30.30.53/1/2");
//    VideoCapture capture("D:\\OPENCV\\Images\\24.mp4");
//    if (!capture.isOpened())
//    {
//        cerr << "Error opening video file\n";
//        return;
//    }
//    int i = 0;
//    while (true)
//    {
//        Mat frame;
//        capture.read(frame);
//        if (i == 15)
//        {
//            mx.lock();
//            video.push(frame);
//            mx.unlock();
//            i = 0;
//        }
//        else
//            i++;
//
//    }
//
//
//}
//int main()
//{
//    Net net;
//    load_net(net, is_cuda);
//    vector<string> class_list = load_class_list();
//    thread t1(capture);
//    thread t2(detect);
//    t1.join();
//    t2.join();
//
//    return 0;
//}




//
//#include <fstream>
//#include <mutex>
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <queue>
//#include <thread>
//
//using namespace std;
//using namespace cv;
//using namespace dnn;
//
//mutex mx;
//bool is_cuda = 0;
//queue<Mat> video;
//vector<string> class_list;
//vector<string> load_class_list()  //reads class labels from a file(classes.txt) and stores them in class_list.
//{
//    ifstream ifs("D:\\OPENCV\\dnn_model\\classes.txt");
//    string line;
//    while (getline(ifs, line))
//    {
//        class_list.push_back(line);
//    }
//    return class_list;
//}
//
//void load_net(Net& net, bool is_cuda) //loads the YOLOv5 model from an ONNX file (yolov5s.onnx) and sets its backend (CUDA or OpenCV) based on the is_cuda flag.
//{
//    // auto result = readNet("D:\\OpenCV_Person_Deteection\\Python\\yolov5-opencv-cpp-python\\config_files\\yolov5s.onnx");
//    auto result = readNet("D:\\OPENCV\\yolov10\\yolov5s.onnx");
//    if (is_cuda)
//    {
//        cout << "Attempt to use CUDA\n";
//        result.setPreferableBackend(DNN_BACKEND_CUDA);
//        result.setPreferableTarget(DNN_TARGET_CUDA_FP16);
//    }
//    else
//    {
//        cout << "Running on CPU\n";
//        result.setPreferableBackend(DNN_BACKEND_OPENCV);
//        result.setPreferableTarget(DNN_TARGET_CPU);
//    }
//    net = result;
//}
//
//const vector<Scalar> colors = { Scalar(0, 255, 0) };
//
//const float INPUT_WIDTH = 640.0;
//const float INPUT_HEIGHT = 640.0;
//const float SCORE_THRESHOLD = 0.5;
//const float NMS_THRESHOLD = 0.6;
//const float CONFIDENCE_THRESHOLD = 0.5;
//
//struct Detection
//{
//    int class_id;
//    float confidence;
//    Rect box;
//};
//
//Mat format_yolov5(const Mat& source) //format_yolov5() function resizes the input image to square dimensions (max x max) required by the YOLOv5 model and returns the resized image.
//{
//    int col = source.cols;
//    int row = source.rows;
//    int max = MAX(col, row);
//    Mat result = Mat::zeros(max, max, CV_8UC3);
//    source.copyTo(result(Rect(0, 0, col, row)));
//    return result;
//}
//
//void detect()
//{
//    Mat blob;
//    Net net;
//    Mat frame1;
//    load_net(net, is_cuda);
//    while (!video.empty())
//    {
//        auto start = std::chrono::high_resolution_clock::now();
//        mx.lock();
//        if (video.empty()) {
//            mx.unlock();
//            break;
//        }
//        frame1 = video.front();
//        video.pop();
//        mx.unlock();
//        if (frame1.empty()) {
//            cerr << "Error: frame is empty" << endl;
//            break;
//        }
//        auto input_image = format_yolov5(frame1);
//
//        blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false); //convert input image to a blob suitable for the network
//        net.setInput(blob);  //set prepared blob as input to the network
//
//        vector<Mat> outputs;  //forward pass through the network to get output detections
//        net.forward(outputs, net.getUnconnectedOutLayersNames());
//
//        if (outputs.empty()) {
//            cerr << "Error: outputs are empty" << endl;
//            break;
//        }
//
//        //Calculate scaling factors to map box coordinates back to original image size
//        float x_factor = input_image.cols / INPUT_WIDTH;
//        float y_factor = input_image.rows / INPUT_HEIGHT;
//
//        // Ensure outputs[0] is not empty and has the expected size
//        const int dimensions = 85;
//        const int rows = 2000;
//            /*outputs[0].total() / dimensions;
//
//        if (outputs[0].total() % dimensions != 0) {
//            cerr << "Error: Output size is not divisible by dimensions" << endl;
//            break;
//        }*/
//
//        float* data = (float*)outputs[0].data;
//        if (data == nullptr) {
//            cerr << "Error: data is null" << endl;
//            break;
//        }
//
//        // Initialize containers for class IDs, confidences, and bounding boxes
//        vector<int> class_ids;
//        vector<float> confidences;
//        vector<cv::Rect> boxes;
//
//        //Iterate through each detection output
//        for (int i = 0; i < rows; ++i)
//        {
//            // Extract confidence score for the detection
//            float confidence = data[4];
//
//            // Check if confidence meets the threshold
//            if (confidence >= CONFIDENCE_THRESHOLD) {
//
//                float* classes_scores = data + 5;//Extract class score
//                Mat scores(1, class_list.size(), CV_32FC1, classes_scores);//create a matrix from class scores
//                Point class_id;
//                double max_class_score;
//                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
//
//                if (max_class_score > SCORE_THRESHOLD)
//                {
//
//                    confidences.push_back(confidence);//store confidence score
//
//                    class_ids.push_back(class_id.x);//store class id
//
//                    float x = data[0];
//                    float y = data[1];
//                    float w = data[2];
//                    float h = data[3];
//
//                    //Calculate bounding box coordinates scaled to the original image size
//                    int left = int((x - 0.5 * w) * x_factor);
//                    int top = int((y - 0.5 * h) * y_factor);
//                    int width = int(w * x_factor);
//                    int height = int(h * y_factor);
//                    boxes.push_back(Rect(left, top, width, height));//store the bounding box
//                }
//            }
//            data += dimensions; //move to next detection
//        }
//
//        vector<Detection> output;
//        vector<int> nms_result;
//        NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
//
//        //store the final detection in output vector
//        for (int i = 0; i < nms_result.size(); i++) {
//            int idx = nms_result[i];
//            Detection result;
//            result.class_id = class_ids[idx];
//            result.confidence = confidences[idx];
//            result.box = boxes[idx];
//            output.push_back(result);
//        }
//        int detections = output.size();
//
//        for (int i = 0; i < detections; ++i)
//        {
//            auto detection = output[i];
//            auto box = detection.box;
//            auto classId = detection.class_id;
//            const auto color = colors[classId % colors.size()];
//            rectangle(frame1, box, color, 2);
//        }
//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double, std::milli> elapsedTime = end - start;
//        std::cout << endl << "------------ time:----------- " << elapsedTime.count() << "ms" << endl;
//
//        imshow("output", frame1);
//        char c = waitKey(1);
//        if (c == 27)
//            break;
//        else if (c == 32)
//            waitKey(0);
//    }
//}
//
//void capture()
//{
//    // VideoCapture capture("rtsp://admin:Oditek123@@10.30.30.53/1/2");
//    VideoCapture capture("D:\\OPENCV\\Images\\24.mp4");
//    if (!capture.isOpened())
//    {
//        cerr << "Error opening video file\n";
//        return;
//    }
//    int i = 0;
//    while (true)
//    {
//        Mat frame;
//        capture.read(frame);
//        if (frame.empty()) {
//            cerr << "Error: frame is empty" << endl;
//            break;
//        }
//        if (i == 15)
//        {
//            mx.lock();
//            video.push(frame);
//            mx.unlock();
//            i = 0;
//        }
//        else
//            i++;
//    }
//}
//
//int main()
//{
//    Net net;
//    load_net(net, is_cuda);
//    vector<string> class_list = load_class_list();
//    if (class_list.empty()) {
//        cerr << "Error: class_list is empty" << endl;
//        return -1;
//    }
//    thread t1(capture);
//    thread t2(detect);
//    t1.join();
//    t2.join();
//
//    return 0;
//}


//
//#include <fstream>
//
//#include <opencv2/opencv.hpp>
//
//std::vector<std::string> load_class_list()
//{
//    std::vector<std::string> class_list;
//    std::ifstream ifs("D:\\OPENCV\\yolov10\\classes.txt");
//    std::string line;
//    while (getline(ifs, line))
//    {
//        class_list.push_back(line);
//    }
//    return class_list;
//}
//
//void load_net(cv::dnn::Net& net, bool is_cuda)
//{
//    auto result = cv::dnn::readNet("D:\\OPENCV\\yolov8n\\yolov8m.onnx");
//    if (is_cuda)
//    {
//        std::cout << "Attempty to use CUDA\n";
//        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
//    }
//    else
//    {
//        std::cout << "Running on CPU\n";
//        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//    }
//    net = result;
//}
//
//const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };
//
//const float INPUT_WIDTH = 640.0;
//const float INPUT_HEIGHT = 640.0;
//const float SCORE_THRESHOLD = 0.2;
//const float NMS_THRESHOLD = 0.4;
//const float CONFIDENCE_THRESHOLD = 0.4;
//
//struct Detection
//{
//    int class_id;
//    float confidence;
//    cv::Rect box;
//};
//
//cv::Mat format_yolov5(const cv::Mat& source) {
//    int col = source.cols;
//    int row = source.rows;
//    int _max = MAX(col, row);
//    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
//    source.copyTo(result(cv::Rect(0, 0, col, row)));
//    return result;
//}
//
//void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
//    cv::Mat blob;
//
//    auto input_image = format_yolov5(image);
//
//    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
//    net.setInput(blob);
//    std::vector<cv::Mat> outputs;
//    net.forward(outputs, net.getUnconnectedOutLayersNames());
//
//    float x_factor = input_image.cols / INPUT_WIDTH;
//    float y_factor = input_image.rows / INPUT_HEIGHT;
//
//    float* data = (float*)outputs[0].data;
//
//    const int dimensions = 85;
//    const int rows = 25200;
//
//    std::vector<int> class_ids;
//    std::vector<float> confidences;
//    std::vector<cv::Rect> boxes;
//
//    for (int i = 0; i < rows; ++i) {
//        float confidence = data[4];
//        if (confidence >= CONFIDENCE_THRESHOLD) {
//
//            float* classes_scores = data + 5;
//            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
//            cv::Point class_id;
//            double max_class_score;
//            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
//            if (max_class_score > SCORE_THRESHOLD) {
//
//                confidences.push_back(confidence);
//
//                class_ids.push_back(class_id.x);
//
//                float x = data[0];
//                float y = data[1];
//                float w = data[2];
//                float h = data[3];
//                int left = int((x - 0.5 * w) * x_factor);
//                int top = int((y - 0.5 * h) * y_factor);
//                int width = int(w * x_factor);
//                int height = int(h * y_factor);
//                boxes.push_back(cv::Rect(left, top, width, height));
//            }
//
//        }
//
//        data += 85;
//
//    }
//
//    std::vector<int> nms_result;
//    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
//    for (int i = 0; i < nms_result.size(); i++) {
//        int idx = nms_result[i];
//        Detection result;
//        result.class_id = class_ids[idx];
//        result.confidence = confidences[idx];
//        result.box = boxes[idx];
//        output.push_back(result);
//    }
//}
//
//int main(int argc, char** argv)
//{
//
//    std::vector<std::string> class_list = load_class_list();
//
//    cv::Mat frame;
//    cv::VideoCapture capture("D:\\OPENCV\\Images\\24.mp4");
//    if (!capture.isOpened())
//    {
//        std::cerr << "Error opening video file\n";
//        return -1;
//    }
//
//    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
//
//    cv::dnn::Net net;
//    load_net(net, is_cuda);
//
//    auto start = std::chrono::high_resolution_clock::now();
//    int frame_count = 0;
//    float fps = -1;
//    int total_frames = 0;
//
//    while (true)
//    {
//        capture.read(frame);
//        if (frame.empty())
//        {
//            std::cout << "End of stream\n";
//            break;
//        }
//
//        std::vector<Detection> output;
//        detect(frame, net, output, class_list);
//
//        frame_count++;
//        total_frames++;
//
//        int detections = output.size();
//
//        for (int i = 0; i < detections; ++i)
//        {
//
//            auto detection = output[i];
//            auto box = detection.box;
//            auto classId = detection.class_id;
//            const auto color = colors[classId % colors.size()];
//            cv::rectangle(frame, box, color, 3);
//
//            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
//            cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
//        }
//
//        if (frame_count >= 30)
//        {
//
//            auto end = std::chrono::high_resolution_clock::now();
//            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//
//            frame_count = 0;
//            start = std::chrono::high_resolution_clock::now();
//        }
//
//        if (fps > 0)
//        {
//
//            std::ostringstream fps_label;
//            fps_label << std::fixed << std::setprecision(2);
//            fps_label << "FPS: " << fps;
//            std::string fps_label_str = fps_label.str();
//
//            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
//        }
//
//        cv::imshow("output", frame);
//
//        if (cv::waitKey(1) != -1)
//        {
//            capture.release();
//            std::cout << "finished by user\n";
//            break;
//        }
//    }
//
//    std::cout << "Total frames: " << total_frames << "\n";
//
//    return 0;
//}