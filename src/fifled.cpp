#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/cudaoptflow.hpp"

using namespace cv;
using namespace std;

struct FlowObject {
	cv::Rect boundingBox;
	cv::String label;
	cv::Vec3b color;
};

static bool comp(FlowObject a, FlowObject b){
	return a.boundingBox.area() < b.boundingBox.area();
}


static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, uchar color, int thresh) {
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			// If flow is greater than 5 pixels
			if (abs(fxy.x) + abs(fxy.y) >= thresh) {
				//line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
				cflowmap.at<uchar>(y, x) = color;
			}
			//circle(cflowmap, Point(x, y), 2, color, -1);
		}
}

static void writeLabels(const char* path, std::vector<FlowObject> rectangles){
	std::ofstream output;
	output.open(path);
	for(int i = 0; i < rectangles.size(); i++){
		cv::Rect r = rectangles[i].boundingBox;
		output << rectangles[i].label;
		output << " ";
		output << r.tl().x << " " << r.tl().y << " ";
		output << r.br().x << " " << r.br().y;
		output << "\n";
	}
	output.close();
}

int main(int argc, char** argv)
{
    const String keys =
        "{help h usage ? |       | print this message }"
        "{bbthresh bt    | 500   | pixel threshold to be accepted as a 'thing' to be bounded }"
        "{flowthresh ft  | 2     | minimum flow in pixels to be marked as a moving pixel }"
        "{infile i       |       | path to a video file or image sequence that should be parsed (default: camera 0)}"
        "{outfile o      |       | path to a video file that should be outputted}"
        "{framep fp      |<none> | path/prefix for saving video frames}"
        "{labelp lp      |<none> | path/prefix for saving label files}"
		"{maxbb mb       | 0     | select only the <n> largest bounding boxes (0 = no maximum) }"
		"{windows w      | ifl   | select which images to display. i = input, f = flow, l = labels }"
		"{labels l       |unknown| label for found boxes. overrides any other classification methods }";
	cv::CommandLineParser parser(argc, argv, keys);
	if(parser.has("help")){
		parser.printMessage();
		exit(EXIT_SUCCESS);
	}
	VideoCapture* cap;
	String in = parser.get<String>("infile");
	if("" != in){
		cap = new VideoCapture(in.c_str());
	}
	else {
		cap = new VideoCapture(0);
	}

	if (!cap->isOpened()){
		throw invalid_argument("Infile/Camera could not be opened");
	}
	// Which mats to show
	bool display_input = false, display_flow = false, display_labels = false;
	for(char c : parser.get<cv::String>("windows")){
		switch(c){
			case 'i':
				display_input = true;
				break;
			case 'f':
				display_flow = true;
				break;
			case 'l':
				display_labels = true;
				break;
		}
	}

	// Mats used
	cv::Mat overlay, dst, cflow, frame;
	cv::cuda::GpuMat flow, frameGPU, cflowGPU, dstGPU;
	cap->read(frame);
	cv::cuda::GpuMat gray, prevgray;
	frameGPU.upload(frame);
	cv::cuda::cvtColor(frameGPU, gray, COLOR_BGR2GRAY);
	gray.copyTo(prevgray);
	cv::cuda::GpuMat uflow(frame.size(), CV_32FC2);
	cv::Mat labelImage(frame.size(), CV_32S);
	cv::Mat connected(frame.size(), CV_8UC3);

	// Writing the labels out
	VideoWriter* out;
	if("" != parser.get<String>("outfile")){
		out = new VideoWriter(parser.get<String>("outfile"), VideoWriter::fourcc('M','J','P','G'), 15, Size(frame.cols, frame.rows));
	}
	else {
		out = NULL;
	}
	bool labels_save = parser.has("labelp");
	char *labels_path_buffer, *labels_path_num_pointer;
	{
		cv::String labels_path = parser.get<cv::String>("labelp");
		labels_path_buffer = new char[labels_path.length() + 100];
		strcpy(labels_path_buffer, labels_path.c_str());
		labels_path_num_pointer = labels_path_buffer+labels_path.length();
	}
	bool frames_save = parser.has("framep");
	char *frames_path_buffer, *frames_path_num_pointer;
	{
		cv::String frames_path = parser.get<cv::String>("framep");
		frames_path_buffer = new char[frames_path.length() + 100];
		strcpy(frames_path_buffer, frames_path.c_str());
		frames_path_num_pointer = frames_path_buffer+frames_path.length();
	}
	cv::String label = parser.get<cv::String>("labels");


	int hullTresh = parser.get<int>("bbthresh");
	int flowthresh = parser.get<int>("flowthresh");
	int maxbb = parser.get<int>("maxbb");
	flowthresh *= flowthresh;
	std::vector<FlowObject> rectangles;
	std::vector<FlowObject> staticObj;

	printf("\n%sPress any key to select static objects. Press ESC to exit.%s\n\n", "\033[0;34m", "\033[0m");
	int frame_num = 0;
	while(cap->read(frame)) {
		frameGPU.upload(frame);
		cv::cuda::cvtColor(frameGPU, gray, COLOR_BGR2GRAY);

		auto farneback = cv::cuda::FarnebackOpticalFlow::create();
		farneback->calc(prevgray, gray, uflow);
		//calcOpticalFlowFarneback(prevgray, gray, uflow, 0.25, 3, 15, 3, 5, 1.2, OPTFLOW_USE_INITIAL_FLOW);
		cv::cuda::cvtColor(prevgray, dstGPU, COLOR_GRAY2BGR);
		dstGPU.download(dst);
		uflow.download(cflow);

		// Calculate which parts of the image are moving
		overlay = Mat::zeros(cflow.size(), CV_8UC1);
		drawOptFlowMap(cflow, overlay, 1, 255, flowthresh);

		// Detect connected moving parts

		int nLabels = connectedComponents(overlay, labelImage, 8);
		// -- Drawing
		vector<vector<Point>> connectedParts(nLabels);

		for (int r = 0; r < dst.rows; ++r) {
			for (int c = 0; c < dst.cols; ++c) {
				int label = labelImage.at<int>(r, c);
				connectedParts[label].push_back(Point(c, r));
			}
		}
		// Compute bounding boxes
		rectangles.clear();
		for (int i = 1; i < nLabels; i++) {
			if (connectedParts[i].size() < hullTresh) {
				continue;
			}
			Rect r = boundingRect(connectedParts[i]);
			FlowObject f {
				r,
				label,
				cv::Vec3b(rand()%255, rand()%255, rand()%255)
			};
			rectangles.push_back(f);
		}
		// Handle maxbb rectangles
		std::vector<FlowObject> res(maxbb);
		if(0 == maxbb || maxbb > rectangles.size()){
			res = rectangles;
		}
		else {
			std::make_heap(begin(rectangles), end(rectangles), comp);
			for(int i = 0; i < maxbb; i++){
				FlowObject r = rectangles[0];
				std::pop_heap(begin(rectangles), end(rectangles), comp);
				rectangles.pop_back();
				res.push_back(r);
			}
		}
		// add static objects, but do now show when occluded by flow rect
		for(FlowObject s : staticObj){
			bool add = true;
			for(FlowObject b : res){
				if(b.boundingBox.contains(s.boundingBox.tl()) && b.boundingBox.contains(s.boundingBox.br())){
					add = false;
					break;
				}
			}
			if(add){
				res.push_back(s);
			}
		}
		// Drawing beautiful output
		for(int i = 0; i < res.size(); i++){
			FlowObject f = res[i];
			cv::rectangle(dst, f.boundingBox, f.color, 1);
			if("" != label){
				int baseline = 0;
				cv::Size textRec = cv::getTextSize(f.label, cv::HersheyFonts::FONT_HERSHEY_PLAIN, 0.8, 1, &baseline);
				cv::rectangle(dst, Rect(f.boundingBox.tl() - Point(0, textRec.height), f.boundingBox.tl() + Point(textRec.width, 0)), f.color, cv::FILLED);
				cv::putText(dst, f.label, f.boundingBox.tl(), cv::HersheyFonts::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
			}
		}

		if(display_input) imshow("input", frame);
		if(display_labels) imshow("labels", dst);
		if(display_flow) imshow("flow", overlay);

		// Storing frames and generated labels
		if(frames_save){
			sprintf(frames_path_num_pointer, "%.10d.png", frame_num);
			printf("%s\n", frames_path_buffer);
			cv::imwrite(frames_path_buffer, frame);
		}
		if(labels_save){
			sprintf(labels_path_num_pointer, "%.10d.txt", frame_num);
			// Write labels
			writeLabels(labels_path_buffer, res);
		}
		if(out != NULL){
			out->write(dst);
		}

		swap(gray, prevgray);
		frame_num ++;
		int c = 0;
		if ((c = waitKey(1)) >= 0 || frame_num == 1){
			if(c==27) {
				break;
			}
			else{
				// Ask for static objects
				cv::destroyAllWindows();
				Rect userIn;
				while(true){
					cv::Mat staticDisplay(frame);
					for(FlowObject f : staticObj){
						cv::rectangle(staticDisplay, f.boundingBox, cv::Scalar(0, 0, 0), 2);
					}
					cv::imshow("first_frame", staticDisplay);
					cv::setWindowTitle("first_frame", "Please select static objects");
					//cv::displayOverlay("first_frame", "Controls: use space or enter to finish current selection and start a new one, use esc to terminate multiple ROI selection process.");
					userIn = cv::selectROI("first_frame", staticDisplay);
					// TODO let user input label
					if(!userIn.empty()){
						cv::String label;
						printf("Please input the label of the static object\n");
						cin >> label;
						FlowObject f {
							userIn,
							label,
							Vec3b(255, 255, 255)
						};
						staticObj.push_back(f);
					}
					else{
						break;
					}
				}
				cv::destroyWindow("first_frame");
			}
		}
	}
	if(out != NULL){
		out->release();
	}
	return EXIT_SUCCESS;
}

