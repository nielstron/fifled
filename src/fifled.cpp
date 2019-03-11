#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


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

int main(int argc, char** argv)
{
    const String keys =
        "{help h usage ? |      | print this message }"
        "{bbthresh bt    | 500  | pixel threshold to be accepted as a 'thing' to be bounded }"
        "{flowthresh ft  | 2    | minimum flow in pixels to be marked as a moving pixel }"
        "{infile i       |      | path to a video file that should be parsed (default: camera 0)}"
        "{outfile o      |      | path to a video file that should be outputted}";
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

	Mat flow, cflow, frame, overlay, dst;
	cap->read(frame);
	Mat gray, prevgray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	gray.copyTo(prevgray);
	Mat uflow(frame.size(), CV_32FC2);

	// Writing the labels out
	VideoWriter* out;
	if("" != parser.get<String>("outfile")){
		out = new VideoWriter(parser.get<String>("outfile"), VideoWriter::fourcc('M','J','P','G'), 15, Size(frame.cols, frame.rows));
	}
	else {
		out = NULL;
	}

	Mat labelImage(frame.size(), CV_32S);
	Mat connected(frame.size(), CV_8UC3);

	int hullTresh = parser.get<int>("bbthresh");
	int flowthresh = parser.get<int>("flowthresh");
	flowthresh *= flowthresh;
	vector<Point> hull;

	while(cap->read(frame)) {
		imshow("input", frame);
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		calcOpticalFlowFarneback(prevgray, gray, uflow, 0.25, 3, 15, 3, 5, 1.2, OPTFLOW_USE_INITIAL_FLOW);
		cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
		cvtColor(prevgray, dst, COLOR_GRAY2BGR);

		// Calculate which parts of the image are moving
		overlay = Mat::zeros(cflow.size(), CV_8UC1);
		drawOptFlowMap(uflow, overlay, 1, 255, flowthresh);

		// Detect connected moving parts

		int nLabels = connectedComponents(overlay, labelImage, 8);
		// -- Drawing
		std::vector<Vec3b> colors(nLabels);
		colors[0] = Vec3b(0, 0, 0);//background
		for (int label = 1; label < nLabels; ++label) {
			colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
		}
		/*for (int r = 0; r < connected.rows; ++r) {
		for (int c = 0; c < connected.cols; ++c) {
		int label = labelImage.at<int>(r, c);
		Vec3b &pixel = connected.at<Vec3b>(r, c);
		pixel = colors[label];
		}
		}*/
		//addWeighted(cflow, 0.5, connected, 0.5, 0.0, dst);
		// Fit ellipses to them or convex hulls or fitting rectangles

		// Convex hull => collect points belonging to one hull
		vector<vector<Point>> connectedParts(nLabels);

		for (int r = 0; r < dst.rows; ++r) {
			for (int c = 0; c < dst.cols; ++c) {
				int label = labelImage.at<int>(r, c);
				connectedParts[label].push_back(Point(c, r));
			}
		}
		for (int i = 1; i < nLabels; i++) {
			if (connectedParts[i].size() < hullTresh) {
				continue;
			}
			Rect r = boundingRect(connectedParts[i]);
			rectangle(dst, r, colors[i], 2);
		}

		swap(gray, prevgray);
		imshow("labels", dst);
		if(out != NULL){
			out->write(dst);
		}
		imshow("flow", overlay);
		if (waitKey(1) >= 0){
			break;
		}
	}
	if(out != NULL){
		out->release();
	}
	return EXIT_SUCCESS;
}

