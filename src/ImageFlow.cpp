#include "opencv2/opencv.hpp"

#include <iostream>

using namespace cv;
using namespace std;


static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
	uchar color)
{
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			// If flow is greater than 5 pixels
			if (abs(fxy.x) + abs(fxy.y) > 2) {
				//line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
				cflowmap.at<uchar>(y, x) = color;
			}
			//circle(cflowmap, Point(x, y), 2, color, -1);
		}
}

std::vector<cv::Point> rect2Vect(cv::Rect r){
	std::vector<cv::Point> hull(5);
	hull[0] = r.tl();
	hull[2] = r.br();
	hull[1].y = hull[0].y;
	hull[1].x = hull[2].x;
	hull[3].y = hull[2].y;
	hull[3].x = hull[0].x;
	hull[4] = r.tl();
	return hull;
}

int main(int argc, char** argv)
{
    const String keys =
        "{help h usage ? |      | print this message }"
        "{thresh tr      | 500  | pixel threshold to be accepted as a 'thing' to be bounded }"
		"{ch convexhull  |      | show convex hull instead of bounding boxes }";
	cv::CommandLineParser parser(argc, argv, keys);
	if(parser.has("help")){
		parser.printMessage();
		exit(EXIT_SUCCESS);
	}
	bool boundingBoxes = !parser.has("ch");
	VideoCapture cap(0);

	if (!cap.isOpened())
		return -1;


	Mat flow, cflow, frame, overlay, dst;
	cap >> frame;
	Mat gray, prevgray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	gray.copyTo(prevgray);
	Mat uflow(frame.size(), CV_32FC2);

	Mat labelImage(frame.size(), CV_32S);
	Mat connected(frame.size(), CV_8UC3);

	// At least hullTresh points for a connected component to be counted
	int hullTresh = parser.get<int>("thresh");
	vector<Point> hull;

	for(;;)
	{
		cap >> frame;
		imshow("webcam", frame);
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		calcOpticalFlowFarneback(prevgray, gray, uflow, 0.25, 3, 15, 3, 5, 1.2, OPTFLOW_USE_INITIAL_FLOW);
		cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
		cvtColor(prevgray, dst, COLOR_GRAY2BGR);

		// Calculate which parts of the image are moving
		overlay = Mat::zeros(cflow.size(), CV_8UC1);
		drawOptFlowMap(uflow, overlay, 1, 255);

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
			if(boundingBoxes){
				Rect r = boundingRect(connectedParts[i]);
				hull = rect2Vect(r);
			}
			else{
				convexHull(connectedParts[i], hull);
			}
			Point cur, last;
			cur = hull[0];
			for (int j = 1; j < hull.size(); j++) {
				last = cur;
				cur = hull[j];
				line(dst, last, cur, colors[i], 2);
			}
		}

		swap(gray, prevgray);
		imshow("flow", dst);
		imshow("flow2", overlay);
		if (waitKey(1) >= 0){
			break;
		}
	}
	return EXIT_SUCCESS;
}

