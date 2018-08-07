#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

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

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{help h||}");
	VideoCapture cap(0);

	if (!cap.isOpened())
		return -1;


	Mat flow, cflow, frame, overlay, dst;
	cap >> frame;
	UMat gray, prevgray, uflow;

	Mat labelImage(frame.size(), CV_32S);
	Mat connected(frame.size(), CV_8UC3);

	// At least hullTresh points for a connected component to be counted
	int hullTresh = 50;
	vector<Point> hull;
	namedWindow("flow", 1);

	for (;;)
	{
		cap >> frame;
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		if (!prevgray.empty())
		{
			calcOpticalFlowFarneback(prevgray, gray, uflow, 0.25, 3, 15, 3, 5, 1.2, OPTFLOW_USE_INITIAL_FLOW);
			cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
			uflow.copyTo(flow);
			cvtColor(prevgray, dst, CV_GRAY2BGR);

			// Calculate which parts of the image are moving
			overlay = Mat::zeros(cflow.size(), CV_8UC1);
			drawOptFlowMap(flow, overlay, 1, 255);

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

				convexHull(connectedParts[i], hull);
				Point cur, last;
				cur = hull[0];
				for (int j = 1; j < hull.size(); j++) {
					last = cur;
					cur = hull[j];
					line(dst, last, cur, colors[i], 2);
				}
			}


			imshow("flow", dst);
			imshow("flow2", overlay);
			//imshow("flow2", overlay);
		}
		if (waitKey(1) >= 0)
			break;
		std::swap(prevgray, gray);
	}
	return 0;
}

