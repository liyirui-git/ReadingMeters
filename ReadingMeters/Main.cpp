#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Windows.h>

#define IMAGE_NUM 4
#define WINDOW_CHANGABLESIZE 0
#define CANNY_THRESHOLD_1 45
#define CANNY_THRESHOLD_2 90
#define MEDIAN_BLUR_SIZE 7 
#define GREY_WHITE 255
#define GREY_BLACK 0
#define ADP_THRESHOLD_LEN 1111
#define LAP_FLITER_LEN 31  //laplacian requires no more zhan 31.
#define HOUGH_LINE_LEN 5000
#define HOUGH_THRESHOL 160
#define ADJACENT_COEFFICIENT_R 0.30
#define ADJACENT_COEFFICIENT_T 0.08

using namespace cv;
using namespace std;

Mat mergeRows(Mat A, Mat B);
Mat mergeCols(Mat A, Mat B);
Mat transformProcess(Mat srcImage, Mat dstImage);
vector<Point2f> getIntersections(vector<vector<float>> edgelines);
vector<Point2f> getFourTops(vector<vector<float>> edgelines, Mat srcImage);
vector<vector<float>> getEdgelines(vector<vector<float>> edgelines, vector<Vec2f> lines);

int main() {
	printf("\n\n<<<<<<<<<<<<  ReadingMeters v0.0.2  >>>>>>>>>>>>\n\n");

	int pic_num = 1;

	char name[] = "0.jpg";
	char dst[] = "result\\0_result.jpg";
	
	for (int i = 0; i < IMAGE_NUM; i++) {
		name[0] = 49 + i;
		dst[7] = 49 + i;

		Mat srcImage = imread(name, 1);
		Mat dstImage;
		
		//-------------------Cut Image
		/*Mat cutImage1 = srcImage(Range(0, srcImage.rows / 2), Range(0, srcImage.cols / 2));
		Mat cutImage2 = srcImage(Range(0, srcImage.rows / 2), Range(srcImage.cols / 2 + 1, srcImage.cols));
		Mat cutImage3 = srcImage(Range(srcImage.rows / 2 + 1, srcImage.rows), Range(0, srcImage.cols / 2));
		Mat cutImage4 = srcImage(Range(srcImage.rows / 2 + 1, srcImage.rows), Range(srcImage.cols / 2 + 1, srcImage.cols));
		Mat dstImage1, dstImage2, dstImage3, dstImage4;

		dstImage1 = transformProcess(cutImage1, dstImage1);
		dstImage2 = transformProcess(cutImage2, dstImage2);
		dstImage3 = transformProcess(cutImage3, dstImage3);
		dstImage4 = transformProcess(cutImage4, dstImage4);

		Mat dstImageLeft = mergeRows(dstImage1,dstImage3);
		Mat dstImageRight = mergeRows(dstImage2, dstImage4);
		dstImage = mergeCols(dstImageLeft, dstImageRight);
		imwrite("4_merge.jpg", dstImage);
		*/

		dstImage = transformProcess(srcImage, dstImage);
		imwrite(dst, dstImage);

	}

	printf("--------END--------\n\n");

	return 0;
}

Mat mergeRows(Mat A, Mat B) {
	CV_Assert(A.cols == B.cols&&A.type() == B.type());
	int totalRows = A.rows + B.rows;
	Mat mergedDescriptors(totalRows, A.cols, A.type());
	Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);
	return mergedDescriptors;
}

Mat mergeCols(Mat A, Mat B) {
	CV_Assert(A.rows == B.rows&&A.type() == B.type());
	int totalCols = A.cols + B.cols;
	Mat mergedDescriptors(A.rows, totalCols, A.type());
	Mat submat = mergedDescriptors.colRange(0, A.cols);
	A.copyTo(submat);
	submat = mergedDescriptors.colRange(A.cols, totalCols);
	B.copyTo(submat);
	return mergedDescriptors;
}

Mat transformProcess(Mat srcImage, Mat dstImage) {
	printf("\n");
	Mat greyImage;
	//-------------------Greyed
	cvtColor(srcImage, greyImage, CV_BGR2GRAY);
	//-------------------Histogram-equalization
	equalizeHist(greyImage, dstImage);
	//-------------------Median Filter
	medianBlur(dstImage, dstImage, MEDIAN_BLUR_SIZE);

	//-------------------Image Binarization
	/*threshold(dstImage, dstImage, GREY_WHITE*0.24, GREY_WHITE, THRESH_BINARY);
	//medianBlur(dstImage, dstImage, 11);
	//adaptiveThreshold(dstImage, dstImage, GREY_WHITE, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
	//LAP_FLITER_LEN, 1);
	/*medianBlur(biImage, dstImage, 11);
	imwrite("3_bi.jpg", dstImage);
	//Mat biImage = dstImage;
	*/
	//--------------------Edge Detection
	Canny(dstImage, dstImage, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);

	//--------------------Laplacian Edge
	/*Laplacian(dstImage, dstImage, CV_8U, LAP_FLITER_LEN);
	medianBlur(dstImage, dstImage, 11);
	imwrite("1_lap.jpg", dstImage);
	*/

	//--------------------convex
	/*vector<vector<Point>> g_vContours;
	vector<Vec4i> g_vHierarchy;
	findContours(dstImage, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point>> hull(g_vContours.size());
	for (unsigned int i = 0; i < g_vContours.size(); i++) {
	convexHull(Mat(g_vContours[i]), hull[i], false);
	}

	//Mat drawing = Mat::zeros(dstImage.size(), CV_8UC3);
	Mat drawing =biImage;
	RNG g_rng(12345);
	for (unsigned int i = 0; i < g_vContours.size(); i++) {
	Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
	drawContours(drawing, g_vContours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}
	*/

	//--------------------Hough Transform
	vector<Vec2f> lines;
	HoughLines(dstImage, lines, 1, CV_PI / 180, HOUGH_THRESHOL, 0, 0);
	

	//--------------------bubble select (in the order of "rho")
	int bubble_flag = 1;
	for (size_t i = 0; i < lines.size(); i++) {
		if (bubble_flag == 1)
			bubble_flag = 0;
		else
			break;
		for (size_t j = 1; j < lines.size(); j++) {
			if (lines[j - 1][0] > lines[j][0]) {
				float swp1 = lines[j - 1][0], swp2 = lines[j - 1][1];
				lines[j - 1][0] = lines[j][0]; lines[j - 1][1] = lines[j][1];
				lines[j][0] = swp1; lines[j][1] = swp2;
				bubble_flag = 1;
			}
		}
	}

	vector<vector<float>> edgelines;
	edgelines = getEdgelines(edgelines, lines);
	
	//imwrite("2_hough.jpg", srcImage);

	vector<Point2f> corners_new = getFourTops(edgelines, srcImage);

	Point2f a = corners_new[0];
	Point2f b = corners_new[1];
	Point2f c = corners_new[2];
	Point2f d = corners_new[3];

	//特殊情况特殊处理，只处理了一种情况，后面遇到再加
	if (a.x == 0 && a.y == 0 && b.x == (float)srcImage.cols && b.y == 0) {
		vector <float> linesinf;
		linesinf.push_back(0); linesinf.push_back(CV_PI/2); linesinf.push_back(0.0);
		edgelines.push_back(linesinf);

		corners_new = getFourTops(edgelines, srcImage);
		a = corners_new[0];
		b = corners_new[1];
		c = corners_new[2];
		d = corners_new[3];
	}

	line(srcImage, a, b, Scalar(0, 0, 255),3); printf("a:(%f,%f)\n", a.x, a.y);
	line(srcImage, b, c, Scalar(0, 0, 255),3); printf("b:(%f,%f)\n", b.x, b.y);
	line(srcImage, c, d, Scalar(0, 0, 255),3); printf("c:(%f,%f)\n", c.x, c.y);
	line(srcImage, d, a, Scalar(0, 0, 255),3); printf("d:(%f,%f)\n", d.x, d.y);
	//line(srcImage, Point(0, 0), Point(srcImage.cols, srcImage.rows), Scalar(0, 255, 0));

	//return srcImage;

	float trans_len = ((b.x - a.x) > (c.x - d.x)) ? (b.x - a.x) : (c.x - d.x);
	float trans_hei = ((d.y - a.y) > (c.y - b.y)) ? (d.y - a.y) : (c.y - b.y);
	vector<Point2f> corners(4);
	corners[0] = Point2f(0,0);
	corners[1] = Point2f(trans_len,0);
	corners[2] = Point2f(trans_len,trans_hei);
	corners[3] = Point2f(0,trans_hei);

	corners_new[0] = (Point2f)a; 
	corners_new[1] = (Point2f)b; 
	corners_new[2] = (Point2f)c; 
	corners_new[3] = (Point2f)d;

	//printf("old:(%d,%d),new:(%d,%d)\n", corners[3].x, corners[3].y, corners_new[3].x, corners_new[3].y);

	Mat transform = getPerspectiveTransform(corners_new, corners);

	vector<Point2f> points, points_trans;
	for (int i = 0; i<srcImage.rows; i++) {
		for (int j = 0; j<srcImage.cols; j++) {
			points.push_back(Point2f(j, i));
		}
	}

	//perspectiveTransform(points, points_trans, transform);

	Mat transImage(trans_hei, trans_len, CV_32FC2);
	warpPerspective(srcImage, transImage, transform, transImage.size());

	return transImage;
}

vector<Point2f> getIntersections(vector<vector<float>> edgelines) {
	vector<Point2f> intersections;
	for (size_t i = 0; i < edgelines.size(); i++) {
		for (size_t j = i + 1; j < edgelines.size(); j++) {
			if (edgelines[i][2] != edgelines[j][2]) {
				float r1 = edgelines[i][0], r2 = edgelines[j][0];
				float theta1 = edgelines[i][1], theta2 = edgelines[j][1];
				double a1 = cos(theta1), b1 = sin(theta1);
				double a2 = cos(theta2), b2 = sin(theta2);
				double x = (b1*r2 - b2*r1) / (a2*b1 - a1*b2);
				double y = (a1*r2 - a2*r1) / (a1*b2 - a2*b1);
				Point2f p = Point2f((float)x, (float)y);
				intersections.push_back(p);
				//printf("(%f,%f)\n", p.x, p.y);
			}
		}
	}
	return intersections;
}

vector<Point2f> getFourTops(vector<vector<float>> edgelines, Mat srcImage) {
	//求出所有直线交点
	vector<Point2f> intersections = getIntersections(edgelines);

	Point2f a = Point2f(0, 0), b = Point2f(srcImage.cols, 0),
		c = Point2f(srcImage.cols, srcImage.rows), d = Point2f(0, srcImage.rows);

	int total = srcImage.rows + srcImage.cols;
	float len_a = total, len_b = total, len_c = total, len_d = total;

	for (int i = 0; i < intersections.size(); i++) {
		Point2f p = intersections[i];
		if (p.x <= (srcImage.cols / 2) && p.y <= (srcImage.rows / 2)) {
			float len = sqrt(p.x*p.x + p.y*p.y);
			if (len < len_a) {
				len_a = len;
				a = p;
				//printf("<a>:(%f,%f),<p>:(%f,%f)\n",a.x,a.y,p.x,p.y);
			}
		}
		else if (p.x >(srcImage.cols / 2) && p.y < (srcImage.rows / 2)) {
			float len = sqrt((p.x - srcImage.cols)*(p.x - srcImage.cols) + p.y*p.y);
			if (len < len_b) {
				len_b = len;
				b = p;
			}
		}
		else if (p.x >= (srcImage.cols / 2) && p.y >= (srcImage.rows / 2)) {
			float len = sqrt((p.x - srcImage.cols)*(p.x - srcImage.cols)
				+ (p.y - srcImage.rows)*(p.y - srcImage.rows));
			if (len < len_c) {
				len_c = len;
				c = p;
			}
		}
		else if (p.x < (srcImage.cols / 2) && p.y >(srcImage.rows / 2)) {
			float len = sqrt(p.x*p.x + (p.y - srcImage.rows)*(p.y - srcImage.rows));
			if (len < len_d) {
				len_d = len;
				d = p;
			}
		}
	}

	vector<Point2f> corners_new(4);
	corners_new[0] = (Point2f)a;
	corners_new[1] = (Point2f)b;
	corners_new[2] = (Point2f)c;
	corners_new[3] = (Point2f)d;

	return corners_new;
}

vector<vector<float>> getEdgelines(vector<vector<float>> edgelines, vector<Vec2f> lines) {
	float rho_f = 0.0, theta_f = 0.0;
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		//get out of nearby lines
		if (rho_f == 0.0 && theta_f == 0.0) {}
		else if (((rho_f*(1 + ADJACENT_COEFFICIENT_R)) >= rho && (rho_f*(1 - ADJACENT_COEFFICIENT_R)) <= rho)
			&& ((theta_f*(1 + ADJACENT_COEFFICIENT_T)) >= theta && (theta_f*(1 - ADJACENT_COEFFICIENT_T)) <= theta)) {
			continue;
		}
		rho_f = rho; theta_f = theta;
		//get out of lines which is not frame
		if ((theta >(CV_PI / 24) && theta < ((11 * CV_PI) / 24)) ||
			(theta >(13 * CV_PI / 24) && theta < ((23 * CV_PI) / 24)))
			continue;
		//printf("\trho:%f,\ttheta:%f\n", rho, theta);

		//----------------draw the line
		/*Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + HOUGH_LINE_LEN*(-b));
		pt1.y = cvRound(y0 + HOUGH_LINE_LEN*(a));
		pt2.x = cvRound(x0 - HOUGH_LINE_LEN*(-b));
		pt2.y = cvRound(y0 - HOUGH_LINE_LEN*(a));
		line(srcImage, pt1, pt2, Scalar(255, 255, 255), 1, 4);
		*/

		//store the line which is qualified
		float type;
		if (theta <= (CV_PI / 24) || theta >= ((23 * CV_PI) / 24))
			type = 1.0;
		else
			type = 0.0;
		vector <float> linesinf;
		linesinf.push_back(rho); linesinf.push_back(theta); linesinf.push_back(type);
		edgelines.push_back(linesinf);
	}
	return edgelines;
}