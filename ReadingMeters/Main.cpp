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
#define CANNY_THRESHOLD_3 40
#define CANNY_THRESHOLD_4 75
#define MEDIAN_BLUR_SIZE 11 
#define GREY_WHITE 255
#define GREY_BLACK 0
#define ADP_THRESHOLD_LEN 1111
#define LAP_FLITER_LEN 31  //laplacian requires no more zhan 31.
#define HOUGH_LINE_LEN 5000
#define HOUGH_THRESHOL 120
#define HOUGH_THRESHOL_2 140
#define ADJACENT_COEFFICIENT_R 0.30
#define ADJACENT_COEFFICIENT_T 0.08

using namespace cv;
using namespace std;

Mat mergeRows(Mat A, Mat B);
Mat mergeCols(Mat A, Mat B);
Mat transformProcess(Mat srcImage, Mat dstImage, int num);
vector<Point2f> getIntersections(vector<vector<float>> edgelines);
vector<Point2f> getFourTops(vector<vector<float>> edgelines, Mat srcImage);
vector<vector<float>> getEdgelines(vector<vector<float>> edgelines, vector<Vec2f> lines, Mat srcImage, boolean draw, char dst[]);
Point MatchMain(Mat findImage, Mat modelImage, char store_path[]);

int main() {
	printf("\n\n\n##########     ReadingMeters v0.0.4     ##########\n");
	printf("##########  Last updating in 2017/9/19  ##########\n\n\n");

	int pic_num = 1;

	char name[] = "0.jpg";
	char dst[] = "result\\0_result.jpg";
	char dst2[] = "result\\0_aftercut.jpg";
	char dst3[] = "result\\0_bi.jpg";
	char dst4[] = "result\\0_find.jpg";
	char dst5[] = "result\\0_edge2.jpg";

	Mat letterA = imread("A.jpg", 1);

	for (int i = 0; i < IMAGE_NUM; i++) {
		name[0] = 49 + i;
		dst[7] = 49 + i;
		dst2[7] = 49 + i;
		dst3[7] = 49 + i;
		dst4[7] = 49 + i;
		dst5[7] = 49 + i;

		Mat srcImage = imread(name, 1);
		Mat dstImage;
		
		dstImage = transformProcess(srcImage, dstImage, i);

		imwrite(dst, dstImage);
		
		//-------------------------Model Match
		Point letterLoc = MatchMain(dstImage,letterA,dst4);

		int letterFlag;
		if (letterLoc.x < (dstImage.cols / 3) && letterLoc.y < (dstImage.rows / 3))
			letterFlag = 1;
		else
			letterFlag = 0;

		dstImage = dstImage(Range(dstImage.rows/16, (15 * dstImage.rows) / 16), 
			Range(dstImage.cols / 16, (15 * dstImage.cols) / 16));

		Mat lineImage = dstImage;
		//-------------------Greyed
		cvtColor(dstImage, dstImage, CV_BGR2GRAY);
		//-------------------Histogram-equalization
		equalizeHist(dstImage, dstImage);
		//-------------------Median Filter
		medianBlur(dstImage, dstImage, MEDIAN_BLUR_SIZE + 8);
		//-------------------binarization
		threshold(dstImage, dstImage, GREY_WHITE*0.1, GREY_WHITE, THRESH_BINARY);
		imwrite(dst3, dstImage);
		//--------------------Edge Detection
		Canny(dstImage, dstImage, CANNY_THRESHOLD_3, CANNY_THRESHOLD_4);
		imwrite(dst5, dstImage);

		vector<Vec2f> lines;
		HoughLines(dstImage, lines, 1, CV_PI / 180, HOUGH_THRESHOL_2, 0, 0);

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
		edgelines = getEdgelines(edgelines, lines, lineImage, true, dst2);

		float theta = 0;
		for (size_t i = 0; i < edgelines.size(); i++) {
			theta += edgelines[i][1];
		}
		theta = theta / edgelines.size();
		float rate = ((2 * theta) / CV_PI) - 1;

		int range;
		if (letterFlag == 0) {
			range = 450;
			printf("%d.jpg-result: %.2f V\n\n", i+1, range*rate);
		}
		else if (letterFlag == 1) {
			range = 150;
			printf("%d.jpg-result: %.2f A\n\n", i+1, range*rate);
		}
			
		//imwrite(dst2, lineImage);
	}

	printf("--------END--------\n\n\a");

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

Mat transformProcess(Mat srcImage, Mat dstImage, int num) {
	printf("\n");
	char dst[] = "result\\0_beforecut.jpg";
	char dstFind[] = "result\\0_dial.jpg";
	char dstMerge[] = "result\\0_merge.jpg";
	char dstEdge[] = "result\\0_edge.jpg";
	char dstHough[] = "result\\0_hough.jpg";
	char dstRec[] = "result\\0_rec.jpg";
	dst[7] = 49 + num;
	dstFind[7] = 49 + num;
	dstMerge[7] = 49 + num;
	dstEdge[7] = 49 + num;
	dstHough[7] = 49 + num;
	dstRec[7] = 49 + num;
	
	Mat greyImage;
	//-------------------Greyed
	cvtColor(srcImage, greyImage, CV_BGR2GRAY);
	//-------------------Model Match
	Mat dialImage = imread("model.jpg",1);
	Point dialLoc = MatchMain(srcImage, dialImage, dstFind);
	//-------------------Histogram-equalization
	equalizeHist(greyImage, dstImage);
	//-------------------Median Filter
	medianBlur(dstImage, dstImage, MEDIAN_BLUR_SIZE);
	//-------------------cut into 4pics & binary
	Mat dstImage1 = dstImage(Range(0, dstImage.rows / 2), Range(0, dstImage.cols / 2));
	Mat dstImage2 = dstImage(Range(0, dstImage.rows / 2), Range(dstImage.cols / 2 + 1, dstImage.cols));
	Mat dstImage3 = dstImage(Range(dstImage.rows / 2 + 1, dstImage.rows), Range(0, dstImage.cols / 2));
	Mat dstImage4 = dstImage(Range(dstImage.rows / 2 + 1, dstImage.rows), Range(dstImage.cols / 2 + 1, dstImage.cols));
	
	threshold(dstImage1, dstImage1, GREY_WHITE*0.55, GREY_WHITE, THRESH_BINARY);
	threshold(dstImage2, dstImage2, GREY_WHITE*0.6, GREY_WHITE, THRESH_BINARY);
	threshold(dstImage3, dstImage3, GREY_WHITE*0.75, GREY_WHITE, THRESH_BINARY);
	threshold(dstImage4, dstImage4, GREY_WHITE*0.75, GREY_WHITE, THRESH_BINARY);

	Mat dstImageLeft = mergeRows(dstImage1, dstImage3);
	Mat dstImageRight = mergeRows(dstImage2, dstImage4);
	dstImage = mergeCols(dstImageLeft, dstImageRight);
	imwrite(dstMerge, dstImage);
	//--------------------Edge Detection
	Canny(dstImage, dstImage, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);
	imwrite(dstEdge, dstImage);
	//--------------------Hough Transform
	vector<Vec2f> lines;
	HoughLines(dstImage, lines, 1, CV_PI / 180, HOUGH_THRESHOL, 0, 0);
	imwrite(dstHough, dstImage);
	//----------------get out of lines which cause buy cuting into 4 pics

	for (size_t i = 0; i < lines.size(); i++) {
		if (((lines[i][0] >= dstImage.rows/2-2 && lines[i][0] <= dstImage.rows/2 + 2) &&
			lines[i][1] <= (CV_PI / 2 + 0.001) && lines[i][1] >= (CV_PI / 2 - 0.001)) ||
			((lines[i][0] >= dstImage.cols - 2 || lines[i][0] <= dstImage.cols + 2) &&
				(lines[i][1] <= (CV_PI + 0.001) && lines[i][1] >= (CV_PI - 0.001))
				||(lines[i][1] <= 0.001 && lines[i][1] >= -0.001))) {
			for (size_t j = i; j < lines.size()-1; j++) {
				lines[j][0] = lines[j + 1][0];
				lines[j][1] = lines[j + 1][1];
			}
			lines.pop_back();
		}
	}
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
	//------------------re group the lines
	vector<Vec2f> lines_new;
	int part_num = 1;
	//int lines_num = 0;
	for (size_t i = 1; i < lines.size(); i++) {
		float r = lines[i][0];
		float theta = lines[i][1];
		float temp_r = lines[i - 1][0];
		float temp_theta = lines[i - 1][1];
		if (r - temp_r <= 30 && abs(theta - temp_theta) <= 0.03){
			part_num++;
		}
		else {
			float total_r = 0, total_theta = 0;
			for (size_t j = i - part_num, k = 0; k < part_num; j++, k++) {
				total_r = total_r + lines[j][0];
				total_theta = total_theta + lines[j][1];
			}
			Vec2f content = { total_r / part_num , total_theta / part_num };
			lines_new.push_back(content);
			part_num = 1;
		}

		if (i == lines.size() - 1) {
			if (part_num == 1) {
				Vec2f content = { r, theta };
				lines_new.push_back(content);
			}
			else {
				float total_r = 0, total_theta = 0;
				for (size_t j = i - part_num + 1, k = 0; k < part_num; j++, k++) {
					total_r = total_r + lines[j][0];
					total_theta = total_theta + lines[j][1];
				}
				Vec2f content = { total_r / part_num, total_theta / part_num };
				lines_new.push_back(content);
			}
		}
	}


	//------------------print lines:
	printf("<%d.jpg's lines>:\n", num+1);
	for (size_t i = 0; i < lines.size(); i++) {
		printf("line%d:%f,%f\n", i, lines[i][0], lines[i][1]);
	}
	printf("\n");

	printf("<%d.jpg's new lines>:\n", num + 1);
	for (size_t i = 0; i < lines_new.size(); i++) {
		printf("line%d:%f,%f\n", i, lines_new[i][0], lines_new[i][1]);
	}
	printf("\n");

	vector<vector<float>> edgelines;
	edgelines = getEdgelines(edgelines, lines_new, srcImage, false,dst);

	vector<Point2f> corners_new = getFourTops(edgelines, srcImage);

	Point2f a = corners_new[0];
	Point2f b = corners_new[1];
	Point2f c = corners_new[2];
	Point2f d = corners_new[3];

	//特殊情况特殊处理，只处理了一种情况（就是表盘的上沿实际上没拍到），后面遇到再加
	if (a.x == 0 && a.y == 0 && b.x == (float)srcImage.cols && b.y == 0) {
		vector <float> linesinf;
		linesinf.push_back(0); linesinf.push_back(CV_PI / 2); linesinf.push_back(0.0);
		edgelines.push_back(linesinf);

		corners_new = getFourTops(edgelines, srcImage);
		a = corners_new[0];
		b = corners_new[1];
		c = corners_new[2];
		d = corners_new[3];
	}
	//################################################# 补充完整剩下的三种情况 ##################################



	//打印表盘的四边
	line(srcImage, a, b, Scalar(0, 0, 255), 3);
	line(srcImage, b, c, Scalar(0, 0, 255), 3); 
	line(srcImage, c, d, Scalar(0, 0, 255), 3);
	line(srcImage, d, a, Scalar(0, 0, 255), 3);

	imwrite(dstRec, srcImage);

	float trans_len = ((b.x - a.x) > (c.x - d.x)) ? (b.x - a.x) : (c.x - d.x);
	float trans_hei = trans_len;

	vector<Point2f> corners(4);
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(trans_len, 0);
	corners[2] = Point2f(trans_len, trans_hei);
	corners[3] = Point2f(0, trans_hei);

	corners_new[0] = (Point2f)a;
	corners_new[1] = (Point2f)b;
	corners_new[2] = (Point2f)c;
	corners_new[3] = (Point2f)d;

	Mat transform = getPerspectiveTransform(corners_new, corners);

	vector<Point2f> points, points_trans;
	for (int i = 0; i<srcImage.rows; i++) {
		for (int j = 0; j<srcImage.cols; j++) {
			points.push_back(Point2f(j, i));
		}
	}

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

vector<vector<float>> getEdgelines(vector<vector<float>> edgelines, vector<Vec2f> lines,
	Mat srcImage, boolean pointer,char dst[]) {

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];

		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;

		if (!pointer) { //对表盘找线
			//get out of lines which is not frame
			if ((theta >(CV_PI / 24) && theta < ((11 * CV_PI) / 24)) ||
				(theta >(13 * CV_PI / 24) && theta < ((23 * CV_PI) / 24)))
				continue;
			//printf("\trho:%f,\ttheta:%f\n", rho, theta);
		} 
		else {      //对指针找线
			if ((((rho - a*srcImage.cols) / b) >= srcImage.rows 
				|| ((rho - a*srcImage.cols) / b) <= (srcImage.rows * 5 / 6))
				&& (((rho - b*srcImage.rows) / a) >= srcImage.cols 
					|| ((rho - b*srcImage.rows) / a) <= (srcImage.cols * 5 / 6)))
				continue;
		}
		
		//this part is drawing
		pt1.x = cvRound(x0 + HOUGH_LINE_LEN*(-b));
		pt1.y = cvRound(y0 + HOUGH_LINE_LEN*(a));
		pt2.x = cvRound(x0 - HOUGH_LINE_LEN*(-b));
		pt2.y = cvRound(y0 - HOUGH_LINE_LEN*(a));
		line(srcImage, pt1, pt2, Scalar(255, 255, 255), 1, 4);
		imwrite(dst, srcImage);

		//store the line which is qualified
		if (!pointer) {
			float type;
			if (theta <= (CV_PI / 24) || theta >= ((23 * CV_PI) / 24))
				type = 1.0;
			else
				type = 0.0;
			vector <float> linesinf;
			linesinf.push_back(rho); linesinf.push_back(theta); linesinf.push_back(type);
			edgelines.push_back(linesinf);
		}
		else {
			vector <float> linesinf;
			linesinf.push_back(rho); linesinf.push_back(theta);
			edgelines.push_back(linesinf);
		}
	}
	return edgelines;

}

Point MatchMain(Mat findImage, Mat modeImage, char store_path[])
{
	Mat dstImage;
	dstImage.create(findImage.rows - modeImage.rows + 1, findImage.cols - modeImage.cols + 1, CV_32FC1);

	//进行模版匹配（归一化平方差匹配法）  
	matchTemplate(findImage, modeImage, dstImage, TM_SQDIFF_NORMED);
	normalize(dstImage, dstImage, 0, 1, 32);

	//首先是从得到的 输出矩阵中得到 最大或最小值（归一化平方差匹配方式是越小越好，所以在这种方式下，找到最小位置）  
	Point minPoint;
	minMaxLoc(dstImage, 0, 0, &minPoint, 0);

	//绘制  
	rectangle(findImage, minPoint, Point(minPoint.x + modeImage.cols, minPoint.y + modeImage.rows)
		, Scalar(theRNG().uniform(0, 255), theRNG().uniform(0, 255), theRNG().uniform(0, 255)), 3, 8);
	imwrite(store_path, findImage);

	Point resultPoint = Point((minPoint.x + modeImage.cols/2), (minPoint.y + modeImage.rows/2));

	//circle(findImage, resultPoint, 5, Scalar(theRNG().uniform(0, 255), theRNG().uniform(0, 255), theRNG().uniform(0, 255)), 3, 8);

	return resultPoint;
	//return minPoint;
}