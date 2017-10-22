//press ctrl + F5 to run this programme in vs 2015
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Windows.h>
#include <math.h>

#define IMAGE_NUM 6
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
#define TRANS_LEN 1500
#define FRAME_CUT 15
#define VARIANCE_THRESHOLD0 1000
#define VARIANCE_THRESHOLD1 10
#define SEP_LEVEL 20

using namespace cv;
using namespace std;

Mat mergeRows(Mat A, Mat B);
Mat mergeCols(Mat A, Mat B);
Mat transformProcess(Mat srcImage, Mat dstImage, int num);
vector<Point2f> getIntersections(vector<vector<float>> edgelines);
vector<Point2f> getFourCorners(vector<vector<float>> edgelines, Mat srcImage);
vector<vector<float>> getEdgelines(vector<vector<float>> edgelines, vector<Vec2f> lines, Mat srcImage, boolean draw, char dst[]);
Point MatchMain(Mat findImage, Mat modelImage, char store_path[]);
vector <Vec2f> MergeLines(vector<Vec2f> lines, int threshold_r, float threshold_theta, int bubble);
void DrawLine(float rho, float theta, Mat srcImage, char dst[], Scalar s);
vector <Vec2f> BubbleSort(vector<Vec2f> lines, int type);
void PrintLines(vector<Vec2f> lines, int num);
Mat SeperateRGB(Mat srcImage, int num, int low, int high);
vector<Vec2f> VarianceSelectOfLines(vector<Vec2f> lines, float threshold0, float threshold1);
int * SamplingMat(int targetrows, int targetcols, Mat inputImage, int num);
Mat transform(Mat dstImage, Mat srcImage, int num);

int main() {
	printf("\n\n\n##########     ReadingMeters v0.0.9     ##########\n");
	printf("##########  Last updating in 2017/10/22   ##########\n\n\n");

	int pic_num = 1;

	char name[] = "0.jpg";
	char dst[] = "result\\0_result.jpg";
	char dst2[] = "result\\0_aftercut.jpg";
	char dst3[] = "result\\0_bi.jpg";
	char dst4[] = "result\\0_find.jpg";
	char dst5[] = "result\\0_edge2.jpg";
	char dst6[] = "result\\0_pointer.jpg";
	char dst7[] = "result\\0_afterDilate.jpg";

	Mat letterA = imread("A.jpg", 1);

	for (int i = 0; i < IMAGE_NUM; i++) {
		name[0] = 49 + i;
		dst[7] = 49 + i;
		dst2[7] = 49 + i;
		dst3[7] = 49 + i;
		dst4[7] = 49 + i;
		dst5[7] = 49 + i;
		dst6[7] = 49 + i;
		dst7[7] = 49 + i;

		Mat srcImage = imread(name, 1);

		// a track here
		if (i == 5) {
			Mat srcImage1 = srcImage(Range(0, 0.75 * srcImage.rows), Range(0.75 * srcImage.cols, srcImage.cols));
			resize(srcImage1, srcImage, Size(round(3072),round(5500)));
		}
		
			
		Mat dstImage;
		dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());
		
		//----------------------transformProcess begin here!
		dstImage = transformProcess(srcImage, dstImage, i);

		imwrite(dst, dstImage);
		
		Mat modelImage = dstImage(Range(0, 600), Range(0, 600));

		//-------------------------Model Match
		Point letterLoc = MatchMain(modelImage,letterA,dst4);

		//printf("letterLoc:(%d,%d)\n\n", letterLoc.x, letterLoc.y);

		int letterFlag;
		//printf("letter flag:(%d, %d)\n", letterLoc.x, letterLoc.y);
		if (letterLoc.x <= 230 && letterLoc.x >= 220 && letterLoc.y <= 195 && letterLoc.y >= 175)
			letterFlag = 1;
		else
			letterFlag = 0;

		dstImage = dstImage(Range(FRAME_CUT, dstImage.rows- FRAME_CUT), Range(FRAME_CUT, dstImage.cols- FRAME_CUT));

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
		//-------------------Hough Transform
		vector<Vec2f> lines;
		HoughLines(dstImage, lines, 1, CV_PI / 180, HOUGH_THRESHOL_2, 0, 0);
		//-------------------MergeLines
		vector<Vec2f> lines_new = BubbleSort(lines, 0);

		//PrintLines(lines_new, i);

		vector<vector<float>> edgelines_f;
		edgelines_f = getEdgelines(edgelines_f, lines_new, lineImage, true, dst2);

		vector<Vec2f> edgelines_2f;
		for (size_t i = 0; i < edgelines_f.size(); i++) {
			Vec2f el = { edgelines_f[i][0], edgelines_f[i][1] };
			edgelines_2f.push_back(el);
		}

		//PrintLines(edgelines_2f, i);

		vector<Vec2f> edgelines = MergeLines(edgelines_2f, 13, 0.018, 1);
		//PrintLines(edgelines, i);

		//draw line
		for (size_t i = 0; i < edgelines.size(); i++) {
			DrawLine(edgelines[i][0], edgelines[i][1], lineImage, dst6, Scalar(255,0,0));
		}

		float theta = 0;
		float rho = 0;
		for (size_t i = 0; i < edgelines.size(); i++) {
			rho += edgelines[i][0];
			theta += edgelines[i][1];
		}
		theta = theta / edgelines.size();
		rho = rho / edgelines.size();
		DrawLine(rho, theta, lineImage, dst6, Scalar(0,255,0));
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
		
	}
	
	printf("--------END--------\n\n\a\a");

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
	char dstHough[] = "result\\0_hough1.jpg";
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
	//-------------------Histogram-equalization
	equalizeHist(greyImage, dstImage);
	//-------------------Median Filter
	medianBlur(dstImage, dstImage, MEDIAN_BLUR_SIZE);
	Mat samBiImage;
	dstImage.copyTo(samBiImage);
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
	//--------------------sampling
	Mat samBiImage1 = samBiImage(Range(0, samBiImage.rows/8), Range(0, samBiImage.cols));
	Mat samBiImage2 = samBiImage(Range(samBiImage.rows / 8 + 1, samBiImage.rows / 4), Range(0, samBiImage.cols));
	Mat samBiImage3 = samBiImage(Range(samBiImage.rows / 4 + 1, samBiImage.rows/2), Range(0, samBiImage.cols));
	Mat samBiImage4 = samBiImage(Range(samBiImage.rows/2 + 1, samBiImage.rows), Range(0, samBiImage.cols));

	threshold(samBiImage1, samBiImage1, GREY_WHITE*0.01, GREY_WHITE, THRESH_BINARY);
	threshold(samBiImage2, samBiImage2, GREY_WHITE*0.08, GREY_WHITE, THRESH_BINARY);
	threshold(samBiImage3, samBiImage3, GREY_WHITE*0.18, GREY_WHITE, THRESH_BINARY);
	threshold(samBiImage4, samBiImage4, GREY_WHITE*0.20, GREY_WHITE, THRESH_BINARY);

	samBiImage = mergeRows(samBiImage1, samBiImage2);
	samBiImage = mergeRows(samBiImage, samBiImage3);
	samBiImage = mergeRows(samBiImage, samBiImage4);

	int * pt = SamplingMat(300, 300, samBiImage, num);

	int x_array[400];
	int y_array[400];
	int newx[400];
	int x_level[400];

	for (int i = 0; i < 400; i++) {
		x_array[i] = *(pt + i);
		y_array[i] = *(pt + i + 400);
		newx[i] = 0;
		x_level[i] = 0;
	}
	
	for (int i = 1; i < 400; i++) {
		if (x_array[i] < 20000)
			x_array[i] = 0;
		else
			x_array[i] = x_array[i] - 20000;
	}

	for (int i = 0; i < 400; i++) {
		if (x_array[i] > 0)
			x_level[i] = 1;
		else
			x_level[i] = 0;
	}

	for (int i = 0; i < 400; i++) {
		if (x_level[i] == 0)
			break;
		else
			x_level[i] = 0;
	}

	int x_num = 0;
	int cut_before = 0;
	int cut_later = 0;
	int cut_len = 300;
	int cut_center = 0;
	for (int i = 1; i < 400; i++) {
		if (x_level[i - 1] == 0 && x_level[i] == 1)
			x_num++;
		if (x_num % 2 == 0) {
			cut_before = i;
		}
		else if (x_num%2 == 1 && x_num != 1) {
			cut_later = i;
			cut_center = (cut_later + cut_before) / 2;
		}
	}
	
	if(x_num > 2)
		printf("There are %d meters in %d.jpg.\n", num/2, num+1);
	else
		printf("There is only one meter in %d.jpg.\n", num+1);

	Mat transImage;

	if (x_num / 2 == 2) {
		//--------------------------------------------------------------------------------
		Mat dstImage1 = dstImage(Range(0, dstImage.rows), Range(0, (3*dstImage.cols) / 8));
		Mat srcImage1 = srcImage(Range(0, srcImage.rows), Range(0, (3*srcImage.cols) / 8));
		transImage = transform(dstImage1, srcImage1, num);
		//---------------------------------------------------------------------------------
		Mat dstImage2;
		Mat srcImage2 = srcImage(Range(0, srcImage.rows), Range((5 * srcImage.cols) / 8, dstImage.cols));
	}
	else
		transImage = transform(dstImage, srcImage, num);

	return transImage;
}

vector<Point2f> getIntersections(vector<vector<float>> edgelines) {
	//求交点
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
			}
		}
	}
	return intersections;
}

vector<Point2f> getFourCorners(vector<vector<float>> edgelines, Mat srcImage) {
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
		//----------find the edge of plate
		if (!pointer) { 
			//get out of lines which is not frame
			if ((theta >(CV_PI / 24) && theta < ((11 * CV_PI) / 24)) ||
				(theta >(13 * CV_PI / 24) && theta < ((23 * CV_PI) / 24)))
				continue;
		} 
		//----------find the edge of pointer
		else {      
			//get out of lines which in right blow caused by the frame of plate
			if (rho >= 1680 && rho <= 1710 && theta >= 0.76 && theta <= 0.83)
			{
				continue;
			}
		}
		
		//this part is drawing
		if(!pointer)
			DrawLine(rho, theta, srcImage, dst, Scalar(255,255,255));

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
			linesinf.push_back(rho); 
			linesinf.push_back(theta);
			edgelines.push_back(linesinf);
		}
	}
	
	/*if (pointer) {
		printf("\nafter right blow lines:\n");
		for (int i = 0; i < edgelines.size(); i++)
			printf("(%f, %f)\n", edgelines[i][0], edgelines[i][1]);
	}*/
	
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

	return resultPoint;
	//return minPoint;
}
vector <Vec2f> BubbleSort(vector<Vec2f> lines, int type) {
	//sort with compare the first number
	if (type == 0) {
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

		return lines;
	}
	//sort with compare the second number
	else if (type == 1) {
		int bubble_flag = 1;
		for (size_t i = 0; i < lines.size(); i++) {
			if (bubble_flag == 1)
				bubble_flag = 0;
			else
				break;
			for (size_t j = 1; j < lines.size(); j++) {
				if (lines[j - 1][1] > lines[j][1]) {
					float swp1 = lines[j - 1][0], swp2 = lines[j - 1][1];
					lines[j - 1][0] = lines[j][0]; lines[j - 1][1] = lines[j][1];
					lines[j][0] = swp1; lines[j][1] = swp2;
					bubble_flag = 1;
				}
			}
		}

		return lines;
	}
	
}


vector <Vec2f> MergeLines(vector<Vec2f> lines_in, int threshold_r, float threshold_theta, int bubble) {
	//--------------------bubble select
	vector<Vec2f> lines = BubbleSort(lines_in, bubble);
	//------------------re group the lines
	vector<Vec2f> lines_new;
	int part_num = 1;
	//int lines_num = 0;
	for (size_t i = 1; i < lines.size(); i++) {
		float r = lines[i][0];
		float theta = lines[i][1];
		float temp_r = lines[i - 1][0];
		float temp_theta = lines[i - 1][1];
		if (r - temp_r <= threshold_r && abs(theta - temp_theta) <= threshold_theta) {
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

	return lines_new;
}

void DrawLine(float rho, float theta, Mat srcImage, char dst[], Scalar s) {
	Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a*rho, y0 = b*rho;

	pt1.x = cvRound(x0 + HOUGH_LINE_LEN*(-b));
	pt1.y = cvRound(y0 + HOUGH_LINE_LEN*(a));
	pt2.x = cvRound(x0 - HOUGH_LINE_LEN*(-b));
	pt2.y = cvRound(y0 - HOUGH_LINE_LEN*(a));
	line(srcImage, pt1, pt2, s, 2, 4);
	imwrite(dst, srcImage);
}

void PrintLines(vector<Vec2f> lines, int num) {
	printf("<%d.jpg's lines>:\n", num + 1);
	for (size_t i = 0; i < lines.size(); i++) {
		printf("line%d:%f,%f\n", i, lines[i][0], lines[i][1]);
	}
	printf("\n");
}


Mat SeperateRGB(Mat srcImage, int num, int low, int high) {

	Mat dstImage;
	dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());

	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			int sB = srcImage.at<Vec3b>(i, j)[0];
			int sG = srcImage.at<Vec3b>(i, j)[1];
			int sR = srcImage.at<Vec3b>(i, j)[2];
			if (sR >= low && sR <= high && sG >= low && sG <= high && sB >= low && sB <= high) {
				dstImage.at<Vec3b>(i, j)[0] = 255;
				dstImage.at<Vec3b>(i, j)[1] = 255;
				dstImage.at<Vec3b>(i, j)[2] = 255;
			}
			else {
				dstImage.at<Vec3b>(i, j)[0] = 0;
				dstImage.at<Vec3b>(i, j)[1] = 0;
				dstImage.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	return dstImage;
}

vector<Vec2f> VarianceSelectOfLines(vector<Vec2f> lines, float threshold0, float threshold1) {
	if (lines.size() > 3) {
		int total0 = 0;
		int total1 = 0;
		for (int i = 0; i < lines.size(); i++) {
			total0 = total0 + lines[i][0];
			total1 = total1 + lines[i][1];
		}

		int average0 = total0 / lines.size();
		int average1 = total1 / lines.size();
		int variance0 = 0;
		int variance1 = 0;
		int max0 = -1;
		int max1 = -1;
		int maxnum0 = 0;
		int maxnum1 = 0;
 		for (int i = 0; i < lines.size(); i++) {
			int square0 = (lines[i][0] - average0) * (lines[i][0] - average0);
			int square1 = (lines[i][1] - average1) * (lines[i][1] - average1);
			if (square0 > maxnum0) {
				maxnum0 = square0;
				max0 = i;
			}
			if (square1 > maxnum1) {
				maxnum1 = square1;
				max1 = i;
			}
			variance0 = variance0 + square0;
			variance1 = variance1 + square1;
		}

		int flag = 0;
		if (variance0 > threshold0) {
			lines.erase(lines.begin() + max0);
			flag = 1;
		}
		if (variance1 > threshold1) {
			lines.erase(lines.begin() + max1);
			flag = 1;
		}
		if (flag == 1) {
			lines = VarianceSelectOfLines(lines, threshold0, threshold1);
		}
	}
	return lines;
}

int* SamplingMat (int rows, int cols, Mat srcImage, int num) {
	Mat dstImage;
	dstImage.create(rows, cols, srcImage.type());

	int a = srcImage.rows / rows;
	int b = srcImage.cols / cols;
	int array1 [800];
	
	char dstSamp[] = "result\\0_samp.jpg";
	dstSamp[7] = 49 + num;

	imwrite(dstSamp, srcImage);

	for (int i = 0; i < 800; i++) {
		array1[i] = 0;
	}

	for (int x = 0, i = a / 2; i < srcImage.rows && x < rows; i += a, x++) {
		for (int y = 0, j = b / 2; j < srcImage.cols && y < cols; j += b, y++) {
			dstImage.at<uchar>(x, y) = srcImage.at<uchar>(i, j);
			array1[y] = array1[y] + 255 - srcImage.at<uchar>(i, j);
			array1[x+400] = array1[x+400] + 255 - srcImage.at<uchar>(i, j);
		}
	}

	/*printf("x轴：");
	
	for (int i = 0; i < 400; i++) {
		printf("%d\n", array1[i]);
	}
	
	printf("y轴：");

	for (int i = 0; i < 400; i++) {
		printf("%d\n", array1[i+400]);
	}
	*/
	return array1;
}

Mat transform(Mat dstImage, Mat srcImage, int num) {
	char dst[] = "result\\0_beforecut.jpg";
	char dstFind[] = "result\\0_dial.jpg";
	char dstMerge[] = "result\\0_merge.jpg";
	char dstEdge[] = "result\\0_edge.jpg";
	char dstHough[] = "result\\0_hough1.jpg";
	char dstRec[] = "result\\0_rec.jpg";

	dst[7] = 49 + num;
	dstFind[7] = 49 + num;
	dstMerge[7] = 49 + num;
	dstEdge[7] = 49 + num;
	dstHough[7] = 49 + num;
	dstRec[7] = 49 + num;
	//--------------------Edge Detection
	Canny(dstImage, dstImage, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);
	imwrite(dstEdge, dstImage);
	//--------------------Hough Transform
	vector<Vec2f> lines;
	HoughLines(dstImage, lines, 1, CV_PI / 180, HOUGH_THRESHOL, 0, 0);

	for (size_t i = 0; i < lines.size(); i++) {
		DrawLine(lines[i][0], lines[i][1], dstImage, dstHough, Scalar(255, 0, 0));
	}

	//----------------get out of lines which cause buy cuting into 4 pics

	for (size_t i = 0; i < lines.size(); i++) {
		if (((lines[i][0] >= dstImage.rows / 2 - 2 && lines[i][0] <= dstImage.rows / 2 + 2) &&
			lines[i][1] <= (CV_PI / 2 + 0.001) && lines[i][1] >= (CV_PI / 2 - 0.001)) ||
			((lines[i][0] >= dstImage.cols / 2 - 2 && lines[i][0] <= dstImage.cols / 2 + 2) &&
			((lines[i][1] <= (CV_PI + 0.001) && lines[i][1] >= (CV_PI - 0.001))
				|| (lines[i][1] <= 0.001 && lines[i][1] >= -0.001)))) {
			for (size_t j = i; j < lines.size() - 1; j++) {
				lines[j][0] = lines[j + 1][0];
				lines[j][1] = lines[j + 1][1];
			}
			lines.pop_back();
		}
	}

	vector<Vec2f> lines_new = MergeLines(lines, 30, 0.3, 0);

	//------------------print lines:
	//PrintLine();

	vector<vector<float>> edgelines;
	edgelines = getEdgelines(edgelines, lines_new, srcImage, false, dst);

	vector<Point2f> corners_new = getFourCorners(edgelines, srcImage);

	Point2f a = corners_new[0];
	Point2f b = corners_new[1];
	Point2f c = corners_new[2];
	Point2f d = corners_new[3];

	//特殊情况特殊处理，只处理了一种情况（就是表盘的上沿实际上没拍到），后面遇到再加
	if (a.x == 0 && a.y == 0 && b.x == (float)srcImage.cols && b.y == 0) {
		vector <float> linesinf;
		linesinf.push_back(0); linesinf.push_back(CV_PI / 2); linesinf.push_back(0.0);
		edgelines.push_back(linesinf);

		corners_new = getFourCorners(edgelines, srcImage);
		a = corners_new[0];
		b = corners_new[1];
		c = corners_new[2];
		d = corners_new[3];
	}
	//################################################# 补充完整剩下的三种情况 ##################################


	//打印表盘的四边
	line(srcImage, a, b, Scalar(0, 0, 255), 5);
	line(srcImage, b, c, Scalar(0, 0, 255), 5);
	line(srcImage, c, d, Scalar(0, 0, 255), 5);
	line(srcImage, d, a, Scalar(0, 0, 255), 5);

	imwrite(dstRec, srcImage);

	vector<Point2f> corners(4);
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(TRANS_LEN, 0);
	corners[2] = Point2f(TRANS_LEN, TRANS_LEN);
	corners[3] = Point2f(0, TRANS_LEN);

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

	Mat transImage(TRANS_LEN, TRANS_LEN, CV_32FC2);
	warpPerspective(srcImage, transImage, transform, transImage.size());

	return transImage;
}