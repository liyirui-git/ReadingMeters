#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Windows.h>

#define WINDOW_CHANGABLESIZE 0
#define CANNY_THRESHOLD_1 45
#define CANNY_THRESHOLD_2 80
#define MEDIAN_BLUR_SIZE 7
#define GREY_WHITE 255
#define GREY_BLACK 0
#define ADP_THRESHOLD_LEN 1111
#define LAP_FLITER_LEN 31  //laplacian requires no more zhan 31.
#define HOUGH_LINE_LEN 5000
#define HOUGH_THRESHOL 160

using namespace cv;
using namespace std;

void preprocess(void) {

}

int main() {
	printf("\n\n<<<<<<<<<<<<  ReadingMeters v0.0.1  >>>>>>>>>>>>\n\n");

	int pic_num = 1;

	Mat srcImage = imread("4.jpg", 1);
	Mat greyImage, dstImage, biImage;
	//-------------------Greyed
	cvtColor(srcImage, greyImage, CV_BGR2GRAY);
	//-------------------Histogram-equalization
	equalizeHist(greyImage, dstImage);
	//-------------------Median Filter
	medianBlur(dstImage, dstImage, MEDIAN_BLUR_SIZE);
	//-------------------Image Binarization
	//threshold(dstImage, biImage, GREY_WHITE*1/4, GREY_WHITE, THRESH_BINARY);
	/*adaptiveThreshold(dstImage, biImage, GREY_WHITE, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 
		LAP_FLITER_LEN, 1);
	medianBlur(biImage, dstImage, 11);
	
	imwrite("3_bi.jpg", dstImage);
	*/
	//--------------------Edge Detection
	Canny(dstImage, dstImage, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);
	//Canny(biImage, dstImage, CANNY_THRESHOLD_1*2, CANNY_THRESHOLD_2*2);
	//bitwise_not(dstImage, dstImage);
	imwrite("4_Canny.jpg", dstImage);
	/*
	Laplacian(dstImage, dstImage, CV_8U, LAP_FLITER_LEN);
	medianBlur(dstImage, dstImage, 11);
	imwrite("1_lap.jpg", dstImage);
	*/
	
	//--------------------Hough Transform
	vector<Vec2f> lines;
	//Mat houghImage;
	HoughLines(dstImage, lines, 1, CV_PI/180, HOUGH_THRESHOL, 0, 0);

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + HOUGH_LINE_LEN*(-b));
		pt1.y = cvRound(y0 + HOUGH_LINE_LEN*(a));
		pt2.x = cvRound(x0 - HOUGH_LINE_LEN*(-b));
		pt2.y = cvRound(y0 - HOUGH_LINE_LEN*(a));
		line(srcImage, pt1, pt2, Scalar(255, 255, 255), 1, LINE_AA);
	}

	imwrite("4_hough.jpg", srcImage);


	printf("--------END--------\n\n");

	return 0;
}

