#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include <Windows.h>

#define WINDOW_CHANGABLESIZE 0
#define CANNY_THRESHOLD_1 45
#define CANNY_THRESHOLD_2 90
#define MEDIAN_BLUR_SIZE 7

using namespace std;
using namespace cv;

void preprocess(void) {

}

int main() {
	printf("\n\n<<<<<<<<<<<<  ReadingMeters v0.0.1  >>>>>>>>>>>>\n\n");

	int pic_num = 1;

	Mat srcImage = imread("3.jpg", 1);

	//Greyed
	Mat greyImage;
	Mat dstImage;

	cvtColor(srcImage, greyImage, CV_BGR2GRAY);
	//Histogram-equalization
	equalizeHist(greyImage, dstImage);
	//Median Filter
	medianBlur(dstImage, dstImage, MEDIAN_BLUR_SIZE);
	//Edge Detection
	Canny(dstImage, dstImage, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);

	imwrite("3_Canny.jpg", dstImage);
	
	printf("-END-\n\n");

	return 0;
}

