#include <string>
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>

//using namespace std;

int main()
{
	IplImage* img = 0;
	cvNamedWindow("win", CV_WINDOW_AUTOSIZE);
	img = cvLoadImage("5.png");
	if (!img) printf("Can't load image.\n");
	cvShowImage("win", img);

	cvWaitKey();

	cvDestroyWindow("win");
	cvReleaseImage(&img);

	return 0;
}