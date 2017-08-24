static void testImageRectification(cv::Mat &image_original)  
{  
    CV_SHOW(image_original); // CV_SHOW是cv::imshow的一个自定义宏，忽略即可  
    cv::Mat &&image = image_original.clone();  
  
    cv::Mat image_gray;  
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);  
    cv::threshold(image_gray, image_gray, g_threshVal, g_threshMax, cv::THRESH_BINARY);  
  
    std::vector< std::vector<cv::Point> > contours_list;   
    {  
        std::vector<cv::Vec4i> hierarchy;  
        // Since opencv 3.2 source image is not modified by this function  
        cv::findContours(image_gray, contours_list, hierarchy,  
                         cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);  
    }  
      
    for (uint32_t index = 0; index < contours_list.size(); ++index) {  
        cv::RotatedRect &&rect = cv::minAreaRect(contours_list[index]);  
        if (rect.size.area() > 1000) {  
            if (rect.angle != 0.) {  
                // 此处可通过cv::warpAffine进行旋转矫正，本例不需要  
            } //if  
  
            cv::Mat &mask = image_gray;  
            cv::drawContours(mask, contours_list, static_cast<int>(index), cv::Scalar(255), cv::FILLED);  
  
            cv::Mat extracted(image_gray.rows, image_gray.cols, CV_8UC1, cv::Scalar(0));  
            image.copyTo(extracted, mask);  
            CV_SHOW(extracted);  
  
            std::vector<cv::Point2f> poly;  
            cv::approxPolyDP(contours_list[index], poly, 30, true); // 多边形逼近，精度(即最小边长)设为30是为了得到4个角点  
            cv::Point2f pts_src[] = { // 此处顺序调整是为了和后面配对，仅作为示例  
                poly[1],  
                poly[0],  
                poly[3],  
                poly[2]  
            };  
      
            cv::Rect &&r = rect.boundingRect(); // 注意坐标可能超出图像范围  
            cv::Point2f pts_dst[] = {   
                cv::Point(r.x, r.y),  
                cv::Point(r.x + r.width, r.y),  
                cv::Point(r.x + r.width, r.y + r.height) ,  
                cv::Point(r.x, r.y + r.height)  
            };  
            cv::Mat &&M = cv::getPerspectiveTransform(pts_dst, pts_src); // 我这里交换了输入，因为后面指定了cv::WARP_INVERSE_MAP，你可以试试不交换的效果是什么  
  
            cv::Mat warp;cv::warpPerspective(image, warp, M, image.size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);  
            CV_SHOW(warp);  
        } //if  
    }  
}  