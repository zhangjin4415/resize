#include <opencv2/opencv.hpp>
#include <iostream>
// using namespace std;

int main(){
    cv::Mat matSrc, matDst1, matDst2;
    matSrc = cv::imread("im.jpg", 0);//2 | 4
//    std::cout << matSrc;

    matDst1 = cv::Mat(cv::Size(50,50),matSrc.type(), cv::Scalar::all(0));
    matDst2 = cv::Mat(matDst1.size(), matSrc.type(), cv::Scalar::all(0));

	// std::cout << matSrc;
	cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 2);
	// std::cout<< "------------------"<< std::endl;
	// std::cout << matDst2;
	cv::imwrite("cubic_2.jpg", matDst2);

    // double scale_x = (double)matSrc.cols / matDst1.cols;
    // double scale_y = (double)matSrc.rows / matDst1.rows;

    // int iscale_x = cv::saturate_cast<int>(scale_x);
	// int iscale_y = cv::saturate_cast<int>(scale_y);

	// for (int j = 0; j < matDst1.rows; ++j)
	// {
	// 	float fy = (float)((j + 0.5) * scale_y - 0.5);
	// 	int sy = cvFloor(fy);
	// 	fy -= sy;
	// 	sy = std::min(sy, matSrc.rows - 3);
	// 	sy = std::max(1, sy);

	// 	const float A = -0.75f;

	// 	float coeffsY[4];
	// 	coeffsY[0] = ((A*(fy + 1) - 5*A)*(fy + 1) + 8*A)*(fy + 1) - 4*A;
	// 	coeffsY[1] = ((A + 2)*fy - (A + 3))*fy*fy + 1;
	// 	coeffsY[2] = ((A + 2)*(1 - fy) - (A + 3))*(1 - fy)*(1 - fy) + 1;
	// 	coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

	// 	short cbufY[4];
	// 	cbufY[0] = cv::saturate_cast<short>(coeffsY[0] * 2048);
	// 	cbufY[1] = cv::saturate_cast<short>(coeffsY[1] * 2048);
	// 	cbufY[2] = cv::saturate_cast<short>(coeffsY[2] * 2048);
	// 	cbufY[3] = cv::saturate_cast<short>(coeffsY[3] * 2048);

	// 	for (int i = 0; i < matDst1.cols; ++i)
	// 	{
	// 		float fx = (float)((i + 0.5) * scale_x - 0.5);
	// 		int sx = cvFloor(fx);
	// 		fx -= sx;

	// 		if (sx < 1) {
	// 			fx = 0, sx = 1;
	// 		}
	// 		if (sx >= matSrc.cols - 3) {
	// 			fx = 0, sx = matSrc.cols - 3;
	// 		}

	// 		float coeffsX[4];
	// 		coeffsX[0] = ((A*(fx + 1) - 5*A)*(fx + 1) + 8*A)*(fx + 1) - 4*A;
	// 		coeffsX[1] = ((A + 2)*fx - (A + 3))*fx*fx + 1;
	// 		coeffsX[2] = ((A + 2)*(1 - fx) - (A + 3))*(1 - fx)*(1 - fx) + 1;
	// 		coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

	// 		short cbufX[4];
	// 		cbufX[0] = cv::saturate_cast<short>(coeffsX[0] * 2048);
	// 		cbufX[1] = cv::saturate_cast<short>(coeffsX[1] * 2048);
	// 		cbufX[2] = cv::saturate_cast<short>(coeffsX[2] * 2048);
	// 		cbufX[3] = cv::saturate_cast<short>(coeffsX[3] * 2048);

	// 		for (int k = 0; k < matSrc.channels(); ++k)
	// 		{
	// 			matDst1.at<cv::Vec3b>(j, i)[k] = abs((matSrc.at<cv::Vec3b>(sy-1, sx-1)[k] * cbufX[0] * cbufY[0] + matSrc.at<cv::Vec3b>(sy, sx-1)[k] * cbufX[0] * cbufY[1] +
	// 				matSrc.at<cv::Vec3b>(sy+1, sx-1)[k] * cbufX[0] * cbufY[2] + matSrc.at<cv::Vec3b>(sy+2, sx-1)[k] * cbufX[0] * cbufY[3] +
	// 				matSrc.at<cv::Vec3b>(sy-1, sx)[k] * cbufX[1] * cbufY[0] + matSrc.at<cv::Vec3b>(sy, sx)[k] * cbufX[1] * cbufY[1] +
	// 				matSrc.at<cv::Vec3b>(sy+1, sx)[k] * cbufX[1] * cbufY[2] + matSrc.at<cv::Vec3b>(sy+2, sx)[k] * cbufX[1] * cbufY[3] +
	// 				matSrc.at<cv::Vec3b>(sy-1, sx+1)[k] * cbufX[2] * cbufY[0] + matSrc.at<cv::Vec3b>(sy, sx+1)[k] * cbufX[2] * cbufY[1] +
	// 				matSrc.at<cv::Vec3b>(sy+1, sx+1)[k] * cbufX[2] * cbufY[2] + matSrc.at<cv::Vec3b>(sy+2, sx+1)[k] * cbufX[2] * cbufY[3] +
	// 				matSrc.at<cv::Vec3b>(sy-1, sx+2)[k] * cbufX[3] * cbufY[0] + matSrc.at<cv::Vec3b>(sy, sx+2)[k] * cbufX[3] * cbufY[1] +
	// 				matSrc.at<cv::Vec3b>(sy+1, sx+2)[k] * cbufX[3] * cbufY[2] + matSrc.at<cv::Vec3b>(sy+2, sx+2)[k] * cbufX[3] * cbufY[3] ) >> 22);
	// 		}
	// 	}
	// }


	// cv::imwrite("cubic_1.jpg", matDst1);

	// cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 2);
	// std::cout << matDst2;
	// cv::imwrite("cubic_2.jpg", matDst2);
}
