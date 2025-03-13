//
//  cv2_functions.hpp
//  Motion_detector
//
//  Created by Oluwatofunmi Sodimu on 2/10/25.
//

#ifndef cv2_functions_hpp
#define cv2_functions_hpp

#include <stdio.h>
#include <opencv4/opencv2/core.hpp>

using namespace cv;

Mat subtract_img(Mat img1, Mat img2, int rows, int cols);
Mat pad_image(Mat img, int rows, int cols, int constant, int kernel_size);
Mat median_blur(Mat img,int rows,int cols);
Mat thresholding(Mat img,int rows, int cols);
Mat erosion(Mat img,int rows,int cols,int kernel_size);
Mat dilation(Mat img, int rows, int cols, int kernel_size);
Mat closing(Mat img, int rows, int cols);
Mat contour_tracing(Mat img, int rows, int cols);
Mat final_masking(Mat img, int rows, int cols);

#endif /* cv2_functions_hpp */
