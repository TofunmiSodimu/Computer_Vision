//
//  cv2_functions.cpp
//  Motion_detector
//
//  Created by Oluwatofunmi Sodimu on 2/10/25.
//

#include <opencv4/opencv2/core.hpp>

using namespace cv;
using namespace std;

// Works similar to cv2.subtract
Mat subtract_img(Mat img1, Mat img2, int rows, int cols)
{
    Mat diff(rows,cols,img1.type());
    
    // Iterate through images and compute difference between pixel values
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Clip to zero if negative
            diff.at<uchar>(i,j) = max(0,(img2.at<uchar>(i,j) - img1.at<uchar>(i,j)));
        }
    }

    return diff;
}

// Pads images with a constant value
Mat pad_image(Mat img, int rows, int cols, int constant, int kernel_size)
{
    // Pad image to obtain the same image size after filtering
    Mat padded(rows+kernel_size-1, cols+kernel_size-1,img.type(),constant);
    int change = (kernel_size-1)/2;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            padded.at<uchar>(i+change,j+change) = img.at<uchar>(i,j);
        }
    }

    return padded;
}


// Works similar to cv2 median blur
Mat median_blur(Mat img,int rows,int cols)
{
    // Create matrix to store filtered image
    Mat filtered(rows,cols,img.type());

    // Pad image
    int kernel_size = 3;
    int constant = 0.0;
    int change = (kernel_size-1)/2;

    Mat padded = pad_image(img, rows, cols, constant, kernel_size);

    // Iterate through and update pixel value with the median of the kernel
    for (int i = change; i < rows+change; i++)
    {
        for (int j = change; j < cols+change; j++)
        {
            vector<double> kernel_vals;
            for (int k = i-change; k <= i+change; k++)
            {
                for (int l = j-change; l <= j+change; l++)
                {
                    kernel_vals.push_back(padded.at<uchar>(k,l));
                }
            }

            // Sort to get median value
            sort(kernel_vals.begin(), kernel_vals.end());

            // Append median value to filtered matrix
            filtered.at<uchar>(i-change,j-change) = kernel_vals[(kernel_vals.size()-1)/2];
        }
    }
    return filtered;
}


// Similar to cv2 adaptive_thresh_mean_c
Mat thresholding(Mat img,int rows, int cols)
{
    // Pad image
    int neighborhood_size = 15;
    int constant = 0.0;
    Mat padded = pad_image(img, rows, cols, constant, neighborhood_size);

    // Iterate through and update pixel value with the mean of the neighborhood - C
    int C = 3;
    int change = (neighborhood_size-1)/2;
    Mat threshed(rows,cols,img.type(),constant);

    for (int i = change; i < rows+change; i++)
    {
        for (int j = change; j < cols+change; j++)
        {
            int neighborhood_mean = 0;
            for (int k = i-change; k <= i+change; k++)
            {
                for (int l = j-change; l <= j+change; l++)
                {
                    neighborhood_mean += padded.at<uchar>(k,l);
                }
            }
            // Calculate mean
            neighborhood_mean /= (neighborhood_size*neighborhood_size);

            // Use mean as threshold
            if (img.at<uchar>(i-change,j-change) >= (neighborhood_mean - C))
            {
                threshed.at<uchar>(i-change,j-change) = 0.0;
            }
            else
            {
                threshed.at<uchar>(i-change,j-change) = 255.0;
            }
        }
    }
    return threshed;
}

Mat erosion(Mat img,int rows,int cols,int kernel_size)
{
    // Pad image before eroding
    int constant = 255.0;
    Mat padded = pad_image(img, rows, cols, constant, kernel_size);

    Mat eroded(rows,cols,img.type(),constant);
    int change = (kernel_size-1)/2;

    for (int i = change; i < rows+change; i++)
    {
        for (int j = change; j < cols+change; j++)
        {
            int kernel_sum = 0;
            for (int k = i-change; k <= i+change; k++)
            {
                for (int l = j-change; l <= j+change; l++)
                {
                    kernel_sum += padded.at<uchar>(k,l);
                }
            }
            // If all pixels have max values, set pixel intensity to max. Else, set to 0.
            if (kernel_sum == 255.0*kernel_size*kernel_size)
            {
                eroded.at<uchar>(i-change,j-change) = 255.0;
            }
            else
            {
                eroded.at<uchar>(i-change,j-change) = 0.0;
            }
        }
    }

    return eroded;
}

Mat dilation(Mat img, int rows, int cols, int kernel_size)
{
    // Pad image before eroding
    int constant = 0.0;
    Mat padded = pad_image(img, rows, cols, constant, kernel_size);

    Mat dilated(rows,cols,img.type(),constant);
    int change = (kernel_size-1)/2;

    for (int i = change; i < rows+change; i++)
    {
        for (int j = change; j < cols+change; j++)
        {
            int kernel_sum = 0;
            for (int k = i-change; k <= i+change; k++)
            {
                for (int l = j-change; l <= j+change; l++)
                {
                    kernel_sum += padded.at<uchar>(k,l);
                }
            }
            // If at least one of the pixels have max value, set pixel intensity to max. Else, set to 0.
            if (kernel_sum >= 255.0)
            {
                dilated.at<uchar>(i-change,j-change) = 255.0;
            }
            else
            {
                dilated.at<uchar>(i-change,j-change) = 0.0;
            }
        }
    }
    return dilated;
}

Mat closing(Mat img, int rows, int cols)
{
    int kernel_size = 5;

    // Dilate
    Mat closed = dilation(img,rows,cols,kernel_size);

    // Erode
    closed = erosion(closed,rows,cols,kernel_size);

    return closed;
}

// Uses Moore-Neighborhood tracing algorithm
Mat contour_tracing(Mat img, int rows, int cols)
{
    // TO-DO
    return img;
}


Mat final_masking(Mat img, int rows, int cols)
{
    Mat masked(rows,cols,CV_8UC3,0.0);
    int kernel_size = 11;
    Mat padded = pad_image(img,rows,cols,0.0,kernel_size);
    int thresh = ((kernel_size * kernel_size)-1)/2;

    int change = (kernel_size-1)/2;

    for (int i = change; i < rows+change; i++)
    {
        for (int j = change; j < cols+change; j++)
        {
            int kernel_sum = 0;
            for (int k = i-change; k <= i+change; k++)
            {
                for (int l = j-change; l <= j+change; l++)
                {
                    kernel_sum += padded.at<uchar>(k,l);
                }
            }
            // If at least thresh no of the pixels have max value, set pixel intensity to max.
            // Else, leave as is.
            if (kernel_sum >= 255.0*thresh)
            {
                masked.at<Vec3b>(i-change,j-change) = Vec3b(0.0, 0.0, 255.0);
            }
            else
            {
                masked.at<Vec3b>(i-change,j-change) = img.at<Vec3b>(i,j);
            }
        }
    }

    return masked;
}
