#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <iostream>
#include "cv2_functions.hpp"

using namespace std;
using namespace cv;

// Declare Variables
int counter = 0; // Counter to keep track of if we have 2 images to subtract
int video_not_started = 0;
VideoWriter vid_writer;
Mat gray_img1;
Mat color_img1;
Mat gray_img2;
Mat color_img2;

int main(int argc, char** argv) 
{
    /* Step 1: Get each frame from video and convert to grayscale*/
	// Check that correct number of command-line arguments passed
	if (argc != 2) 
	{ 
		printf("usage should be: ./motion_detector <Video_Path>\n"); 
		return -1;
	}

    // Open video file
    VideoCapture vid_reader(argv[1]); 
        
    // Check if file opened successfully
    if(!vid_reader.isOpened()){
        cout << "File not opened." << endl;
        return -1;
    }

    while(1)
    {
        if (counter < 2)
        {
            // Read current frame from video
            Mat curr_frame;
            vid_reader >> curr_frame;
        
            // Break if curr_frame is empty or at end of video
            if (curr_frame.empty())
            {
                break;
            }

            // Ensure curr_frame meets size requirement
            if (curr_frame.rows > 450 | curr_frame.cols > 450)
            {
                cout << "Video frames too large." << endl;
                return -1;
            }

            // Convert curr_frame to grayscale
            if (gray_img1.empty())
            {
                color_img1 = curr_frame;
                cvtColor(curr_frame, gray_img1, COLOR_BGR2GRAY);
            }
            else
            {
                color_img2 = curr_frame;
                cvtColor(curr_frame, gray_img2, COLOR_BGR2GRAY);
            }

            // Update counter
            counter++; 

        }
        else
        {
            // Check if frames are the same size (a properly encoded video should have frames that are always the same size)
            if (gray_img1.size() != gray_img2.size())
            {
                // Skip first frame
                gray_img1 = gray_img2;
                color_img1 = color_img2;
                gray_img2.release();
                color_img2.release();
                continue;
            }
            
            /* Step 2: Compute frame difference */
            int rows = gray_img1.rows;
            int cols = gray_img1.cols;
            Mat diff_img = subtract_img(gray_img1,gray_img2,rows,cols);
       
            /* Step 3: Remove noise and create a motion mask */
            // Remove noise using non-linear filter like median filter
            Mat filtered_img = median_blur(diff_img,rows,cols);

            // Apply thresholding
            filtered_img = thresholding(filtered_img,rows,cols);

            // Blur image once more
            filtered_img = median_blur(filtered_img,rows,cols);

            // Use closing to remove 'pepper' in image
            filtered_img = closing(filtered_img,rows,cols);

            /* Step 4: Shade regions with motion */
            Mat final_img = final_masking(filtered_img,rows,cols);

            // namedWindow("Display Image", WINDOW_AUTOSIZE); 
            // imshow("Display Image", color_img2); 
            // waitKey(0);

        
            /* Step 5: Write motion mask to video */
            // Create video capture object to store video
            if (video_not_started == 0)
            {

                //Create and initialize the VideoWriter object 
                VideoWriter vid_writer("motion.avi", VideoWriter::fourcc('H', '2', '6', '4'), 
                                                                        20, color_img2.size(), true); 
                video_not_started = 1;

                // Check if able to open writer
                if (!vid_writer.isOpened()) {
                    cout << "Video writer not opened!" << endl;
                    return -1;
                }
            }

            // Write image to video
            vid_writer.write(final_img);

            // Update counter and frames
            counter--;
            gray_img1 = gray_img2;
            color_img1 = color_img2;
            gray_img2.release();
            color_img2.release();
        }
    }
    
    // Release video capture objects
    vid_reader.release();
    vid_writer.release();
    
    // Close all frames
    destroyAllWindows();
    
    return 0;

}
