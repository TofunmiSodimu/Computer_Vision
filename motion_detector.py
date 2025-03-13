import sys
import cv2
import numpy as np
from canny_edge_detector import Canny_Edge_Detector

def main():
    # Check that correct number of command-line arguments are passed
    if (len(sys.argv) != 3):
        print("usage should be: python3 motion_detector.py <Input_Video_Path> <Output_Video_Path>")
        return -1
          
    # Open video file
    cap = cv2.VideoCapture(sys.argv[1])
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return -1

    video_not_started = True

    # Step 1: Get each frame from video and convert to grayscale
    ret, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    rows, cols = gray1.shape

    while cap.isOpened():

        ret, frame2 = cap.read()
        if ret:
            # Convert frame to grayscale
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Check if frames are the same size (a properly encoded video should have frames that are always the same size)
            if (gray1.shape != gray2.shape):
                print("Poorly encoded file")
                return -1
            
            # Step 2: Compute frame difference 
            diff_img = cv2.subtract(gray2,gray1)
            # filtered_img = Canny_Edge_Detector(diff_img).detector()
       
            # Step 3: Remove noise and create a motion mask 
            # Remove noise using non-linear filter like median filter
            kernel_size = 5
            filtered_img = cv2.medianBlur(diff_img,kernel_size)

            # Apply thresholding
            block_size = 7
            filtered_img = cv2.adaptiveThreshold(filtered_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,0)

            # Blur image once more
            filtered_img = cv2.medianBlur(filtered_img,kernel_size)

            # Use closing to remove 'pepper' in image
            c_size = 7
            filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, np.ones((c_size,c_size), np.uint8), iterations=1)

            # Step 4: Shade regions with motion
            final_img = filtered_img.astype(np.uint8)
            # print(np.where(final_img != 0))
        
            # Step 5: Write motion mask to video
            # Create video capture object to store video
            if (video_not_started == True):
                # Create and initialize the VideoWriter object 
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
                fps = 20
                video = cv2.VideoWriter(sys.argv[2], fourcc, fps, (cols, rows)) 
                video_not_started = False

            # Write image to video
            video.write(cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR))

            # Update frames
            gray1 = gray2
        else: break
    
    # Release video capture object and close all windows
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    main()