import cv2
import numpy as np
import sys
from scipy import ndimage

class Canny_Edge_Detector:
    img = np.array([])
    rows = 0
    cols = 0
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    gaussian_size = 0
    stdev = 0

    def __init__(self,img,gaussian_size=3,stdev=3):
        # Read image from path and store in object property
        if type(img) == str: self.img = cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        else: self.img = img

        # Check if the image is empty
        if self.img is None:
            print("Error: Could not read image")

        self.rows, self.cols = self.img.shape
        self.gaussian_size = gaussian_size
        self.stdev = stdev
    
    def detector(self):
        # Step 1 : Filter Image with Derivative of Gaussian
        img_mag, img_orientation = self.filter_d_o_g()

        # Step 2: Non maximum suppression to thin wide ridges down to a single pixel
        nms_img = self.non_max_suppression(img_mag, img_orientation)

        # Step 3: Hysteresis: using high thresholds to start curves and low thresholds to continue them
        final_img = self.hysteresis(nms_img, high_ratio=0.5, low_ratio=0.17)

        # Display images for each step
        # self.display_all(img_mag, nms_img, final_img)

        # Save images
        cv2.imwrite("Img_magnitude.png", (img_mag * 255).astype(np.uint8))
        cv2.imwrite("Final_img.png", (final_img * 255).astype(np.uint8))

        return final_img

    def display(self, image=np.array([])):
        if (image.shape == (0,0)): image = self.img
        cv2.imshow("image", image)
        cv2.waitKey(0) # Hold the window on screen, till user presses a key
        cv2.destroyAllWindows() # Delete created GUI window from screen and memory
    
    def display_all(self, img_mag, nms_img, final_img):
        cv2.imshow("Original Image", self.img)
        cv2.imshow("Norm of Image Gradient", img_mag)
        cv2.imshow("Thinned Image", nms_img)
        cv2.imshow("Thresholded/Final Image", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gaussian_kernel(self, kernel_size, stdev):
        size = kernel_size // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        kernel = (1 / (2.0 * np.pi * stdev**2)) * (np.exp(-((x**2 + y**2) / (2.0*stdev**2))))
        return kernel
    
    def pad(self, img, filter_size):
        padding = filter_size//2
        return np.pad(img, padding, mode='constant', constant_values = 0)
    
    def convolution(self, padded_img, filter, rows, cols, result):
        change = filter.shape[0]//2
        for i in range(change,rows-change):
            for j in range(change,cols-change):
                result[i-change][j-change] = np.sum(padded_img[i-change:i+change+1,j-change:j+change+1] * filter)
        return result

    def gradient(self):
        padded_img = self.pad(self.img, self.prewitt_x.shape[0]) # Pad image
        rows_p, cols_p = padded_img.shape

        # Calculate gradient
        img_dx = self.convolution(padded_img,self.prewitt_x,rows_p,cols_p,np.zeros((self.rows,self.cols)))
        img_dy = self.convolution(padded_img,self.prewitt_y,rows_p,cols_p,np.zeros((self.rows,self.cols)))

        return img_dx, img_dy
    
    # Filter with derivative of gaussian
    def filter_d_o_g(self):
        # Get derivative of gaussian
        gaussian = self.gaussian_kernel(self.gaussian_size, self.stdev) # Get gaussian kernel
        padded_gaussian = self.pad(gaussian, self.prewitt_x.shape[0]) # Pad gaussian
        rows_p, cols_p = padded_gaussian.shape
        dog_x = self.convolution(padded_gaussian,self.prewitt_x,rows_p,cols_p,np.zeros((self.gaussian_size,self.gaussian_size),dtype=np.float32))
        dog_y = self.convolution(padded_gaussian,self.prewitt_y,rows_p,cols_p,np.zeros((self.gaussian_size,self.gaussian_size),dtype=np.float32))

        # Calculate gradient
        padded_img = self.pad(self.img, dog_x.shape[0]) # Pad image
        rows_p, cols_p = padded_img.shape
        img_dx = self.convolution(padded_img,dog_x,rows_p,cols_p,np.zeros((self.rows,self.cols), dtype=np.float32))
        img_dy = self.convolution(padded_img,dog_y,rows_p,cols_p,np.zeros((self.rows,self.cols), dtype=np.float32))

        # Get magnitude and orientation
        img_mag = np.sqrt(np.square(img_dx) + np.square(img_dy))
        img_mag = (img_mag / img_mag.max()) * 1.0
        img_orientation = np.arctan2(img_dy, img_dx) * 180/np.pi

        return img_mag, img_orientation
    
    def blur_derivative(self):
        # Blur image
        gaussian = self.gaussian_kernel(3, stdev = 3) # Get gaussian kernel
        padded_img = self.pad(self.img, gaussian.shape[0]) # Pad image
        rows_p, cols_p = padded_img.shape
        blurred_img = self.convolution(padded_img,gaussian,rows_p,cols_p,np.zeros((self.rows,self.cols),dtype=np.uint8))

        # Calculate gradient
        padded_img = self.pad(blurred_img, self.prewitt_x.shape[0]) # Pad image
        rows_p, cols_p = padded_img.shape
        img_dx = self.convolution(padded_img,self.prewitt_x,rows_p,cols_p,np.zeros((self.rows,self.cols),dtype=np.uint8))
        img_dy = self.convolution(padded_img,self.prewitt_y,rows_p,cols_p,np.zeros((self.rows,self.cols), dtype=np.uint8))

        # Get magnitude and orientation
        img_mag = np.sqrt(np.square(img_dx) + np.square(img_dy))
        img_mag = ((img_mag / img_mag.max()) * 255).astype(np.uint8)
        img_orientation = np.arctan2(img_dy,img_dx) * 180/np.pi

        return img_mag, img_orientation
    
    def non_max_suppression(self, img_mag, img_orientation):
        nms_img = np.zeros((self.rows,self.cols),np.float32)
        # Adjust img_orientation values to range of 0 to 180
        img_orientation[img_orientation < 0] += 180

        # Check two neighbors of each pixel and pick largest intensity
        for i in range(1,self.rows-1):
            for j in range(1,self.cols-1):
                # Sideways
                if ((0 <= img_orientation[i][j] < 22.5) or (157.5 <= img_orientation[i][j] <= 180)):
                    if ((img_mag[i][j] > img_mag[i][j-1]) and (img_mag[i][j] > img_mag[i][j+1])):
                        nms_img[i][j] = img_mag[i][j]

                # Right diagonal
                if (22.5 <= img_orientation[i][j] < 67.5) :
                    if ((img_mag[i][j] > img_mag[i-1][j+1]) and (img_mag[i][j] > img_mag[i+1][j-1])):
                        nms_img[i][j] = img_mag[i][j]

                # Left diagonal
                if (112.5 <= img_orientation[i][j] < 157.5):
                    if ((img_mag[i][j] > img_mag[i-1][j-1]) and (img_mag[i][j] > img_mag[i+1][j+1])):
                        nms_img[i][j] = img_mag[i][j]
                
                # Up and down
                if (67.5 <= img_orientation[i][j] < 112.5):
                    if ((img_mag[i][j] > img_mag[i-1][j]) and (img_mag[i][j] > img_mag[i+1][j])):
                        nms_img[i][j] = img_mag[i][j]
        return nms_img
    
    def hysteresis(self, image, high_ratio, low_ratio):
        hysteresis_img = np.ones((self.rows,self.cols)) * 0.5

        # Calculate thresholds
        high_threshold = high_ratio * image.max()
        low_threshold = high_threshold * low_ratio

        hysteresis_img[image < low_threshold] = 0.0 #Set pixels with intensities lower than low_threshold to 0
        hysteresis_img[image >= high_threshold] = 1.0 # Set pixels with intensities greater than high_threshold to 1

        # Look for strong pixel and convert low pixel to strong if connected to strong pixel
        for i in range(1,self.rows-1):
            for j in range(1,self.cols-1):

                if (hysteresis_img[i][j] == 0.5):
                    if np.count_nonzero(hysteresis_img[i-1:i+2, j-1:j+2]) > 0:
                        hysteresis_img[i][j] = 1.0
                    else: hysteresis_img[i][j] = 0.0
        
        return hysteresis_img


def main():
    # Check that correct number of command-line arguments are passed
    if (len(sys.argv) != 2):
        print("usage should be: python3 canny_edge_detector.py <Input_Image_Path> ")
        return -1
    ced = Canny_Edge_Detector(sys.argv[1])
    ced.detector()


if __name__ == "__main__":
    main()