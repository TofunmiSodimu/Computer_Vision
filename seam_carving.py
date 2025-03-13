import cv2
import numpy as np
from scipy import ndimage

class Seam_Carving:
    img = np.array([])
    rows = 0
    cols = 0
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


    def __init__(self,img_path):
        # Read image from path and store in object property
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint16)
        self.rows, self.cols = self.img.shape

    def display(self, image=np.array([])):
        if (image.shape == (0,0)): image = self.img
        cv2.imshow("image", image)
        cv2.waitKey(0) # Hold the window on screen, till user presses a key
        cv2.destroyAllWindows() # Delete created GUI window from screen and memory
    
    def display_all(self, img_energy, final_img):
        cv2.imshow("Original Image", self.img.astype(np.uint8))
        cv2.imshow("Energy of Image", img_energy)
        cv2.imshow("Carved Image", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def pad(self, img, filter_size):
        padding = filter_size//2
        return np.pad(img, padding, mode='constant', constant_values = 0)
    
    def convolution(self, padded_img, filter, rows, cols, result):
        change = filter.shape[0]//2
        for i in range(change,rows-change):
            for j in range(change,cols-change):
                result[i-change][j-change] = np.sum(padded_img[i-change:i+change+1,j-change:j+change+1] * filter)
        return result

    def energy(self):
        padded_img = self.pad(self.img, self.prewitt_x.shape[0]) # Pad image
        rows_p, cols_p = padded_img.shape

        # Calculate gradient
        img_dx = self.convolution(padded_img,self.prewitt_x,rows_p,cols_p,np.zeros((self.rows,self.cols)))
        img_dy = self.convolution(padded_img,self.prewitt_y,rows_p,cols_p,np.zeros((self.rows,self.cols)))

        # Get magnitude/energy
        img_mag = np.sqrt(np.square(img_dx) + np.square(img_dy))
        img_mag = (img_mag / img_mag.max())

        return img_mag

    def min_cost(self, img_energy):
        dp_arr = np.copy(img_energy)

        for i in range(1,self.rows):
            for j in range(self.cols):
                if (j == 0): 
                    dp_arr[i][j] += min(dp_arr[i-1][j],dp_arr[i-1][j+1])
                elif (j == self.cols-1): dp_arr[i][j] += min(dp_arr[i-1][j-1],dp_arr[i-1][j])
                else: dp_arr[i][j] += min(dp_arr[i-1][j-1],dp_arr[i-1][j],dp_arr[i-1][j+1])

        return dp_arr
    
    def remove_path(self, img, img_energy, dp_arr):

        i = dp_arr.shape[0]-1
        j = np.argmin(dp_arr[i,:])
        final_img = []
        new_energy = []
        new_dp = []
        final_img.append(np.concatenate((img[i,0:j],img[i,j+1:])))
        new_energy.append(np.concatenate((img_energy[i,0:j],img_energy[i,j+1:])))
        new_dp.append(np.concatenate((dp_arr[i,0:j],dp_arr[i,j+1:])))

        while (i > 0):

            if (j == 0):
                if (dp_arr[i-1][j] < dp_arr[i-1][j+1]): 
                    final_img.append(np.concatenate((img[i,0:j],img[i,j+1:])))
                    new_energy.append(np.concatenate((img_energy[i,0:j],img_energy[i,j+1:])))
                    new_dp.append(np.concatenate((dp_arr[i,0:j],dp_arr[i,j+1:])))
                elif (dp_arr[i-1][j] > dp_arr[i-1][j+1]): 
                    final_img.append(np.concatenate((img[i,0:j+1],img[i,j+2:])))
                    new_energy.append(np.concatenate((img_energy[i,0:j+1],img_energy[i,j+2:])))
                    new_dp.append(np.concatenate((dp_arr[i,0:j+1],dp_arr[i,j+2:])))
                    j += 1

            elif (j == self.cols-1):
                if (dp_arr[i-1][j-1] < dp_arr[i-1][j]): 
                    final_img.append(np.concatenate((img[i,0:j-1],img[i,j:])))
                    new_energy.append(np.concatenate((img_energy[i,0:j-1],img_energy[i,j:])))
                    new_dp.append(np.concatenate((dp_arr[i,0:j-1],dp_arr[i,j:])))
                    j -= 1
                elif (dp_arr[i-1][j-1] > dp_arr[i-1][j]): 
                    final_img.append(np.concatenate((img[i,0:j],img[i,j+1:])))
                    new_energy.append(np.concatenate((img_energy[i,0:j],img_energy[i,j+1:])))
                    new_dp.append(np.concatenate((dp_arr[i,0:j],dp_arr[i,j+1:])))
                    
            else: 
                if (dp_arr[i-1][j] >= dp_arr[i-1][j-1]) and (dp_arr[i-1][j+1] >= dp_arr[i-1][j-1]): 
                    final_img.append(np.concatenate((img[i,0:j-1],img[i,j:])))
                    new_energy.append(np.concatenate((img_energy[i,0:j-1],img_energy[i,j:])))
                    new_dp.append(np.concatenate((dp_arr[i,0:j-1],dp_arr[i,j:])))
                    j -= 1
                elif (dp_arr[i-1][j-1] >= dp_arr[i-1][j]) and (dp_arr[i-1][j+1] >= dp_arr[i-1][j]):
                    final_img.append(np.concatenate((img[i,0:j],img[i,j+1:])))
                    new_energy.append(np.concatenate((img_energy[i,0:j],img_energy[i,j+1:])))
                    new_dp.append(np.concatenate((dp_arr[i,0:j],dp_arr[i,j+1:])))
                elif (dp_arr[i-1][j-1] >= dp_arr[i-1][j+1]) and (dp_arr[i-1][j] >= dp_arr[i-1][j+1]):
                    final_img.append(np.concatenate((img[i,0:j+1],img[i,j+2:])))
                    new_energy.append(np.concatenate((img_energy[i,0:j+1],img_energy[i,j+2:])))
                    new_dp.append(np.concatenate((dp_arr[i,0:j+1],dp_arr[i,j+2:])))
                    j += 1
            
            i -= 1
        
        return np.vstack(final_img), np.vstack(new_energy), np.vstack(new_dp)



def main():
    carver = Seam_Carving("/Users/oluwatofunmisodimu/Documents/Perception_py/Kinkaku-ji.jpg")
    
    # Step 1 : Get energy/magnitude of image
    img_energy = carver.energy()
    img_energy_formatted = (img_energy * 255).astype(np.uint8)

    # Step 2: Compute minimum cost to get to each pixel from the top - vertical seam
    dp_img = carver.min_cost(img_energy)
    # carver.display((dp_img/dp_img.max() * 255).astype(np.uint8))

    # Step 3: Backtrack in dp_array to get min_path and delete from og_img
    final_img = np.copy(carver.img)
    for _ in range(100):
        final_img, img_energy, dp_img = carver.remove_path(final_img, img_energy, dp_img)
    final_img_formatted = (final_img/final_img.max() * 255).astype(np.uint8)

    # Display images
    carver.display_all(img_energy_formatted,final_img_formatted)


if __name__ == "__main__":
    main()