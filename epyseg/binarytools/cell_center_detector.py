import math
from skimage.measure import label, regionprops
from timeit import default_timer as timer
import numpy as np

def create_horizontal_gradient(cells):

    horiz_gradient = np.zeros_like(cells)

    for j in range(0, cells.shape[-2], 1):
        counter = 1
        last_id = 0
        for i in range(0, cells.shape[-1], 1):
            # dark stuff or beginning of line creates a beginning of the gradient and a counter
            # then try go back by n pixels when new white/black line is detected

            current_cell_id = cells[j][i]

            if current_cell_id != 0 and current_cell_id != last_id:
                last_id = current_cell_id

            if current_cell_id == 0 and last_id != 0:
                last_id = 0
                # divide counter by 2
                half_counter = int(counter / 2.)
                counter = 1  # restart counter
                # found a new cell edge --> go backwards to create the gradient
                reverse_counter = 1
                for ii in range(0, -half_counter + 1, -1):
                    horiz_gradient[j][i + ii] = reverse_counter
                    reverse_counter += 1

            if current_cell_id != 0 and current_cell_id == last_id:
                horiz_gradient[j][i] = counter
                counter += 1
    return horiz_gradient

def create_vertical_gradient(cells):

    vertical_gradient = np.zeros_like(cells)

    for i in range(0, cells.shape[-1], 1):
        counter = 1
        last_id = 0
        for j in range(0, cells.shape[-2], 1):
            current_cell_id = cells[j][i]

            if current_cell_id != 0 and current_cell_id != last_id:
                last_id = current_cell_id

            if current_cell_id == 0 and last_id != 0:
                last_id = 0
                # divide counter by 2
                half_counter = int(counter / 2.)
                counter = 1  # restart counter
                # found a new cell edge --> go backwards to create the gradient
                reverse_counter = 1
                for jj in range(0, -half_counter + 1, -1):
                    vertical_gradient[j + jj][i] = reverse_counter
                    reverse_counter += 1

            if current_cell_id != 0 and current_cell_id == last_id:
                vertical_gradient[j][i] = counter
                counter += 1
    return vertical_gradient

def get_gradient_and_seeds(cells, horiz_gradient, vertical_gradient):
    # combine gradients
    combined_gradients_ready_for_wshed = horiz_gradient + vertical_gradient

    highest_pixels = np.zeros_like(cells)

    for region in regionprops(cells, intensity_image=combined_gradients_ready_for_wshed):
        max_val = 0

        factor = math.sqrt(region.area)*30/100 # try scale seed by cell size --> use area for that # ideally should have jut one seed per cell and scale with size

        for j in range(0, region.intensity_image.shape[-2], 1):
            for i in range(0, region.intensity_image.shape[-1], 1):
                if region.intensity_image[j, i] >= max_val:
                    max_val = region.intensity_image[j,i]

        region.image[...] = False  # do it better with fill

        region.image[region.intensity_image >= (max_val-factor)] = True

        for coordinates in region.coords:
            highest_pixels[coordinates[0], coordinates[1]] = 255
    return combined_gradients_ready_for_wshed, highest_pixels

if __name__ == '__main__':
    start = timer()

