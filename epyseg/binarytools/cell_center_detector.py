import math
from skimage.measure import label, regionprops
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

def add_2d_border(image2d):
    """
    Adds a border to a 2D image.

    Args:
        image2d (ndarray): The 2D image to which the border will be added.

    Returns:
        ndarray: The image with the added border.

    Examples:
        >>> image2d = np.array([[1, 2, 3],
        ...                     [4, 5, 6],
        ...                     [7, 8, 9]])
        >>> result = add_2d_border(image2d)
        >>> print(result)
        [[0 0 0]
         [0 5 0]
         [0 0 0]]
    """
    # Define the region where the border will be inserted
    insertHere = (slice(1, image2d.shape[0] - 1), slice(1, image2d.shape[1] - 1))

    # Create an array filled with zeros to hold the image with the border
    cells_with_borders = np.zeros_like(image2d)
    cells_with_borders.fill(0)

    # Copy the interior of the image to the array with the border
    cells_with_borders[insertHere] = image2d[insertHere]

    # Update the original image with the image containing the border
    image2d = cells_with_borders

    return image2d



def create_horizontal_gradient(cells):
    """
    Creates a horizontal gradient based on the cell IDs in a 2D array.

    Args:
        cells (ndarray): The 2D array containing cell IDs.

    Returns:
        ndarray: The horizontal gradient array.

    # Examples:
    #     >>> cells = np.array([[0, 0, 0],
    #     ...                   [1, 1, 0],
    #     ...                   [2, 2, 0]])
    #     >>> result = create_horizontal_gradient(cells)
    #     >>> print(result)
    #     [[0 0 0]
    #      [0 1 0]
    #      [0 0 0]]

    """
    # Add a border to the cells array
    cells = add_2d_border(cells)

    # Create an array of zeros with the same shape as cells to hold the horizontal gradient
    horiz_gradient = np.zeros_like(cells)

    # Iterate over the rows and columns of the cells array
    for j in range(0, cells.shape[-2], 1):
        counter = 1
        last_id = 0

        for i in range(0, cells.shape[-1], 1):
            # Check the current cell ID and compare it with the previous cell ID

            current_cell_id = cells[j][i]

            if current_cell_id != 0 and current_cell_id != last_id:
                last_id = current_cell_id

            if current_cell_id == 0 and last_id != 0:
                last_id = 0
                # Divide the counter by 2
                half_counter = int(counter / 2.)
                counter = 1  # Restart the counter

                # Found a new cell edge, go backwards to create the gradient
                reverse_counter = 1
                for ii in range(0, -half_counter + 1, -1):
                    horiz_gradient[j][i + ii] = reverse_counter
                    reverse_counter += 1

            if current_cell_id != 0 and current_cell_id == last_id:
                horiz_gradient[j][i] = counter
                counter += 1

    return horiz_gradient



def create_vertical_gradient(cells):
    """
    Creates a vertical gradient based on the cell IDs in a 2D array.

    Args:
        cells (ndarray): The 2D array containing cell IDs.

    Returns:
        ndarray: The vertical gradient array.

    # Examples:
    #     >>> cells = np.array([[0, 0, 0],
    #     ...                   [1, 1, 0],
    #     ...                   [2, 2, 0]])
    #     >>> result = create_vertical_gradient(cells)
    #     >>> print(result)
    #     [[0 0 0]
    #      [0 1 0]
    #      [0 0 0]]
    """
    # Add a border to the cells array
    cells = add_2d_border(cells)

    # Create an array of zeros with the same shape as cells to hold the vertical gradient
    vertical_gradient = np.zeros_like(cells)

    # Iterate over the columns and rows of the cells array
    for i in range(0, cells.shape[-1], 1):
        counter = 1
        last_id = 0

        for j in range(0, cells.shape[-2], 1):
            current_cell_id = cells[j][i]

            if current_cell_id != 0 and current_cell_id != last_id:
                last_id = current_cell_id

            if current_cell_id == 0 and last_id != 0:
                last_id = 0
                # Divide the counter by 2
                half_counter = int(counter / 2.)
                counter = 1  # Restart the counter

                # Found a new cell edge, go backwards to create the gradient
                reverse_counter = 1
                for jj in range(0, -half_counter + 1, -1):
                    vertical_gradient[j + jj][i] = reverse_counter
                    reverse_counter += 1

            if current_cell_id != 0 and current_cell_id == last_id:
                vertical_gradient[j][i] = counter
                counter += 1

    return vertical_gradient


def get_seeds(cells, one_seed_per_cell=True):
    """
    Obtain seeds for watershed segmentation based on cell gradients.

    Args:
        cells (ndarray): The 2D array containing cell IDs.
        one_seed_per_cell (bool): Whether to ensure only one seed per cell. Default is True.

    Returns:
        tuple: A tuple containing the combined gradients array and the seeds array.

    # Examples:
    #     >>> cells = np.array([[0, 0, 0],
    #     ...                   [1, 1, 0],
    #     ...                   [2, 2, 0]])
    #     >>> result = get_seeds(cells)
    #     >>> combined_gradients, seeds = result
    #     >>> print(combined_gradients)
    #     [[0 0 0]
    #      [1 2 0]
    #      [1 2 0]]
    #     >>> print(seeds)
    #     [[0 0 0]
    #      [0 2 0]
    #      [0 0 0]]
    """
    # Create horizontal and vertical gradients
    horiz_gradient = create_horizontal_gradient(cells)
    vertical_gradient = create_vertical_gradient(cells)

    # Perform further processing on gradients to obtain combined gradients and seeds
    combined_gradients_ready_for_wshed, seeds = _get_seeds(cells, horiz_gradient, vertical_gradient)

    # If one_seed_per_cell is True, ensure only one seed per cell
    if one_seed_per_cell:
        new_seeds = label(seeds.astype(np.uint8), connectivity=1, background=0)
        props_seeds = regionprops(new_seeds)

        extra_seeds_to_remove = []

        for region in regionprops(cells):
            cells_found = []
            for coordinates in region.coords:
                cells_found.append(new_seeds[coordinates[0], coordinates[1]])
            cells_found = list(dict.fromkeys(cells_found))
            if 0 in cells_found:
                cells_found.remove(0)
            if len(cells_found) > 1:
                max_area = 0
                for cell in cells_found:
                    region = props_seeds[cell - 1]
                    max_area = max(max_area, region.area)

                for cell in cells_found:
                    if not props_seeds[cell - 1].area == max_area:
                        extra_seeds_to_remove.append(cell)

        for region in props_seeds:
            if region.label in extra_seeds_to_remove:
                for coordinates in region.coords:
                    seeds[coordinates[0], coordinates[1]] = 0

    return combined_gradients_ready_for_wshed, seeds



def _get_seeds(cells, horiz_gradient, vertical_gradient):
    """
    Obtain seeds based on combined gradients for watershed segmentation.

    Args:
        cells (ndarray): The 2D array containing cell IDs.
        horiz_gradient (ndarray): The 2D array representing the horizontal gradient.
        vertical_gradient (ndarray): The 2D array representing the vertical gradient.

    Returns:
        tuple: A tuple containing the combined gradients array and the highest pixel values array.

    # Examples:
    #     >>> cells = np.array([[0, 0, 0],
    #     ...                   [1, 1, 0],
    #     ...                   [2, 2, 0]])
    #     >>> horiz_gradient = np.array([[0, 0, 0],
    #     ...                            [1, 2, 0],
    #     ...                            [1, 2, 0]])
    #     >>> vertical_gradient = np.array([[0, 0, 0],
    #     ...                              [0, 1, 0],
    #     ...                              [0, 1, 0]])
    #     >>> result = _get_seeds(cells, horiz_gradient, vertical_gradient)
    #     >>> combined_gradients, highest_pixels = result
    #     >>> print(combined_gradients)
    #     [[0 0 0]
    #      [1 3 0]
    #      [1 3 0]]
    #     >>> print(highest_pixels)
    #     [[  0   0   0]
    #      [  0 255   0]
    #      [  0 255   0]]
    """
    # Combine gradients
    combined_gradients_ready_for_wshed = horiz_gradient + vertical_gradient

    # Create array for highest pixel values
    highest_pixels = np.zeros_like(cells)

    # Iterate over labeled regions in cells with intensity_image set to combined_gradients_ready_for_wshed
    for region in regionprops(cells, intensity_image=combined_gradients_ready_for_wshed):
        max_val = 0

        # Scale seed by cell size using the square root of the region's area
        factor = math.sqrt(region.area) * 30 / 100

        # Find the maximum value in the intensity image
        for j in range(0, region.intensity_image.shape[-2], 1):
            for i in range(0, region.intensity_image.shape[-1], 1):
                if region.intensity_image[j, i] >= max_val:
                    max_val = region.intensity_image[j, i]

        # Set the region's image to False
        region.image.fill(False)

        # Set pixels in the region's image to True where the intensity exceeds (max_val - factor)
        region.image[region.intensity_image >= (max_val - factor)] = True

        # Update the highest_pixels array with the region's coordinates
        for coordinates in region.coords:
            highest_pixels[coordinates[0], coordinates[1]] = 255

    return combined_gradients_ready_for_wshed, highest_pixels



if __name__ == '__main__':
    start = timer()
