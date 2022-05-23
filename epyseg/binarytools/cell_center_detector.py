import math
from skimage.measure import label, regionprops
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt


def add_2d_border(image2d):
    insertHere = (slice(1, image2d.shape[0] - 1), slice(1, image2d.shape[1] - 1))
    cells_with_borders = np.zeros_like(image2d)
    cells_with_borders.fill(0)

    cells_with_borders[insertHere] = image2d[insertHere]
    # plt.imshow(cells_with_borders)
    # plt.show()
    # print(insertHere)
    image2d = cells_with_borders

    # print(image2d.shape)
    return image2d


def create_horizontal_gradient(cells):
    cells = add_2d_border(cells)
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
    cells = add_2d_border(cells)
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


def get_seeds(cells, one_seed_per_cell=True):
    # TODO really need make sure there is only one seed per cell --> if many then reduce to one by keeping only the biggest --> good idea and may improve things
    # really worth a test...
    horiz_gradient = create_horizontal_gradient(cells)
    vertical_gradient = create_vertical_gradient(cells)

    # plt.imshow(horiz_gradient)
    # plt.show()
    #
    # plt.imshow(vertical_gradient)
    # plt.show()

    combined_gradients_ready_for_wshed, seeds = _get_seeds(cells, horiz_gradient, vertical_gradient)

    # plt.imshow(combined_gradients_ready_for_wshed)
    # plt.show()

    # Img(combined_gradients_ready_for_wshed, dimensions='hw').save('/E/Sample_images/sample_images_epiguy_pyta/images_with_different_bits/predict/gradient.tif')

    # if there are several seeds for a cell then just keep the biggest --> good idea
    # how can I do that --> maybe simply do so by counting ids for stuff and remove smallest
    # count how many seeds are found for each cell

    if one_seed_per_cell:
        new_seeds = label(seeds.astype(np.uint8), connectivity=1, background=0)
        props_seeds = regionprops(new_seeds)

        extra_seeds_to_remove = []

        for region in regionprops(cells):
            # remove small seeds
            cells_found = []
            for coordinates in region.coords:
                cells_found.append(new_seeds[coordinates[0], coordinates[1]])
            cells_found = list(dict.fromkeys(cells_found))
            if 0 in cells_found:
                cells_found.remove(0)
            if len(cells_found) > 1:
                # extra_seeds_to_remove
                # loop over seeds by area and keep only the best/biggest
                # see
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
                    seeds[coordinates[0], coordinates[1]] = 0  # do remove the seed

    return combined_gradients_ready_for_wshed, seeds


def _get_seeds(cells, horiz_gradient, vertical_gradient):
    # combine gradients
    combined_gradients_ready_for_wshed = horiz_gradient + vertical_gradient

    highest_pixels = np.zeros_like(cells)

    for region in regionprops(cells, intensity_image=combined_gradients_ready_for_wshed):
        max_val = 0

        factor = math.sqrt(
            region.area) * 30 / 100  # try scale seed by cell size --> use area for that # ideally should have jut one seed per cell and scale with size

        for j in range(0, region.intensity_image.shape[-2], 1):
            for i in range(0, region.intensity_image.shape[-1], 1):
                if region.intensity_image[j, i] >= max_val:
                    max_val = region.intensity_image[j, i]

        # region.image[...] = False  # do it better with fill
        region.image.fill(False)

        region.image[region.intensity_image >= (max_val - factor)] = True

        for coordinates in region.coords:
            highest_pixels[coordinates[0], coordinates[1]] = 255
    return combined_gradients_ready_for_wshed, highest_pixels


if __name__ == '__main__':
    start = timer()
