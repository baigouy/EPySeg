import random
import matplotlib.pyplot as plt
import numpy as np

def plot_image_with_pixels(image, cmap='gray'):
    '''
    Plots an image with white lines surrounding each pixel. NB: This only make sense for small images

    Args:
        image (numpy.ndarray): The input image array.
        cmap (str, optional): The colormap to use for displaying the image. Defaults to 'gray'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plotted image.

    # Examples:
    #     >>> image = np.random.rand(10, 10)
    #     >>> fig = plot_image_with_pixels(image)
    #     >>> plt.show()
    '''
    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Show the image using imshow()
    ax.imshow(image, cmap=cmap, interpolation=None)

    # Set the tick locations to center the ticks at the pixels
    ax.set_xticks(np.arange(image.shape[1]), minor=False)
    ax.set_yticks(np.arange(image.shape[0]), minor=False)

    # Shift the grid by 0.5 to surround the pixels
    ax.set_xticks(np.arange(image.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(image.shape[0]+1)-0.5, minor=True)

    # If the image is too large, hide the tick labels
    if image.shape[0] > 50:
        ax.set_yticklabels([])

    if image.shape[1] > 50:
        ax.set_xticklabels([])

    # Add the lines around each pixel using grid()
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    # Return the figure object
    return fig


if __name__ == '__main__':

    if True:
        # Generate a random image of a random type
        image = random.choice([np.random.rand(10, 10), np.random.rand(100, 10, 3)])

        # Plot the image with white lines surrounding each pixel
        fig = plot_image_with_pixels(image)

        # Show the plot
        plt.show()


    if True:
        from epyseg.img import pop, Img
        img = Img('/E/Sample_images/sample_images_FIJI/AuPbSn40.jpg')
        pop(img)
        # sys.exit(0)




