
import os

import os

def save_IJ_ROIs_or_overlays(rois, output_filename):
    """
    Saves ImageJ ROIs or overlays to a file.

    Args:
        rois (list): List of ROIs or overlays to be saved.
        output_filename (str): Output filename for saving the ROIs.

    Returns:
        None
    """

    # Check the number of ROIs
    if len(rois) > 1:
        # If there are multiple ROIs, check if the output filename ends with '.zip'.
        # If not, append '.zip' to the output filename.
        if not output_filename.lower().endswith('.zip'):
            output_filename += '.zip'

        # Check if the output file already exists and delete it if it does.
        if os.path.exists(output_filename):
            os.remove(output_filename)
    else:
        # If there is only one ROI, check if the output filename ends with '.roi'.
        # If not, append '.roi' to the output filename.
        if not output_filename.lower().endswith('.roi'):
            output_filename += '.roi'

    # Check if the ROIs list is not empty or None
    if rois is not None and rois:
        # Iterate over each ROI
        for roi in rois:
            # Save the ROI to the output filename
            roi.tofile(output_filename)


if __name__ == '__main__':
    pass