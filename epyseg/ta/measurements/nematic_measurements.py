import math

def compute_polarity_nematic(centroid, boundary_coords, image_to_compute_polarity_for):
    """
    Computes the polarity nematic tensor.

    Args:
        centroid (tuple): The centroid coordinates (y, x).
        boundary_coords (list): List of boundary coordinates.
        image_to_compute_polarity_for (numpy.ndarray): Image for polarity computation.

    Returns:
        list: The computed polarity values [S1, S2, total_intensity].

    """

    center_x = centroid[1]
    center_y = centroid[0]

    # Compute theta values for each boundary point
    theta_unsorted = []
    for pt in boundary_coords:
        deltaX = pt[1] - center_x
        deltaY = pt[0] - center_y
        rSquared = deltaX * deltaX + deltaY * deltaY
        r = math.sqrt(rSquared)
        cosTheta = deltaX / r
        sinTheta = deltaY / r
        theta = math.atan2(sinTheta, cosTheta)
        theta_unsorted.append(theta)

    # Create a hashmap with boundary points and corresponding theta values
    hashmap_point_angle = {}
    for z in range(len(boundary_coords)):
        hashmap_point_angle[z] = theta_unsorted[z]

    # Sort the hashmap by theta values
    cells_sorted_by_theta = {k: v for k, v in sorted(hashmap_point_angle.items(), key=lambda item: item[1])}

    # Compute necessary trigonometric quantities for each point
    theta_sorted = []
    cos2Theta_sorted = []
    sin2Theta_sorted = []
    for k, v in cells_sorted_by_theta.items():
        pt = boundary_coords[k]
        deltaX = pt[1] - center_x
        deltaY = pt[0] - center_y
        rSquared = deltaX * deltaX + deltaY * deltaY
        r = math.sqrt(rSquared)
        cosTheta = deltaX / r
        sinTheta = deltaY / r
        theta = math.atan2(sinTheta, cosTheta)
        theta_sorted.append(theta)
        cos2Theta_sorted.append(2. * cosTheta * cosTheta - 1.)
        sin2Theta_sorted.append(2. * cosTheta * sinTheta)

    # Compute polarity using the sorted values
    polarity = computeS1S2ForChannel(boundary_coords, cells_sorted_by_theta, theta_sorted, cos2Theta_sorted,
                                     sin2Theta_sorted, image_to_compute_polarity_for)

    return polarity


def computeS1S2ForChannel(boundary_points, cells_sorted_by_theta, theta_sorted, cos2Theta_sorted, sin2Theta_sorted,
                          channel_for_quantification):
    """
    Computes S1 and S2 values for a given channel.

    Args:
        boundary_points (list): List of boundary points.
        cells_sorted_by_theta (dict): Sorted boundary cells by theta.
        theta_sorted (list): Sorted theta values.
        cos2Theta_sorted (list): Sorted cos(2 * theta) values.
        sin2Theta_sorted (list): Sorted sin(2 * theta) values.
        channel_for_quantification (numpy.ndarray): Channel image for quantification.

    Returns:
        list: The computed S1, S2, and total intensity values.

    """

    k = 0
    S1_ch1 = 0
    S2_ch1 = 0
    sint_ch1 = 0.

    for idx in cells_sorted_by_theta.keys():
        pt = boundary_points[idx]
        curIntensity_ch1 = channel_for_quantification[pt[0], pt[1]]
        numPoints = len(boundary_points)

        # Compute theta values for neighboring points
        theta_last = theta_sorted[numPoints - 1] - (math.pi * 2.) if k == 0 else theta_sorted[k - 1]
        theta_next = theta_sorted[0] + (math.pi * 2.) if k == numPoints - 1 else theta_sorted[k + 1]

        # Compute deltaTheta as a three-point differential
        deltaTheta = 0.5 * (theta_next - theta_last)

        # Update S1, S2, and total intensity sums
        S1_ch1 += cos2Theta_sorted[k] * deltaTheta * curIntensity_ch1
        S2_ch1 += sin2Theta_sorted[k] * deltaTheta * curIntensity_ch1
        sint_ch1 += deltaTheta * curIntensity_ch1

        k += 1

    return [S1_ch1.tolist(), S2_ch1.tolist(),
            sint_ch1.tolist()]
