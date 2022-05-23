import math


def compute_polarity_nematic(centroid, boundary_coords, image_to_compute_polarity_for):
    center_x = centroid[1]
    center_y = centroid[0]
    # WE MUST SORT OUR PIXELS SO THAT THETA IS MONOTONICALLY INCREASING
    theta_unsorted = []

    for pt in boundary_coords:
        deltaX = pt[1] - center_x
        deltaY = pt[0] - center_y
        rSquared = deltaX * deltaX + deltaY * deltaY
        r = math.sqrt(rSquared)
        cosTheta = deltaX / r
        sinTheta = deltaY / r
        theta = math.atan2(sinTheta, cosTheta)  # BECAUSE OF THE NEED TO CALCULATE dTheta THIS CANNOT BE AVOIDED
        theta_unsorted.append(theta)

    # print('theta_unsorted',theta_unsorted) # so far so good

    #       HashMap<Point, Double> hashmap_point_angle = new HashMap<Point, Double>();
    # #         for (int z = 0; z < boundary_points.size(); z++) {
    # #             hashmap_point_angle.put(boundary_points.get(z), theta_unsorted.get(z));                        //SORT HASHMAP BASED ON ANGLES
    # #         }

    # do I even need that I doubt it strongly
    hashmap_point_angle = {}
    for z in range(len(boundary_coords)):
        # hashmap_point_angle[str(boundary_coords[z])]= theta_unsorted[z]                      #SORT HASHMAP BASED ON ANGLES
        hashmap_point_angle[z] = theta_unsorted[z]  # SORT HASHMAP BASED ON ANGLES
        # alternatively could just store Z

    # print('hashmap_point_angle',hashmap_point_angle)

    # need sort the dict by its values
    cells_sorted_by_theta = {k: v for k, v in sorted(hashmap_point_angle.items(), key=lambda item: item[1])}

    # print('cells_sorted_by_theta', cells_sorted_by_theta) # ça marche mais voir si il faut inverser le truc ou pas

    # COMPUTE NECESSARY TRIGONOMETRIC QUANTITES FOR EACH POINT
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

    # COMPUTE NEMATIC TENSOR
    # IT IS IMPORTANT THAT theta_sorted BE MONOTONICALLY INCREASING BEFORE THIS LOOP (SORT IT!)

    # print('cos2Theta_sorted',cos2Theta_sorted)
    # print('sin2Theta_sorted',sin2Theta_sorted)

    # compute it for all channels --> gain of time in fact
    polarity = computeS1S2ForChannel(boundary_coords, cells_sorted_by_theta, theta_sorted, cos2Theta_sorted,
                                     sin2Theta_sorted, image_to_compute_polarity_for)

    # print('polarity',polarity)
    return polarity


def computeS1S2ForChannel(boundary_points, cells_sorted_by_theta, theta_sorted, cos2Theta_sorted, sin2Theta_sorted,
                          channel_for_qunantification):
    k = 0
    S1_ch1 = 0
    S2_ch1 = 0
    sint_ch1 = 0.  # sum of intensities

    for idx in cells_sorted_by_theta.keys():
        pt = boundary_points[idx]
        # GET THE CURRENT POINT AND INTENSITY

        curIntensity_ch1 = channel_for_qunantification[pt[0], pt[1]]
        numPoints = len(boundary_points)
        #
        # USE PERIODIC BOUNDARY CONDITIONS
        # theta_last = (k == 0) ? theta_sorted[numPoints - 1] - (math.pi*2.) : theta_sorted[k - 1]
        # theta_next = (k == numPoints - 1) ? theta_sorted[0] + (math.pi*2.) : theta_sorted[k + 1]
        # same in python format:
        theta_last = theta_sorted[numPoints - 1] - (math.pi * 2.) if k == 0 else theta_sorted[k - 1]
        theta_next = theta_sorted[0] + (math.pi * 2.) if k == numPoints - 1 else theta_sorted[k + 1]

        #
        # THIS IS NECESSARY TO GET 0. FOR CONSTANT INTENSITY
        deltaTheta = 0.5 * (theta_next - theta_last)  # COMPUTE deltaTheta AS A THREE-POINT DIFFERENTIAL
        #             //double deltaTheta=theta_sorted.get(j)-theta_next;  //COMPUTE deltaTheta AS A TWO-POINT BACKWARDS DIFFERENTIAL
        #
        # ADD THE SUMMANDS TO THEIR RESPECTIVE SUMS...
        # polarity
        S1_ch1 += cos2Theta_sorted[k] * deltaTheta * curIntensity_ch1
        S2_ch1 += sin2Theta_sorted[k] * deltaTheta * curIntensity_ch1
        sint_ch1 += deltaTheta * curIntensity_ch1
        # //            sint_ch2 += deltaTheta * curIntensity_ch2;
        # //            sint_ch3 += deltaTheta * curIntensity_ch3;
        # //            sint_12bits += deltaTheta * curIntensity_12bits;
        k += 1

    # print(S1_ch1, S2_ch1, sint_ch1)
    return [S1_ch1.tolist(), S2_ch1.tolist(),
            sint_ch1.tolist()]  # this is a nematic object # in fact it does compute for all channels --> it works even better than expected

# --> very good I do have polarity now !!!
# just check that it is the same as in TA and do a nematic object in python too --> because can be useful!!!
