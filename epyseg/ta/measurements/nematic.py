import math
import sys
import traceback

from numba import jit, njit
from PIL.ImageDraw import ImageDraw
from skimage.draw import line_aa, line
import numpy as np
from skimage.measure import regionprops
from skimage.measure import label
from skimage.measure._regionprops import RegionProperties

from epyseg.draw.shapes.line2d import Line2D

def compute_stretch_nematic(pointsWithinArea_or_region, normalizeByArea=True):
    """
    Computes the stretch nematic from a given set of points within an area or region.

    Args:
        pointsWithinArea_or_region (numpy.ndarray or RegionProperties): The points within the area or region.
        normalizeByArea (bool, optional): Flag indicating whether to normalize the nematic by the area.

    Returns:
        Nematic: The computed stretch nematic.

    """

    if pointsWithinArea_or_region is None:
        return

    if isinstance(pointsWithinArea_or_region, RegionProperties):
        pointsWithinArea_or_region = pointsWithinArea_or_region.coords

    S1, S2, center = _compute_stretch_nematic(pointsWithinArea_or_region, normalizeByArea=normalizeByArea)
    return Nematic(S1=S1, S2=S2, center=center)


def _compute_stretch_nematic(pointsWithinArea_or_region, normalizeByArea=True):
    """
    Computes the stretch nematic tensor from a given set of points within an area or region.

    Args:
        pointsWithinArea_or_region (numpy.ndarray): The points within the area or region.
        normalizeByArea (bool, optional): Flag indicating whether to normalize the nematic by the area.

    Returns:
        float: The S1 component of the nematic tensor.
        float: The S2 component of the nematic tensor.
        list: The center coordinates.

    """

    center_y, center_x = np.average(pointsWithinArea_or_region, axis=0)
    center = [center_y, center_x]

    S1 = 0.
    S2 = 0.

    for point in pointsWithinArea_or_region:
        deltaX = float(point[1]) - center_x
        deltaY = float(point[0]) - center_y
        rSquared = deltaX * deltaX + deltaY * deltaY
        cos2Theta = 2. * deltaX * deltaX / rSquared - 1.
        sin2Theta = 2. * deltaX * deltaY / rSquared

        S1 += cos2Theta
        S2 += sin2Theta

    if normalizeByArea:
        S1 /= pointsWithinArea_or_region.shape[0]
        S2 /= pointsWithinArea_or_region.shape[0]

    return S1, S2, center


def compute_average_nematic(*args):
    """
    Computes the average nematic from a series of nematics or their S1S2 components.

    Args:
        *args: Variable number of arguments, either Nematic objects or S1S2 component pairs.

    Returns:
        Nematic: The computed average nematic.

    """

    if args is None or len(args) == 0:
        return

    S1 = 0
    S2 = 0

    if isinstance(args[0], Nematic):
        for nemat in args:
            S1 += nemat.S1
            S2 += nemat.S2

        S1 /= len(args)
        S2 /= len(args)

        return Nematic(S1=S1, S2=S2)
    else:
        for S1S2 in args:
            S1 += S1S2[0]
            S2 += S1S2[1]

        S1 /= len(args)
        S2 /= len(args)


class Nematic():

    # NB angle should be in radians not in degrees could maybe offer the option to have degrees too...
    def __init__(self, S1S2=None, S1=None, S2=None, S0=None, angle=None, base=None, tip=None,
                 center=None):  # TODO maybe some day use base and tip --> revive old TA CODE FOR IT or delete it
        """
        Initializes a Nematic object.

        Args:
            S1S2 (list): List containing the values of S1 and S2.
            S1 (float): Value of S1.
            S2 (float): Value of S2.
            S0 (float): Value of S0.
            angle (float): Value of the angle in radians.
            base (list): List containing the x and y coordinates of the base point.
            tip (list): List containing the x and y coordinates of the tip point.
            center (list): List containing the x and y coordinates of the center point.
        """
        self.rescaling_factor = 1.

        # create a null nematic
        self.S1 = 0
        self.S2 = 0
        self.center = [0, 0]

        if base is not None and tip is not None:
            # print('init',base, tip)
            from epyseg.ta.measurements.TAmeasures import distance_between_points
            center = ((base[0] + tip[0]) / 2., (tip[1] + base[1]) / 2.)
            angle = math.atan2(tip[0] - base[0], tip[1] - base[1])
            S0 = distance_between_points(center, tip)

        # if S1 and S2 are defined then use those
        if S1S2 is not None:
            self.S1 = S1S2[0]
            self.S2 = S1S2[1]
            self.center = [0, 0]
        if S1 is not None:
            self.S1 = S1
        if S2 is not None:
            self.S2 = S2
        if center is not None:
            self.center = center
        if angle is not None and S0 is not None:
            while angle < 0:
                angle += math.pi
            while angle >= math.pi:
                angle -= math.pi
            cos2Theta0 = math.cos(2. * angle);
            sin2Theta0 = math.sin(2. * angle);
            self.S1 = cos2Theta0 * S0
            self.S2 = sin2Theta0 * S0

    def set_rescaling_factor(self, rescaling_factor):
        """
        Sets the rescaling factor for the Nematic object.

        Args:
            rescaling_factor (float): The rescaling factor.
        """
        self.rescaling_factor = rescaling_factor

    def getS0(self):
        """
        Returns the value of S0.

        Returns:
            float: The value of S0.
        """
        return self.rescaling_factor * self.getMagnitude()

    def getMagnitude(self):
        """
        Returns the magnitude of the Nematic object.

        Returns:
            float: The magnitude.
        """
        return math.sqrt((self.S1) * (self.S1) + (self.S2) * (self.S2))

    def getCenter(self):
        """
        Returns the center of the Nematic object.

        Returns:
            list: The x and y coordinates of the center.
        """
        return self.center

    def setCenter(self, center):
        """
        Sets the center of the Nematic object.

        Args:
            center (list): List containing the x and y coordinates of the center.
        """
        self.center = center

    def getS1S2S0(self):
        """
        Returns the values of S1, S2, and S0 as a list.

        Returns:
            list: The values of S1, S2, and S0.
        """
        return [self.S1, self.S2, self.getS0()]

    def getNormal(self):
        """
        Returns the normal for the current Nematic object.

        Returns:
            Nematic: The normal Nematic object.
        """
        return Nematic(S1=-self.S1, S2=-self.S2, center=self.center)

    def getS1S2(self):
        """
        Returns the values of S1 and S2 as a list.

        Returns:
            list: The values of S1 and S2.
        """
        return [self.S1, self.S2]

    def get_angle2(self):
        """
        Get the angle of the nematic in radians using a more elegant calculation.

        Returns:
            float: The angle of the nematic in radians.
        """
        angle = 0.5 * math.atan2(self.S2, self.S1)
        while angle < 0:
            angle += math.pi
        while angle >= math.pi:
            angle -= math.pi
        return angle

    def getAngleInRadians(self):
        """
        Get the orientation of the nematic in radians.

        Returns:
            float: The orientation of the nematic in radians.
        """
        S0 = self.getMagnitude()
        cos2Theta0 = self.S1 / S0
        sinTheta0 = math.sqrt(0.5 * (1. - cos2Theta0))
        sin2Theta0 = self.S2 / S0
        cosTheta0 = 0.5 * sin2Theta0 / sinTheta0
        angle = math.acos(cosTheta0)
        while angle < 0:
            angle += math.pi
        while angle >= math.pi:
            angle -= math.pi
        return angle

    def getBeginAndEnd(self, rescaling_factor=1.):
        """
        Get the beginning and end points of the nematic.

        Args:
            rescaling_factor (float, optional): Rescaling factor. Defaults to 1.

        Returns:
            list: A list containing the beginning and end points of the nematic as [point_begin, point_end].
        """
        S0 = math.sqrt(self.S1 * self.S1 + self.S2 * self.S2)
        if S0 == 0:
            return None
        cos2Theta0 = self.S1 / S0
        sin2Theta0 = self.S2 / S0
        sinTheta0 = math.sqrt(0.5 * (1. - cos2Theta0))
        cosTheta0 = 0.5 * sin2Theta0 / sinTheta0
        scale = rescaling_factor * S0
        xBegin = self.center[1] - scale * cosTheta0
        xEnd = self.center[1] + scale * cosTheta0
        yBegin = self.center[0] - scale * sinTheta0
        yEnd = self.center[0] + scale * sinTheta0
        pts = []
        pts.append([yBegin, xBegin])
        pts.append([yEnd, xEnd])
        return pts

    def toLine2D(self, rescaling_factor=1.):
        """
        Convert the nematic to a Line2D object.

        Args:
            rescaling_factor (float, optional): Rescaling factor. Defaults to 1.

        Returns:
            matplotlib.lines.Line2D: The Line2D object representing the nematic.
        """
        base_n_tip = None
        try:
            base_n_tip = self.getBeginAndEnd(rescaling_factor)
        except:
            return None
        if base_n_tip is None:
            return None
        return Line2D(base_n_tip[0][1], base_n_tip[0][0], base_n_tip[1][1], base_n_tip[1][0])

    def draw(self, img, color=0xFFFF00, stroke=1, rescaling_factor=None):
        """
        Draw the nematic on an image.

        Args:
            img (numpy.ndarray or ImageDraw): The image on which to draw the nematic.
            color (int or str, optional): The color of the nematic. Defaults to 0xFFFF00.
            stroke (int or float or str, optional): The stroke width of the nematic. Defaults to 1.
            rescaling_factor (float, optional): Rescaling factor. Defaults to None.
        """
        if color is None:
            color = 0xFFFF00
        if rescaling_factor is None:
            base_n_tip = self.getBeginAndEnd(self.rescaling_factor)
        else:
            base_n_tip = self.getBeginAndEnd(rescaling_factor)
        if base_n_tip is None:
            return
        if isinstance(img, np.ndarray):
            rr, cc = line(int(math.ceil(base_n_tip[0][0])), int(math.ceil(base_n_tip[0][1])),
                          int(math.ceil(base_n_tip[1][0])), int(math.ceil(base_n_tip[1][1])))
            img[rr, cc] = color
        elif isinstance(img, ImageDraw):
            if isinstance(stroke, str):
                stroke = float(stroke)
            if stroke is None or stroke <= 0:
                stroke = 1
            if stroke < 1:
                stroke = int(1)
            if isinstance(stroke, float):
                stroke = int(round(stroke))
            if isinstance(color, str):
                if color.startswith('#'):
                    color = color.replace('#', '')
                    color = int(color, 16)
            color = ((color >> 16) & 255, (color >> 8) & 255, (color & 255))
            img.line((base_n_tip[0][1], base_n_tip[0][0], base_n_tip[1][1], base_n_tip[1][0]), fill=color, width=stroke)

    def getUnitNematicComponents(self):
        """
        Get the components of the unit Nematic.

        Returns:
            list: A list containing the components of the unit Nematic as [S1_unit, S2_unit].
        """
        S0 = self.getMagnitude()
        return [self.S1 / S0, self.S2 / S0]

    def getUnitNemat(self):
        """
        Get the unit Nematic.

        Returns:
            Nematic: The unit Nematic.
        """
        unit0 = self.getUnitNematicComponents()
        return Nematic(S1=unit0[0], S2=unit0[1], center=self.center)

    def __str__(self):
        """
        Get a string representation of the Nematic object.

        Returns:
            str: The string representation of the Nematic object.
        """
        return str(self.S1) + " " + str(self.S2) + " " + str(self.center) + " " + str(self.getS0()) + " " + str(
            self.getAngleInRadians()) + " " + str(self.getBeginAndEnd()[0]) + " " + str(self.getBeginAndEnd()[1])

if __name__ == "__main__":

    # props = measure.regionprops(
    #     labels, image, extra_properties=[image_height]
    # )

    if True:
        base = (301.77769935043347,        21.280930792689446)
        tip = (230.48494928416977,        715.8784747926309)

        from epyseg.ta.measurements.TAmeasures import distance_between_points

        nematic = Nematic(base=base, tip=tip)

        print(nematic)

        print(nematic.getBeginAndEnd())

        # so if I wanna scale it -> I can easily do that
        print('rescaled', nematic.getBeginAndEnd(rescaling_factor=0.75))

        dist_reduced = distance_between_points(*nematic.getBeginAndEnd(rescaling_factor=0.75))
        orig_dist = distance_between_points(base, tip)
        print(dist_reduced,orig_dist, dist_reduced/orig_dist) # --> all is perfect --> I can really use that

        import sys
        sys.exit(0)

    if True:
        handcorrection = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012/handCorrection.tif')
        handcorrection = label(handcorrection, connectivity=1, background=255)
        rps = regionprops(handcorrection) # , extra_properties=[compute_stretch_nematic_regionprops_extra] --> marche pas car il m'envoir juste un mask et ça ne m'interesse pas...



        first_cell = rps[0]

        # print(rps[0].compute_stretch_nematic_regionprops_extra)
        # print(first_cell.coords.shape)
        # put it in a coordinate list --> TODO --> maybe also allow to pass in the array
        # convert the coords to a numpy array and loop over it
        # print(type(first_cell))
        print(compute_stretch_nematic(first_cell.coords))
        print(compute_stretch_nematic(first_cell))



        sys.exit(0)

    if True:
        Q1 = -42.17042288690961
        Q2 = -58.92645735167784

        xCenter = 449.56
        yCenter = 344.48
        test3 = Nematic(center=[yCenter, xCenter], S1=Q1, S2=Q2)

        print(test3)

        print(test3.get_angle2())
        print(test3.getAngleInRadians())
        # in TA --> -42.17042288690961 -58.92645735167784 Point2D.Double[449.56, 344.48] 72.46152042622275 2.0456200121341603 Point2D.Double[482.68807708785437, 280.03465531986757] Point2D.Double[416.43192291214564, 408.92534468013247]

    if False:
        xCenter = 256
        yCenter = 256
        Q1 = 215.43
        Q2 = -0.4503167411  # //-0.4503167411
        #             //double Q1 = -79.9;
        #             //double Q2 = -75.09;
        #             test3 = Nematic(center=[xCenter, yCenter], S1=Q1, S2=Q2)
        test3 = Nematic(center=[xCenter, yCenter], S0=Q1,
                        angle=Q2)  # seems to work as in TA --> just need invert center coords to fit
        print(test3)

    # TA  133.80650714020177 -168.83691405299712 Point2D.Double[256.0, 256.0] 215.43 2.691275912489793 Point2D.Double[449.95362942365074, 162.2339371979757] Point2D.Double[62.04637057634926, 349.7660628020243]
    # PyTA 133.80650714020177 -168.83691405299712 [256, 256] 215.43 2.691275912489793 [162.2339371979757, 449.95362942365074] [349.7660628020243, 62.04637057634926]

    if False:
        tmp = Nematic(center=[20, 10], S0=10.44030650891055,
                      angle=0.1457283972389337);  # //new Nematic(new Point2D.Double(10, 20), new Point2D.Double(20.329643389031034, 21.516069739638404)) #//new Nematic(-10, 3, new Point2D.Double(10, 20));//new Nematic(-1020, -10000, new Point2D.Double(25, 25));
        print(tmp)
        # //      print(tmp.getAngleInRadians() , tmp.getAngle2() + " " + tmp.getAngleTest());
        # //        tmp.getAngleTest();

        test = Nematic(S1=-10, S2=3, center=[20, 10])
        print(test)
        # //        System.out.println(test);

        # //--> ca marche il y a une tte petite reounding error
        # test2 = Nematic([10, 20], [20.329643389031034, 21.516069739638404]) --> not implemented --> center and any extremity...
        # //        System.out.println(test2);

        test3 = Nematic(center=[20, 10], S0=10.44030650891055, angle=0.1457283972389337)
        # //        System.out.println(test3);

        unit_nemat = test3.getUnitNematicComponents()
        print(unit_nemat)
        print(math.sqrt(unit_nemat[1] * unit_nemat[1] + unit_nemat[0] * unit_nemat[0]))

        print(test3.getUnitNemat().getMagnitude())

    # nb all seems fine now

# 215.43
# 154.19875129088464
# Point2D.Double[449.95362942365074, 162.2339371979757] Point2D.Double[62.04637057634926, 349.7660628020243]
# 69.64703070271749
# 54.08721586034971
# 15.559814842367771
# -15.559814842367771
# Point2D.Double[0.9578262852211513, 0.28734788556634566]
# 1.0
# 1.0

#             System.out.println(test3.getMagnitude());
#             System.out.println(Math.toDegrees(test3.getAngleInRadians()));
#             BufferedImage test = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
#             Graphics2D g2d = test.createGraphics();
#             test3.toMyLine2D(1).drawAndFill(g2d);
#
#             System.out.println(test3.getBeginAndEnd()[0] + " " + test3.getBeginAndEnd()[1]);
#
#             MyLine2D.Double line = new MyLine2D.Double(xCenter - Q1 / 2, yCenter - Q2 / 2, xCenter + Q1, yCenter + Q2);
#             line.setDrawColor(0xFF0000);
#             line.drawAndFill(g2d);
#             g2d.dispose();
#             Saver.popJ(test);
#             try {
#                 Thread.sleep(3000);
#             } catch (InterruptedException ex) {
#                 //  Logger.getLogger(Nematic.class.getName()).log(Level.SEVERE, null, ex);
#             }
#             //System.exit(0);
#         }
#
#         System.out.println(Math.toDegrees(1.21557));
#         System.out.println(Math.toDegrees(0.944));
#
#         System.out.println(Math.toDegrees(1.21557 - 0.944));
#         System.out.println(Math.toDegrees(-1.21557 + 0.944));//System.out.println(Math.toDegrees(0.26777283585356804));
#
# //        ArrayList<String> list = new LoadListeToArrayList().apply("/list.lst");
#         //long start_time = System.currentTimeMillis();
# //        int x1 = 2;
# //        int x2 = 3;
# //        double val = (x1+x2)/2.;
# //        System.out.println(val);
# //        if (true)
# //        return;
#         //--> tjrs le deuxieme
#         Nematic tmp = new Nematic(new Point2D.Double(10, 20), 10.44030650891055, 0.1457283972389337);//new Nematic(new Point2D.Double(10, 20), new Point2D.Double(20.329643389031034, 21.516069739638404));//new Nematic(-10, 3, new Point2D.Double(10, 20));//new Nematic(-1020, -10000, new Point2D.Double(25, 25));
# //        System.out.println(tmp.getAngleInRadians() + " " + tmp.getAngle2() + " " + tmp.getAngleTest());
# //        tmp.getAngleTest();
#
#         Nematic test = new Nematic(-10, 3, new Point2D.Double(10, 20));
# //        System.out.println(test);
#
#         //--> ca marche il y a une tte petite reounding error
#         Nematic test2 = new Nematic(new Point2D.Double(10, 20), new Point2D.Double(20.329643389031034, 21.516069739638404));
# //        System.out.println(test2);
#
#         Nematic test3 = new Nematic(new Point2D.Double(10, 20), 10.44030650891055, 0.1457283972389337);
# //        System.out.println(test3);
#
#         Point2D.Double unit_nemat = test3.getUnitNematicComponents();
#         System.out.println(unit_nemat);
#         System.out.println(Math.sqrt(unit_nemat.x * unit_nemat.x + unit_nemat.y * unit_nemat.y));
#         //--> is ok
#
#         System.out.println(test3.getUnitNemat().getMagnitude());
#
#         //bonne idee il faut tester les nematics
#         //et leur normalisation
# //        test.apply(list);
#         //System.out.println("ellapsed time --> " + (System.currentTimeMillis() - start_time) / 1000.0 + "s");
#         System.exit(0);
#     }
# }
