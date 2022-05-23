# TA implementation of nematic --> need pythonize this


# package Geometry;
#
# import Commons.Point3D;
# import Commons.Saver;
# import MyShapes.MyLine2D;
# import java.awt.Graphics2D;
# import java.awt.Point;
# import java.awt.geom.Point2D;
# import java.awt.image.BufferedImage;
# import java.util.ArrayList;
#
# /**
#  * Nematic class
#  *
#  * @author Benoit Aigouy
#  */
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

#
#   /**
#      *
#      * @param stretch_rescaling_factor
#      * @return the components of the strecth nematic of the flooded area (and
#      * the coordinates of the nematic extremities in case it has to be plotted)
#      */
#     public Point2D.Double[] compute_stretch(double stretch_rescaling_factor) {
#         double count = 0.;
#
#         double S1 = 0.;
#         double S2 = 0.;
#
#         Point2D.Double center = getCenter2D();
# //COMPUTE NEMATIC TENSOR FOR THE STRETCH
#
#         for (int k = 0; k < queueSize; k++) {
#             //GET THE CURRENT POINT (CONSTANT INTENSITY FOR THE STRETCH)
#             //COMPUTE NECESSARY TRIGONOMETRIC QUANTITES FOR EACH POINT
#             double deltaX = (double) flood_x[k] - center.x; //ARE THESE CASTS NECESSARY?
#             double deltaY = (double) flood_y[k] - center.y;
#             double rSquared = deltaX * deltaX + deltaY * deltaY;
#             double cos2Theta = 2. * deltaX * deltaX / rSquared - 1.;
#             double sin2Theta = 2. * deltaX * deltaY / rSquared;
#
#             //ADD THE SUMMANDS TO THEIR RESPECTIVE SUMS...
#             S1 += cos2Theta;
#             S2 += sin2Theta;
#             count++;
#         }
#
#         S1 /= count;//we rescale by the area
#         S2 /= count;
#
#         //COMPUTE THE STRETCH NEMATIC
#         double S0 = Math.sqrt(S1 * S1 + S2 * S2);
#         double cos2ThetaS = S1 / S0;
#         double sin2ThetaS = S2 / S0;
#         double sinThetaS = Math.sqrt(0.5 * (1. - cos2ThetaS));
#         double cosThetaS = 0.5 * sin2ThetaS / sinThetaS;
#
# //IN THE END WE WANT TO DRAW A LINE THROUGH THE CENTER OF THE CELL REPRESENTING THE STRETCH NEMATIC
#         double scale = stretch_rescaling_factor * S0;
#
#         double xDem = center.x - scale * cosThetaS;
#         double xFin = center.x + scale * cosThetaS;
#         double yDem = center.y - scale * sinThetaS;
#         double yFin = center.y + scale * sinThetaS;
#
#         Point2D.Double[] out = new Point2D.Double[3];
#         out[0] = new Point2D.Double(xDem, yDem);
#         out[1] = new Point2D.Double(xFin, yFin);
#         out[2] = new Point2D.Double(S1, S2);
#         return out;
#     }

# Computes a stretch nematic based on cell area
from epyseg.img import Img


# def compute_stretch_nematic_regionprops_extra(regionmask, intensity_image=None):
#     print(type(regionmask))
#     print(regionmask.shape)# --> 11,39 --> not at all what I want
#     print(regionmask)
#
#
#     return compute_stretch_nematic(regionmask)


def compute_stretch_nematic(pointsWithinArea_or_region, normalizeByArea=True):
    if pointsWithinArea_or_region is None:
        return
    if isinstance(pointsWithinArea_or_region, RegionProperties):
        pointsWithinArea_or_region = pointsWithinArea_or_region.coords
    # center_x = 0
    # center_y = 0
    # counter = 0
    # for point in pointsWithinArea_or_region:
    #     center_x += point[1]
    #     center_y += point[0]
    #     counter += 1
    # center_x/=counter
    # center_y/=counter

    # print(center_x, center_y)


    # print(np.average(pointsWithinArea_or_region, axis=0))
    S1,S2,center = _compute_stretch_nematic(pointsWithinArea_or_region,normalizeByArea=normalizeByArea)
    return Nematic(S1=S1, S2=S2, center=center)

# @njit
def _compute_stretch_nematic(pointsWithinArea_or_region, normalizeByArea=True):
    # center_y = 0.
    # center_x=0.
    center_y, center_x = np.average(pointsWithinArea_or_region, axis=0)

    # center_x /= pointsWithinArea_or_region.shape[0]
    # center_y /= pointsWithinArea_or_region.shape[0]

    # print(center_x, center_y)

    center = [center_y, center_x]

    S1 = 0.
    S2 = 0.

    # I guess it's really time to recode this properly using numpy --> do that when I have time
    # MEGA TODO --> CONVERT IT TO NUMPY BUT OK FOR NOW
    # //COMPUTE NEMATIC TENSOR FOR THE STRETCH
    for point in pointsWithinArea_or_region:
        # //GET THE CURRENT POINT (CONSTANT INTENSITY FOR THE STRETCH)
        # //Point pt = points_unsrt.get(k);
        #
        # //COMPUTE NECESSARY TRIGONOMETRIC QUANTITES FOR EACH POINT

        deltaX = float(point[1]) - center_x  # //ARE THESE CASTS NECESSARY?
        deltaY = float(point[0]) - center_y
        rSquared = deltaX * deltaX + deltaY * deltaY
        cos2Theta = 2. * deltaX * deltaX / rSquared - 1.
        sin2Theta = 2. * deltaX * deltaY / rSquared

        # //ADD THE SUMMANDS TO THEIR RESPECTIVE SUMS...
        S1 += cos2Theta
        S2 += sin2Theta

    # //--> MAYBE REMOVE THIS TO MAKE IT SIZE DEPENDENT OR ADD A BOOLEAN
    if normalizeByArea:
        S1 /= pointsWithinArea_or_region.shape[0]
        S2 /= pointsWithinArea_or_region.shape[0]
    return S1,S2, center

# builds an average nematic from a series of nematics
# build from other nemats and/or their S1S2 components
# never tried --> really need do it!!!
# shall I offer a center ???
def compute_average_nematic(*args):
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
        self.rescaling_factor = 1.

        # create a null nematic
        self.S1 = 0
        self.S2 = 0
        self.center = [0, 0]
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
        self.rescaling_factor = rescaling_factor

    #     /**
    #      * This constructor converts a vector to a nematic
    #      *
    #      * @param v vector we want to convert to a nematic
    #      */
    #     public Nematic(Vector v) {
    #         this(v.getBase(), v.getTip());
    #     }
    #     /**
    #      * Nematic constructor
    #      *
    #      * @param center center of the nematic
    #      * @param any_extremity_pt coordinates of any extremity of the neamtic
    #      */
    #     public Nematic(Point2D.Double center, Point2D.Double any_extremity_pt) {
    #         this(center, any_extremity_pt.distance(center), Math.atan2(any_extremity_pt.y - center.y, any_extremity_pt.x - center.x));
    #     }
    #
    #     /**
    #      * Nematic constructor
    #      *
    #      * @param base
    #      * @param tip
    #      * @param inutile
    #      */
    #     public Nematic(Point2D.Double base, Point2D.Double tip, boolean inutile) {
    #         this(new Point2D.Double((base.x + tip.x) / 2., (tip.y + base.y) / 2.), tip);
    #     }
    #

    #
    #     public Nematic(ArrayList<Point2D.Double> S1S2) {
    #         this(S1S2.toArray(new Point2D.Double[S1S2.size()]));
    #     }
    #
    #     /**
    #      * builds an average nematic from a series of nematics
    #      *
    #      * @param S1andS2Components
    #      * @since <B>Tissue Analyzer 1.0</B>
    #      */
    #     public Nematic(Nematic... S1andS2Components) {
    #         if (S1andS2Components == null || S1andS2Components.length == 0) {
    #             return;
    #         }
    #         for (Nematic double1 : S1andS2Components) {
    #             S1 += double1.S1;
    #             S2 += double1.S2;
    #         }
    #         S1 /= S1andS2Components.length;
    #         S2 /= S1andS2Components.length;
    #     }
    #
    #     /**
    #      * builds an average nematic from a series of nematics
    #      *
    #      * @param S1andS2Components
    #      * @since <B>Tissue Analyzer 1.0</B>
    #      */
    #     public Nematic(Point2D.Double... S1andS2Components) {
    #         if (S1andS2Components == null || S1andS2Components.length == 0) {
    #             return;
    #         }
    #         for (Point2D.Double double1 : S1andS2Components) {
    #             S1 += double1.x;
    #             S2 += double1.y;
    #         }
    #         S1 /= (double) S1andS2Components.length;
    #         S2 /= (double) S1andS2Components.length;
    #     }
    #

    def getS0(self):
        return self.getMagnitude()

    # return the magnitude of the nematic
    def getMagnitude(self):
        return math.sqrt((self.S1) * (self.S1) + (self.S2) * (self.S2))

    # return the center of the nematic
    def getCenter(self):
        return self.center

    #
    def setCenter(self, center):
        self.center = center

    #
    #     /**
    #      *
    #      * @return a point3D containing S1, S2 and the magnitude of the nematic
    #      */
    def getS1S2S0(self):
        return [self.S1, self.S2, self.getS0()]

    # return the normal for the current nematic
    def getNormal(self):
        return Nematic(S1=-self.S1, S2=-self.S2, center=self.center)

    # return the components of the nematic
    def getS1S2(self):
        return [self.S1, self.S2]

    # TODO replace the complex acos original stuff below with that more elegant code
    def get_angle2(self):
        angle = 0.5 * math.atan2(self.S2, self.S1)
        while angle < 0:
            angle += math.pi
        while angle >= math.pi:
            angle -= math.pi
        return angle

    # return the orientation of the nematic in radians
    def getAngleInRadians(self):
        S0 = self.getMagnitude()
        cos2Theta0 = self.S1 / S0
        sinTheta0 = math.sqrt(0.5 * (1. - cos2Theta0))
        sin2Theta0 = self.S2 / S0
        cosTheta0 = 0.5 * sin2Theta0 / sinTheta0
        angle = math.acos(cosTheta0)
        while (angle < 0):
            angle += math.pi
        while (angle >= math.pi):
            angle -= math.pi
        return angle

    #
    #     /**
    #      *
    #      * @param rescaling_factor
    #      * @return the coordinates of the base of the nematic after rescaling
    #      */
    #     public Point2D.Double getBase(double rescaling_factor) {
    #         double S0 = Math.sqrt(S1 * S1 + S2 * S2);
    #         double cos2Theta0 = S1 / S0;
    #         double sin2Theta0 = S2 / S0;
    #         double sinTheta0 = Math.sqrt(0.5 * (1. - cos2Theta0));
    #         double cosTheta0 = 0.5 * sin2Theta0 / sinTheta0;
    #         double scale = rescaling_factor * S0;
    #         double xDem = center.x - scale * cosTheta0;
    #         double yDem = center.y - scale * sinTheta0;
    #         return new Point2D.Double(xDem, yDem);
    #     }
    #
    #     /**
    #      *
    #      * @param rescaling_factor
    #      * @return the coordinates of the tip of the nematic after rescaling
    #      */
    #     public Point2D.Double getTip(double rescaling_factor) {
    #         double S0 = Math.sqrt(S1 * S1 + S2 * S2);
    #         double cos2Theta0 = S1 / S0;
    #         double sin2Theta0 = S2 / S0;
    #         double sinTheta0 = Math.sqrt(0.5 * (1. - cos2Theta0));
    #         double cosTheta0 = 0.5 * sin2Theta0 / sinTheta0;
    #         double scale = rescaling_factor * S0;
    #         double xFin = center.x + scale * cosTheta0;
    #         double yFin = center.y + scale * sinTheta0;
    #         return new Point2D.Double(xFin, yFin);
    #     }
    #
    #     /**
    #      *
    #      * @return the coordinates of the base and tip of the nematic
    #      */
    #     public Point2D.Double[] getBeginAndEnd() {
    #         return getBeginAndEnd(1.);
    #     }
    #
    #     /**
    #      * @param rescaling_factor
    #      * @return the coordinates of the base and tip of the nematic after
    #      * rescaling
    #      */
    def getBeginAndEnd(self, rescaling_factor=1.):
        S0 = math.sqrt(self.S1 * self.S1 + self.S2 * self.S2)
        if S0 == 0:
            # return [[self.center[0], self.center[1]],[self.center[0], self.center[1]]]
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

    #
    #     /**
    #      * Converts a nematic to a MyLine2D
    #      *
    #      * @param rescaling_factor
    #      * @return a vectorial line object corresponding to the nematic
    #      */
    #     public MyLine2D.Double toMyLine2D(double rescaling_factor) {
    #         Point2D.Double[] base_n_tip = getBeginAndEnd(rescaling_factor);
    #         return new MyLine2D.Double(base_n_tip[0], base_n_tip[1]);
    #     }
    #
    #     public void draw(Graphics2D g2d) {
    #         toMyLine2D(1).drawAndFill(g2d);
    #     }
    #
    #     public boolean isNaN() {
    #         return (Double.isNaN(S1) || Double.isNaN(S2));
    #     }
    #

    # drawing in python over numpy array https://stackoverflow.com/questions/28647383/numpy-compatible-image-drawing-library --> maybe try the other libs such as wand because may be interesting

    def toLine2D(self, rescaling_factor=1.):
        base_n_tip = None
        try:
             base_n_tip =self.getBeginAndEnd(rescaling_factor)
        except:
            # traceback.print_exc()
            # print('error nematic could not be converted to Line2D')
            return None

        # print('self.S1, self.S2',self.S1, self.S2)
        if base_n_tip is None:
            return None
        # line(int(math.ceil(base_n_tip[0][0])), int(math.ceil(base_n_tip[0][1])),
        #      int(math.ceil(base_n_tip[0][1])), int(math.ceil(base_n_tip[1][1])))
        # print(base_n_tip[0][0], base_n_tip[0][1],base_n_tip[1][1], base_n_tip[1][1])
        return Line2D(base_n_tip[0][1], base_n_tip[0][0], base_n_tip[1][1], base_n_tip[1][0]) # KEEP NB BE careful --> first x then y and not the opposite --> maybe change this because dangerous in python where everything is inverted but then need change everythng


    # nb please set the rescaling factor or set it here
    def draw(self, img, color=0xFFFF00, stroke=1, rescaling_factor=None):
        if color is None:
            color = 0xFFFF00
        if rescaling_factor is None:
            base_n_tip = self.getBeginAndEnd(self.rescaling_factor)
        else:
            base_n_tip = self.getBeginAndEnd(rescaling_factor)
        if base_n_tip is None:
            # null nematic --> we don't plot it
            return
        if isinstance(img, np.ndarray):
            # draw the nematic on the image

            # new
            # MyLine2D.Double(base_n_tip[0], base_n_tip[1]);
            rr, cc = line(int(math.ceil(base_n_tip[0][0])), int(math.ceil(base_n_tip[0][1])),
                          int(math.ceil(base_n_tip[1][0])), int(math.ceil(base_n_tip[1][1])))


            # if out of bonds --> errors --> really sucks --> see how I can fix that !!!
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
                    # parse it or maybe best is to convert the entire column rather than do things here
                    color = color.replace('#', '')
                    color = int(color, 16)

            color = ((color >> 16) & 255, (color >> 8) & 255, (color & 255))
            # print(color)

            # print(base_n_tip)
            # is there a bug ???? --> yes the coords must be given as x and y --> opposite of numpy!!!!
            img.line((base_n_tip[0][1], base_n_tip[0][0], base_n_tip[1][1], base_n_tip[1][0]), fill=color, width=stroke)
            # img.line((350, 200, 450, 100), fill=(255, 255, 0), width=10)

        #         return new MyLine2D.Double(base_n_tip[0], base_n_tip[1]);

    # return the components of the unit Nematic
    def getUnitNematicComponents(self):
        S0 = self.getMagnitude()
        return [self.S1 / S0, self.S2 / S0]

    #
    #     /**
    #      *
    #      * @return the unit nematic
    #      */
    # return the unit nematic
    def getUnitNemat(self):
        unit0 = self.getUnitNematicComponents()
        return Nematic(S1=unit0[0], S2=unit0[1], center=self.center)

    def __str__(self):
        return str(self.S1) + " " + str(self.S2) + " " + str(self.center) + " " + str(self.getS0()) + " " + str(
            self.getAngleInRadians()) + " " + str(self.getBeginAndEnd()[0]) + " " + str(self.getBeginAndEnd()[1])
        # return (S1 + " " + S2 + " " + center + " " + this.getS0() + " " + this.getAngleInRadians() + " " +
        #         this.getBeginAndEnd(1)[0] + " " + this.getBeginAndEnd(1)[1]);





if __name__ == "__main__":

    # props = measure.regionprops(
    #     labels, image, extra_properties=[image_height]
    # )

    if True:
        handcorrection = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/handCorrection.tif')
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
