# try to find a code that finds the new scaling so that the image fits to the new size
# assume all images have the same height in a row and the same width in a column

# can I not even store everything as a big numpy array ???

# test with 3 rectangles

# keep AR the same --> need just apply a scaling to the image and in fact to all images the same, by the way

#
# def find_scaling_to_fit():
#     incompressible_length = (len(widths)-1)*space_between_images
#     image_width_to_fit_not_counting_incompressible_length = final_width-incompressible_length
#     sum_individual = sum(widths)
#     common_scaling = image_width_to_fit_not_counting_incompressible_length/sum_individual
#
#     print(common_scaling)
#
#     # print(widths*common_scaling)
#     corrected_widths = [w*common_scaling for w in widths]
#     print(widths)
#     print(corrected_widths)
#     print(sum(corrected_widths))
#     print(sum(corrected_widths)+incompressible_length)
#     print(incompressible_length)
#     # print()

    # ça marche super en fait et c'est vraiment facile

# computes the rescaling factor so that images get rescaled to fit the desired dimension --> much smarter that way than using brute force!!!
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import QRectF


# nb the pb asssumes AR is kept constant which is true for images but not with panels containing images and incompressible space between them --> need some extra computation then


common_value = 0


def get_common_scaling_factor(dimensions_in_the_desired_dimension, desired_final_size, size_of_incompressible_elements_in_dimension=3):
    incompressible_length = (len(dimensions_in_the_desired_dimension) - 1) * size_of_incompressible_elements_in_dimension
    image_width_to_fit_not_counting_incompressible_length = desired_final_size - incompressible_length
    sum_individual = sum(dimensions_in_the_desired_dimension)
    common_scaling = image_width_to_fit_not_counting_incompressible_length / sum_individual

    # print(common_scaling)

    # print(widths*common_scaling)
    # corrected_widths = [w * common_scaling for w in dimensions_in_the_desired_dimension]
    # print(dimensions_in_the_desired_dimension)
    # print(corrected_widths)
    # print(sum(corrected_widths))
    # print(sum(corrected_widths) + incompressible_length)
    # print(incompressible_length)

    return common_scaling

#########################################################KEEP EQUATIONS TO SOLVE THE CHANGE IN SIZE
# final_width = width_obj1 + width_obj2+ width_obj3 + (nb_of_objs-1)*space
# new_width = new_width_obj1 + new_width_obj2+ new_width_obj3 + (nb_of_objs-1)*space
# new_width = width_obj1*scaling + new_width_obj2*scaling+ new_width_obj3*scaling + (nb_of_objs-1)*space
# height = same whatever the condition but can change --> need solve the equa so that height is the same while width obeys the new stuff
# heightA = heightB = heightC # always
# comment relier la width à la height ???

# final_AR = widthA/heightA = (incomp_widthA+widthA_non_incom)/(incomp_heightA+heightA_non_incom)
# heightA =(incomp_heightA+heightA_non_incom)
# widthA = (incomp_widthA+withA_non_incom)
# AR = withA_non_incom/heightA_non_incom
# heightA_non_incomp = widthA_non_incomp/AR
# heightA = incomp_heightA+(widthA_non_incomp/AR)
# widthA = (incomp_widthA+withA_non_incom)
# --> with that can I compute the best change for all
# final_AR = ((scaling_x*current_widthA) +widthA_non_incom)/((scaling_y*current_heightA)+heightA_non_incom)
# new_widthA = ((scaling_xA*current_widthA) +widthA_non_incom)
# new_heightA = ((scaling_yA*current_heightA)+heightA_non_incom)
# I need find both scalings at the same time
# reorder equa for scaling -> TODO
# scaling_xA = (new_widthA - widthA_non_incom)/current_widthA
# scaling_yA = (current_heightA - heightA_non_incom)/current_heightA
# need find both so that everything fits
# --> easy in fact and that is what I need to have
# height is always related to width too given the incompressible AR ratio --> YES IN FACT IT MUST BE
# heightA = heightB = heightC # whatever happens the images all have the same height in a row
# so by definition
# height A is fixed but yet to an unknown value
# heightA = heightB = heightC
# width is fixed and height must be adjusted but the same in all cases
# --> how does one compute that
# that must be doable but complex
# j'ai 6 inconnues --> mes scalings en x et en y et j'ai une taille constante pour un et une egalite pour l'autre
#


# final AR = ((scalingx * old_compressible_dim_x)+incomp x)/((scalingy * old_compressible_dim_y)+incomp y)) =  ((scalingx * old_compressible_dim_x)+incomp x)/((scalingy * (old_compressible_dim_x/AR))+incomp y))
# finalAR = final_dimx/final dimy
# final_dimx/final dimy = ((scalingx * old_compressible_dim_x)+incomp x)/((scalingy * (old_compressible_dim_x/AR))+incomp y))
# final_dimx = final dimy* (((scalingx * old_compressible_dim_x)+incomp x)/((scalingy * (old_compressible_dim_x/AR))+incomp y)))
# final_dimy = final dimx/(((scalingx * old_compressible_dim_x)+incomp x)/((scalingy * (old_compressible_dim_x/AR))+incomp y)))
# final_dimy1 = final_dimy2 =final_dimy3
# the early things are known and either the final height or width is known --> not true but i know they are eaqual --> so I can solve it
# and get all the parameters

# sum of dims + sum incomp = desired width
# sum(incomp)+sum(images) = desired_width
# I need get the 3 heights that would fulfill the criterion I want


# same height --> ????
# how do I do that






# faire ça

# start with just two images and solve it
# heightA = heightB
# final_width = widthA + widthB + incomp_width
# need find widthA and widthB so that the heights are the same
# final_width = scalingxA * original_widthA +scalingxB * original_widthB  + incomp_width
# final_height = scalingyA * original_heightA+scalingyB * original_heightB
# how can I relate width and height
# final_AR is unknown

# in fact I don't find any other way than brute force but there has to be a way ??? --> too bad I forgot my basic math classes
# ratio = sizex/sizey --> ce n'est pas fixe mais ça depend des
# the pb is that this ratio is unknown


# height +  ((leny-1)*space) = AR*width + ((lenx-1)*space)
# height = AR*width + ((lenx-1)*space)-  ((leny-1)*space)
# heights of the three should be equal --> can compute
# y = ax+b # image1 --> what I want is a ???
# y = dx'+c
# all same height
#dx'+c = ax+b
# sum of width = fixed size
# dx'+c + ax+b= 512
# c and b are fixed
# dx'+ax = 512-c-b # --> equation 1
# need another equation
# need find d and b
# est ce que c'est ça ???
# https://calculis.net/systeme-n-equations#solutions

# dx'+c = ax+b
# d = (ax+b-c)/x'
# a = (dx'+c-b)/x
# deux equations à deux inconnues
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html or that ???
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html # or that

# or solve for different heights
# y1 = a1*x1+b1
# y2 = a2*x2+b2
# y1 = y2
# a1x1 = a2*x2+b1-b2
# a1*x1+b1 + a2*x2+b2 + c = 512
# https://www.youtube.com/watch?v=QkkpcVbNLVE --> je pense que c'est ce que je cherche
# du coup voir comment le resoudre pratiquement
# or brute force solve it numerically
# see the smart and fastest way to do that
# i need find a1 and x1 and a2 and x2 -> need another equation then



# autre solution faire usr un systeme packé et le changer en non packé --> marche si et seulement si le truc est incompressible dans une seule dimension
#


















#########################################################KEEP EQUATIONS TO SOLVE THE CHANGE IN SIZE

# worst case --> I can brute force it a bit but I would love not to
# or need a post process correction to fit in same height but pb is that it will also change width and therefore I will have trouble beacuse it will change width and I will need do that recursively until I reach the global minima
# -> think about it
# think about it

# AR = (real_width + incomp_width)/(real_height + incomp_height)
# incomp_width =
# width = sum_width_objects
# probably easy --> I just need to have in the formula the AR
# it should fit in the given space
# if same size --> incompressible fits
def get_common_scaling_factor_taking_incompressibility_into_account(dimensions_in_the_desired_dimension, desired_final_size, size_of_incompressible_elements_in_dimension=3):
    incompressible_length = (len(dimensions_in_the_desired_dimension) - 1) * size_of_incompressible_elements_in_dimension
    image_width_to_fit_not_counting_incompressible_length = desired_final_size - incompressible_length
    sum_individual = sum(dimensions_in_the_desired_dimension)
    common_scaling = image_width_to_fit_not_counting_incompressible_length / sum_individual

    # print(common_scaling)

    # print(widths*common_scaling)
    # corrected_widths = [w * common_scaling for w in dimensions_in_the_desired_dimension]
    # print(dimensions_in_the_desired_dimension)
    # print(corrected_widths)
    # print(sum(corrected_widths))
    # print(sum(corrected_widths) + incompressible_length)
    # print(incompressible_length)

    return common_scaling

def get_master_bounds2(group_of_shapes):
    bounds = QRectF()
    max_width = 0
    max_height = 0
    min_x=100000000
    min_y=100000000
    for shape in group_of_shapes:
       rect = shape.boundingRect()
       max_width = max(max_width, rect.x()+rect.width())
       max_height = max(max_height, rect.y()+rect.height())
       min_x = min(min_x, rect.x())
       min_y = min(min_y, rect.y())
    bounds.setX(min_x)
    bounds.setY(min_y)
    bounds.setWidth(max_width-min_x)
    bounds.setHeight(max_height-min_y)
    return bounds
#
# gets the master rect of a list of objects
# Nb should I allow negative bounds ??? --> maybe not in fact
def get_master_bounds(group_of_shapes):
    bounds = QRectF()
    max_width = 0
    max_height = 0
    for shape in group_of_shapes:
       rect = shape.boundingRect()
       max_width = max(max_width, rect.x()+rect.width())
       max_height = max(max_height, rect.y()+rect.height())
    bounds.setWidth(max_width)
    bounds.setHeight(max_height)
    return bounds

def preview( shapes_to_draw):
    if not isinstance(shapes_to_draw, list):
        shapes_to_draw=[shapes_to_draw]
    from epyseg.figure.ezfig import MyWidget
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # app = QGuiApplication(sys.argv)
    widget = MyWidget()

    widget.shapes_to_draw = shapes_to_draw
    widget.update_size()

    widget.show()

    sys.exit(app.exec_())


if __name__ == '__main__':


    if True:
        # test solving equation
        import numpy as np

        # images have same height --> can fix that to the desired value
        # y = a0*x0+b0
        # y = a1*x1+b1
        # y-b0 = a0*x0
        # y-b1 = a1*x1

        # image 1 128*256
        # image 2 128*128
        # image 3 64*128
        # 256 = a0 * 128 --> 128 = a0 * 64 # image1
        # 128 = a1 * 128 -> im2
        # 64 = a2* 128

        # ça ne marche pas
        # manque une equa qui me donne ce que je veux
        a = np.array([[64, 0,0], [0, 128,0],[0, 0,128]]) #--> indeed need scale it by a factor2 --> # why not 0.5
        b = np.array([128,128,64])
        x = np.linalg.solve(a, b)
        print(x)


        # somme des trois -->
        # a = np.array([[64, 0, 0], [0, 128, 0], [0, 0, 128]])  # --> indeed need scale it by a factor2 --> # why not 0.5
        # b = np.array([128, 128, 64])
        # x = np.linalg.solve(a, b)
        # print(x)
        # j'ai n equations --> comment ensuite les combiner

        # sinon calculer les images à trois hauteurs differentes et calculer comment obtenir le truc à la hauteur souhaitée --> à faire
        # facile de faire des set to height et de mesurer les valeurs
        # a = np.array([[512, 0,0], [0, 512,0], [512,512,1024]])  # --> indeed need scale it by a factor2 --> # why not 0.5
        # b = np.array([512, 512, 1024])
        # x = np.linalg.solve(a, b)
        # print(x)

        a = np.array([[64, 0], [64, 128]]) #--> indeed need scale it by a factor2 --> # why not 0.5
        b = np.array([128,256])
        # 2 et 1 -->
        # 256 = 2*128+128 # --> ok but not what I want
        x = np.linalg.solve(a, b)
        print(x)

        # n = len(ni)
        # from sympy import symbols, solve
        #
        # A = symbols('a0:%d' % n)
        # equations = []  # set up equations list
        # for i in range(n):
        #     newsum = (rSv + (A[i] + ni[1]) * nb[i])
        #     equations.append(Eq((A[i] + d2sni[i]) * nb[i] / newsum, S[i]))  # append instead of overwriting equations.
        # sol = solve(equations)

        # is that what I want
        # from scipy.optimize import fsolve
        # import math
        #
        #
        # def equations(p):
        #     x, y = p
        #     return (x + y ** 2 - 4, math.exp(x) + x * y - 3)
        #
        #
        # x, y = fsolve(equations, (1, 1))
        #
        # print(equations((x, y)))



        # https://docs.sympy.org/dev/modules/solvers/solvers.html#sympy.solvers.solvers.nsolve
        # import sympy
        from sympy import nsolve, exp, Symbol
        # The first argument is a list of equations, the second is list of variables and the third is an initial guess.
        y = Symbol('y')
        x= Symbol('x')
        # y = a*x
        # y-ax
        # comment faire ????
        print(nsolve([x + y ** 2 - 4, exp(x) + x * y - 3], [x, y], [1, 1]))
        # --> ok that works
        # can I formulate my math this way

        # final_dimx = final dimy* (((scalingx * old_compressible_dim_x)+incomp x)/((scalingy * (old_compressible_dim_x/AR))+incomp y)))
        # final_dimy = final dimx/(((scalingx * old_compressible_dim_x)+incomp x)/((scalingy * (old_compressible_dim_x/AR))+incomp y)))
        # final_dimy1 = final_dimy2 =final_dimy3
        # sum of x = final_dimx+final dimx2+final dimx3 + sum(incomp)


        # images 512*512 AR 1
        # image 128*512 AR 0.25
        # image 256*256 AR 1
        # incomp width = 3*2
        # desired width = 512
        a1 = Symbol('a1')
        a2 = Symbol('a2')
        a3 = Symbol('a3')


        # cette version marche mais faut voir
        print('test2',nsolve([a1*512+a2*128-(512-3), a1*1*512+0-a2*(1/0.25)*256-0], [a1, a2], [1, 1]))
        # print('test3',nsolve([a1*512+a2*128+a3*256-(512-6), (a1*1*512-0)+(a2*(1/0.25)*256-0)-2*(a3*1*256-0), 2*(a1*1*512-0)-(a2*(1/0.25)*256-0)-(a3*1*256-0)], [a1, a2,a3], [1, 1,1]))
        # print('test4',nsolve([(a1*512+a2*128+a3*256)-(512-6), (a1*1*512-0)-(a2*(1/0.25)*256-0), (a2*(1/0.25)*256-0)-(a3*1*256-0)], [a1, a2,a3], [1, 1,1])) # ça a l'air de marche en fait --> à voir
        print('test4',nsolve([a1*512+a2*128+a3*256+6-512, (a1*1*512+0)-(a2*(1/0.25)*128+0), (a2*(1/0.25)*128-0)-(a3*1*256-0)], [a1, a2,a3], [1, 1,1])) # ça a l'air de marche en fait --> à voir
        # print('test4',nsolve([a1*512+a2*128+a3*256+6-512, (a1*1*512+0)-(a2*0.25*128+0), (a2*0.25*128-0)-(a3*1*256-0)], [a1, a2,a3], [1, 1,1])) # ça a l'air de marche en fait --> à voir
        # 235.52 + 29.764705882352896 +238.117647058823424 # --> marche pas du tout en fait
        # --> un bug ???
        # 224.768 + 224.768 + 224.888888888888832 # colle pas mais la hauteur est en effet la meme --> mais pb pas bonne largeur --> bug dans mon equa ???

        # test4 Matrix([[0.0470610119047619], [0.752976190476190], [1.50595238095238]])
        print('test5',nsolve([a1*512+a2*128+a3*256+6-512, (a1*1*512+0)-(a2*(1/0.25)*128+0), (a2*(1/0.25)*128-0)-(a3*1*256-0)], [a1, a2,a3], [1, 2,2])) # ça a l'air de marche en fait --> à voir
        test = nsolve([a1*512+a2*128+a3*256+6-512, (a1*1*512+0)-(a2*(1/0.25)*128+0), (a2*(1/0.25)*128-0)-(a3*1*256-0)], [a1, a2,a3], [1, 2,2])
        dims_x = [512,128,256]
        dims_y = [512,512,256]
        summ_x = []
        summ_y = []
        for iii,val in enumerate(test):
            print('-->', dims_x[iii]*val, val)
            summ_x.append(dims_x[iii]*val)
            summ_y.append(dims_y[iii]*val)

        print('sumX', sum(summ_x)+6) # --> si ça marche en fait
        print('sumY', summ_y) # --> si ça marche en fait

        # --> Matrix([[0.883680555555556], [0.441840277777778]])

        # 153.6
        # 339.2
        # --> 492
        # --> a l'air ok --> voir sure un truc plus complexe
        # voir aussi comment générer les equas de maniere 100% autonome
        # --> pas trop dur je pense...

        # all seems ok --> just try with a more complex shape to see if that works
        # see also how to handle groups --> is that easy or do they need be decomposed into smaller things ???

        if True:
            from epyseg.draw.shapes.image2d import Image2D
            from epyseg.figure.column import Column
            from epyseg.figure.row import Row
            # from epyseg.draw.shapes.image2d import Image2D
            img1 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')
            img2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
            img3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
            img4 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')
            img5 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/04.png')
            img6 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/05.png')
            img7 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/06.png')
            img8 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/07.png')
            img9 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/08.png')


            images = [img1, img2, img3, img4]

            # get AR and stuff for all the images --> TODO
            # solution (128.00066196944888, 42.80999999999984, 42.80999999999984)
            space = 3
            col1 = Column(img1, img2, space=space)
            # col2 = Column(img3, img4, space=space)
            # col1 /= col2

            row1 = Row(img5, img6,  space=space) # img7,

            col1 |= row1
            # there is a bug in there in fact --> it starts not to work with very complex objects
            # --> try it and do my best
            # somehow need same height and also stuff

            # col1.setToHeight(256) # --> ça marche par contre la set
            # col1.setToWidth(256) # --> ça marche par contre la set to width ne marche pas --> probably needs be fixed because the user will want that

            # col1.setToWidth(128)

            # get width for all and incomp for all

            # (128.01439372226696, 40.469999999999914, 40.46999999999991) --> avec juste 4 images --> voir la vraie solution
            print(img1.width(False))
            print(img2.width(False))
            print(img5.width(False))
            print(img6.width(False))
            # print(img7.width(False))

            print('h',img1.height(False))
            print(img2.height(False))
            print(img5.height(False))
            print(img6.height(False))
            # print(img7.height(False))

            print('AR', img1.width(False)/img1.height(False))
            print(img2.width(False)/img2.height(False))
            print(img5.width(False)/img5.height(False))
            print(img6.width(False)/img6.height(False))
            # print(img7.width(False)/img7.height(False))



            # incomp height = 3 pr col et 0 pr autres
            # incomp_w=6

            # test = nsolve([a1 * 512 + a2 * 128 + a3 * 256 + 6 - 512, (a1 * 1 * 512 + 0) - (a2 * (1 / 0.25) * 128 + 0),(a2 * (1 / 0.25) * 128 - 0) - (a3 * 1 * 256 - 0)], [a1, a2, a3], [1, 1, 1])
            # need get equa per stuff
            # --> see how to do that

            # get equa for row
            # if nested equa --> TODO --> return nested equa
            # height

            # 288.0
            # 145.0
            # 299.0
            # 299.0
            # h
            # 322.0
            # 303.0
            # 352.0
            # 299.0
            # AR
            # 0.8944099378881988
            # 0.47854785478547857
            # 0.8494318181818182
            # 1.0

            a1 = Symbol('a1')
            a2 = Symbol('a2')
            a3 = Symbol('a3')
            a4 = Symbol('a4')

            # for each row in stuff --> do for all elements of the row


            # test = nsolve([a1 * 512 + a2 * 128 + a3 * 256 + 6 - 512, (a1 * 1 * 512 + 0) - (a2 * (1 / 0.25) * 128 + 0),(a2 * (1 / 0.25) * 128 - 0) - (a3 * 1 * 256 - 0)], [a1, a2, a3], [1, 1, 1])

            # need solve for width and for height --> need different equas for rows and columns


            # test = nsolve([a1 * 288 + a2 * 145 + a3 * 299 + 6 - 512, (a1 * (1/0.8944099378881988) * 512 + 0) - (a2 * (1 / 0.47854785478547857) * 145 + 0),(a2 *(1 / 0.47854785478547857) * 145 - 0) - (a3 * (1/0.8494318181818182) * 299 - 0)], [a1, a2, a3], [1, 1, 1])
            # dims_x = [288,145, 299]
            # dims_y = [322, 303, 352]
            # summ_x = []
            # summ_y = []
            # for iii, val in enumerate(test):
            #     print('-->', dims_x[iii] * val, val)
            #     summ_x.append(dims_x[iii] * val)
            #     summ_y.append(dims_y[iii] * val)
            #
            # print('sumX', sum(summ_x) + 6)  # --> si ça marche en fait
            # print('sumY', summ_y)  # --> si ça marche en fait

            #ça marche mais si une equa est complexe alors il faut mettre l'equa complexe dedans avec ses variables aussi
            # --> voir comment faire --> faut faire generer une equa par truc ...
            # --> voir comment faire et faire generer les equas par row et par col --> des sous equas du coup

            # peut etre résoudre la meme pour chaque colonne en fait


            # for a ROW
            dims_x = []
            dims_y = []
            summ_x = []
            summ_y = []
            equations_single = []
            expressions_a = []
            expressions_b = []
            expressions_c = []
            equations = []
            variables = []
            estimates = []
            for iii, img in enumerate(images):
                dims_x.append(img.width(False))
                dims_y.append(img.height(False))
                AR = img.width(False)/img.height(False)

                variable = Symbol('a'+str(iii))
                variables.append(variable)
                estimates.append(1)
                equations_single.append(variable*(1./AR)*img.width(False)+0)
                expr = variable * (1. / AR) * img.width(False) + 0
                expressions_a.append(expr)
                expr= variable *img.width(False)
                expressions_b.append(expr)

                # expr = variable*AR*img.height(False) # this is the new width
                # expr = variable*AR*img.height(False) # this is the new width
                #

            # add an equa that is the sum of all
            # final_equa = # sum of all and substraction
            # final_equa = []

            print(estimates, equations_single, variables, expressions_b)
            final_equa = None
            for expr in expressions_b:
                if final_equa is None:
                    final_equa = expr
                else:
                    final_equa+=expr



            # --> equas are all heights should be same

            desired_width = 512
            final_equa+= (len(images)-1)*space -desired_width
            print(final_equa)  # --> ok that works

            # do by pairs of equal
            pairs_of_equations = None
            other_equas = []
            for iii in range(1,len(expressions_a)):
                # print(iii)
                # first - sec = 0
                equa =  expressions_a[0]-expressions_a[iii]
                other_equas.append(equa)

            other_equas.append(final_equa)

            print('toto',other_equas, variables, len(expressions_a))

            test = nsolve(other_equas,variables,estimates)
            summ_x = []
            summ_y = []
            for iii, val in enumerate(test):
                print('-->', dims_x[iii] * val, val)
                summ_x.append(dims_x[iii] * val)
                summ_y.append(dims_y[iii] * val)

            print('sumX', sum(summ_x) +  (len(images)-1)*space)  # --> si ça marche en fait
            print('sumY', summ_y)  # --> si ça marche en fait

            # --> cool ça marche
            # -->
            # is that all ok
            # for a COL
            # dims_x = []
            # dims_y = []
            # summ_x = []
            # summ_y = []
            # equations_single = []
            # expressions_a = []
            # expressions_b = []
            # equations = []
            # variables = []
            # estimates = []
            # for iii, img in enumerate(images):
            #     dims_x.append(img.width(False))
            #     dims_y.append(img.height(False))
            #     AR = img.width(False) / img.height(False)
            #
            #     variable = Symbol('a' + str(iii))
            #     variables.append(variable)
            #     estimates.append(1)
            #     equations_single.append(variable * (1. / AR) * img.width(False) + 0)
            #     expr = variable * (1. / AR) * img.width(False) + 0
            #     expressions_a.append(expr)
            #
            #     expr = variable * img.width(False)
            #     expressions_b.append(expr)
            #
            # # add an equa that is the sum of all
            # # final_equa = # sum of all and substraction
            # # final_equa = []
            #
            # print(estimates, equations_single, variables, expressions_b)
            # final_equa = None
            # for expr in expressions_b:
            #     if final_equa is None:
            #         final_equa = expr
            #     else:
            #         final_equa += expr
            #
            # desired_width = 512
            # final_equa += (len(images) - 1) * space - desired_width
            # print(final_equa)  # --> ok that works
            #
            # # do by pairs of equal
            # pairs_of_equations = None
            # other_equas = []
            # for iii in range(1, len(expressions_a)):
            #     # print(iii)
            #     # first - sec = 0
            #     equa = expressions_a[0] - expressions_a[iii]
            #     other_equas.append(equa)
            #
            # other_equas.append(final_equa)
            #
            # print('toto', other_equas, variables, len(expressions_a))
            #
            # test = nsolve(other_equas, variables, estimates)
            # summ_x = []
            # summ_y = []
            # for iii, val in enumerate(test):
            #     print('-->', dims_x[iii] * val, val)
            #     summ_x.append(dims_x[iii] * val)
            #     summ_y.append(dims_y[iii] * val)
            #
            # print('sumX', sum(summ_x) + (len(images) - 1) * space)  # --> si ça marche en fait
            # print('sumY', summ_y)  # --> si ça marche en fait






            # test autosolver for a row at least --> TODO


            # remettre mes nouvelles equas ici:



        # panel 1 ---> imageA | imageB --> incompx = 3 incompy = 0
        # panels2 imageC/imageD/imageE -- >  incompx = 0 incmp y =6
        # final fig = panel1 | panels2
        # all is ok for now
        #

        # do it for compressed chape then compute the 3 changes of AR that are needed to incorporate within the same fig the incompressible part of the image --> maybe three equations so I can solve them
        # --> TODO
        # worst case scenario --> back to brute force...

        # imageA -E --> 256w * 128h
        #
        # packed width --> 128 | 256|256|256 --> width = 128+256*3 -> 896
        # packed height --> 128 | 128|128|128 --> height = 128
        # if I wanna add to it spacer I need to do -->
        # if I wanna add a space between them --> I need add 3 pixels
        #  128 |3| 256|256|256 --> new width 896+3 --> 899--> if I want them to fit in 896 I need scale both stuff that are fully compressible by a value that is
        # I need * width and height by 896/897 =0.99888517279821627648 then it will fit --> easy because compressible and that would also change the height










        # test 3 width
        # x1 + x2 + x3 + sum incompressible = desired_width
        # rescaling
        # y1+y2 = 2*y3 --> mes equations
        # en fait y peut aussi etre part de mon equation




        # all seems ok --> can i now solve all my equations
        # from sympy import Symbol, nsolve
        # import mpmath
        # mpmath.mp.dps = 15
        # x1 = Symbol('x1')
        # x2 = Symbol('x2')
        # f1 = 3 * x1 ** 2 - 2 * x2 ** 2 - 1
        # f2 = x1 ** 2 - 2 * x1 + x2 ** 2 + 2 * x2 - 8
        # print(nsolve((f1, f2), (x1, x2), (-1, 1)))










        # 3 images 512*512
        # on veut à la fin 1536 --> scaling factor =1 pr chaque
        # finaliser aussi le code des clones
        # --> faire ça avec et ou sans divisions --> à faire donc
        #
        # put the n equations
        # a1*




        # 128 = 2*64
        # 128 = 1* 128
        # mais du coup le facteur de retaillage c'est pas ça
        # essayer


        # poser les eaqution pour que la hauteur soit la meme

        # how can I get my n equations
        # Solve the system of equations x0 + 2 * x1 = 1 and 3 * x0 + 5 * x1 = 2:
        # array([-1.,  1.]) # --> -1+2*1=1 3*-1+5*1=2 --> ça marche
        # that seems to work --> image same height -> 2 et 1
        # a0 = 2 et a1 = 1
        # facteur 2
        # a1x1+a2x2+a3x3 = 512-b1-b2-b3
        # une equation


        # desired width = 128
        # real example from columns --> with space 3
        # size of inner elm PyQt5.QtCore.QRectF(0.0, 0.0, 6.491758918437604, 49.188463457232196)
        # size of inner elm PyQt5.QtCore.QRectF(9.491758918437604, 0.0, 118.50824108156237, 42.42126355836911)
        # <class 'epyseg.figure.row.Row'>
        # PyQt5.QtCore.QRectF(0.0, 0.0, 127.99999999999997, 49.188463457232196)

        # with space 0
        #size of inner elm PyQt5.QtCore.QRectF(0.0, 0.0, 7.348408669130177, 45.49171603236653)
        # size of inner elm PyQt5.QtCore.QRectF(7.348408669130176, 0.0, 120.65159133086982, 45.49171603236654)
        # <class 'epyseg.figure.row.Row'>
        # PyQt5.QtCore.QRectF(0.0, 0.0, 128.0, 45.49171603236654)
        # if I put them to sameheight --> then with size 3 --> PyQt5.QtCore.QRectF(0.0, 0.0, 145.9477387942681, 49.188463457232196) --> not 128 anymore --> need resize it
        # need change the height from there one by one so that I reach 128 in width again


        # a = np.array([[1, 2], [3, 5]])
        # b = np.array([1, 2])
        # x = np.linalg.solve(a, b)
        # print(x)



        sys.exit(0)

    # find_scaling_to_fit()

    widths = [30, 15, 12, 3]
    final_width = 120
    space_between_images = 3  # 3px

    print(get_common_scaling_factor(widths, final_width, space_between_images))
    # so cool --> that will always work --> I love it
    # tester en grandeur reelle


    # ça marche --> can even preview a single image --> good
    from epyseg.draw.shapes.image2d import Image2D
    img = Image2D('/E/Sample_images/sample_images_EZF/counter/00.png')
    preview(img)




    # faire un autopanel maybe ???
    # or by default do a figure
    # try it all and see what I can do also add plots --> figures --> either from pandas or from a figure from matplotlib --> TODO



