from ctypes import pointer

import numpy as np

from epyseg.ta.measurements.TAmeasures import distance_between_points
from epyseg.ta.utils.rps_tools import get_feret_from_points, extreme_points




if __name__ == '__main__':

    test = np.asarray([1,10,10,1])
    print('test where', np.where(test == test.max())) # see why with cdist I can't find the two ???

    # Bug somehow (but not really because there are several solutions, the only thing is that the solution may not be optimal for bonds) --> it should have returned the other extreme

    coords = ((612, 457), (612, 458), (612, 459), (612, 460), (612, 461), (612, 462), (612, 463), (612, 464), (613, 441), (613, 442), (613, 443), (613, 444), (613, 445), (613, 446), (613, 447), (613, 448), (613, 449), (613, 450), (613, 451), (613, 452), (613, 453), (613, 454), (613, 455), (613, 456), (614, 426), (614, 427), (614, 428), (614, 429), (614, 430), (614, 431), (614, 432), (614, 433), (614, 434), (614, 435), (614, 436), (614, 437), (614, 438), (614, 439), (614, 440), (615, 411), (615, 412), (615, 413), (615, 414), (615, 415), (615, 416), (615, 417), (615, 418), (615, 419), (615, 420), (615, 421), (615, 422), (615, 423), (615, 424), (615, 425), (616, 395), (616, 396), (616, 397), (616, 398), (616, 399), (616, 400), (616, 401), (616, 402), (616, 403), (616, 404), (616, 405), (616, 406), (616, 407), (616, 408), (616, 409), (616, 410), (617, 380), (617, 381), (617, 382), (617, 383), (617, 384), (617, 385), (617, 386), (617, 387), (617, 388), (617, 389), (617, 390), (617, 391), (617, 392), (617, 393), (617, 394), (617, 499), (617, 500), (617, 501), (617, 502), (617, 503), (617, 504), (617, 505), (617, 506), (617, 507), (617, 508), (617, 509), (617, 510), (617, 511), (617, 512), (617, 513), (617, 514), (617, 515), (617, 516), (618, 372), (618, 373), (618, 374), (618, 375), (618, 376), (618, 377), (618, 378), (618, 379), (618, 473), (618, 474), (618, 475), (618, 476), (618, 477), (618, 478), (618, 479), (618, 480), (618, 481), (618, 482), (618, 483), (618, 484), (618, 485), (618, 486), (618, 487), (618, 488), (618, 489), (618, 490), (618, 491), (618, 492), (618, 493), (618, 494), (618, 495), (618, 496), (618, 497), (618, 498), (618, 517), (618, 518), (618, 519), (618, 520), (618, 521), (618, 522), (618, 523), (618, 524), (618, 525), (618, 526), (619, 374), (619, 375), (619, 447), (619, 448), (619, 449), (619, 450), (619, 451), (619, 452), (619, 453), (619, 454), (619, 455), (619, 456), (619, 457), (619, 458), (619, 459), (619, 460), (619, 461), (619, 462), (619, 463), (619, 464), (619, 465), (619, 466), (619, 467), (619, 468), (619, 469), (619, 470), (619, 471), (619, 472), (619, 527), (619, 528), (619, 529), (619, 530), (619, 531), (619, 532), (619, 533), (619, 534), (619, 535), (619, 536), (620, 376), (620, 377), (620, 378), (620, 421), (620, 422), (620, 423), (620, 424), (620, 425), (620, 426), (620, 427), (620, 428), (620, 429), (620, 430), (620, 431), (620, 432), (620, 433), (620, 434), (620, 435), (620, 436), (620, 437), (620, 438), (620, 439), (620, 440), (620, 441), (620, 442), (620, 443), (620, 444), (620, 445), (620, 446), (620, 537), (620, 538), (620, 539), (620, 540), (620, 541), (620, 542), (620, 543), (620, 544), (620, 545), (620, 546), (621, 379), (621, 380), (621, 395), (621, 396), (621, 397), (621, 398), (621, 399), (621, 400), (621, 401), (621, 402), (621, 403), (621, 404), (621, 405), (621, 406), (621, 407), (621, 408), (621, 409), (621, 410), (621, 411), (621, 412), (621, 413), (621, 414), (621, 415), (621, 416), (621, 417), (621, 418), (621, 419), (621, 420), (621, 547), (621, 548), (621, 549), (621, 550), (621, 551), (621, 552), (621, 553), (621, 554), (621, 555), (621, 556), (622, 381), (622, 382), (622, 383), (622, 384), (622, 385), (622, 386), (622, 387), (622, 388), (622, 389), (622, 390), (622, 391), (622, 392), (622, 393), (622, 394), (622, 557), (622, 558), (622, 559), (622, 560), (622, 561), (622, 562), (622, 563), (622, 564), (622, 565), (622, 566), (623, 567), (623, 568), (623, 569), (623, 570), (623, 571), (623, 572), (623, 573), (623, 574), (623, 575), (623, 576), (624, 577), (624, 578), (624, 579), (624, 580), (624, 581), (624, 582), (624, 583), (624, 584), (624, 585), (624, 586), (625, 587), (625, 588), (625, 589), (625, 590), (625, 591), (625, 592), (625, 593), (625, 594), (625, 595), (625, 596), (626, 597), (626, 598), (626, 599), (626, 600), (626, 601), (626, 602), (626, 603), (626, 604), (626, 605), (626, 606), (627, 607), (627, 608), (627, 609), (627, 610), (627, 611), (627, 612), (627, 613), (627, 614), (627, 615), (627, 616), (628, 617), (628, 618), (628, 619), (628, 620), (628, 621), (628, 622), (629, 623), (629, 624), (629, 625), (630, 626), (630, 627), (630, 628), (631, 629), (631, 630), (631, 631), (632, 632), (632, 633), (632, 634), (633, 635), (633, 636), (633, 637), (634, 638), (634, 639), (634, 640), (635, 641), (635, 642), (635, 643), (636, 644), (636, 645), (636, 646), (637, 647), (637, 648), (637, 649), (638, 650), (638, 651), (638, 652), (639, 653), (639, 654), (639, 655), (640, 656), (640, 657), (640, 658), (641, 659), (641, 660), (641, 661), (642, 662), (642, 663), (643, 664), (643, 665), (643, 666), (644, 667), (644, 668), (644, 669), (645, 670), (645, 671), (645, 672), (645, 819), (645, 820), (645, 821), (645, 822), (645, 823), (645, 824), (645, 825), (645, 826), (645, 827), (645, 828), (645, 829), (645, 830), (645, 831), (645, 832), (645, 833), (645, 834), (645, 835), (645, 836), (645, 837), (645, 838), (645, 839), (645, 840), (645, 841), (645, 842), (645, 843), (645, 844), (645, 845), (645, 846), (645, 847), (645, 848), (645, 849), (645, 850), (645, 851), (645, 852), (645, 853), (645, 854), (645, 855), (645, 856), (645, 857), (645, 858), (645, 859), (645, 860), (645, 861), (645, 862), (645, 863), (645, 864), (645, 865), (645, 866), (645, 867), (645, 868), (645, 869), (645, 870), (645, 871), (645, 872), (645, 873), (645, 874), (645, 875), (645, 876), (645, 877), (645, 878), (645, 879), (645, 880), (645, 881), (646, 673), (646, 674), (646, 675), (646, 750), (646, 751), (646, 752), (646, 753), (646, 754), (646, 755), (646, 756), (646, 757), (646, 758), (646, 759), (646, 760), (646, 761), (646, 762), (646, 763), (646, 764), (646, 765), (646, 766), (646, 767), (646, 768), (646, 769), (646, 770), (646, 771), (646, 772), (646, 773), (646, 774), (646, 775), (646, 776), (646, 777), (646, 778), (646, 779), (646, 780), (646, 781), (646, 782), (646, 783), (646, 784), (646, 785), (646, 786), (646, 787), (646, 788), (646, 789), (646, 790), (646, 791), (646, 792), (646, 793), (646, 794), (646, 795), (646, 796), (646, 797), (646, 798), (646, 799), (646, 800), (646, 801), (646, 802), (646, 803), (646, 804), (646, 805), (646, 806), (646, 807), (646, 808), (646, 809), (646, 810), (646, 811), (646, 812), (646, 813), (646, 814), (646, 815), (646, 816), (646, 817), (646, 818), (647, 676), (647, 677), (647, 678), (647, 745), (647, 746), (647, 747), (647, 748), (647, 749), (648, 679), (648, 680), (648, 681), (648, 740), (648, 741), (648, 742), (648, 743), (648, 744), (649, 682), (649, 683), (649, 684), (649, 736), (649, 737), (649, 738), (649, 739), (650, 685), (650, 686), (650, 687), (650, 731), (650, 732), (650, 733), (650, 734), (650, 735), (651, 688), (651, 689), (651, 690), (651, 726), (651, 727), (651, 728), (651, 729), (651, 730), (652, 691), (652, 692), (652, 693), (652, 721), (652, 722), (652, 723), (652, 724), (652, 725), (653, 694), (653, 695), (653, 696), (653, 716), (653, 717), (653, 718), (653, 719), (653, 720), (654, 697), (654, 698), (654, 699), (654, 712), (654, 713), (654, 714), (654, 715), (655, 700), (655, 701), (655, 702), (655, 707), (655, 708), (655, 709), (655, 710), (655, 711), (656, 703), (656, 704), (656, 705), (656, 706))
    point1, point2= get_feret_from_points(coords) # -->((618, 372), (645, 881)) but this is incorrect !!!!
    print(point1, point2)
    found_dist = distance_between_points(point1, point2)
    print(distance_between_points(point1, point2))
    print(distance_between_points((618,372), (645,881))) #these are the real feret points --> in fact the distance is 100% equal but the second solution would have been better --> how can I do that
    # indeed the pb is that the distance is equal --> so if there are many equal distances I shall maybe take the extreme most points
    coords = extreme_points(np.asarray(coords), return_array=True )
    print(coords)
    # below does not work --> I just want the other pair --> take it and check if pixels exist and if one is more extreme --> take it
    for coord1 in coords:
        for coord2 in coords:
            if coord1 == coord2:
                continue
            if distance_between_points(coord1, coord2) >= found_dist:
                # if tuple(coord1) != tuple(point1):
                print('as_good or better', coord1, coord2)
                # si un des deux points differe -> le prendre
                print((tuple(coord1)!=tuple(point1) or tuple(coord1)!=tuple(point2)) , ((tuple(coord2)!=tuple(point1) or tuple(coord2)!=tuple(point2))))
                print(tuple(coord1)!=tuple(point1) , tuple(coord1)!=tuple(point2) , tuple(coord2)!=tuple(point1) , tuple(coord2)!=tuple(point2))
                if (tuple(coord1)!=tuple(point1) or tuple(coord1)!=tuple(point2)):
                    if tuple(coord2) == tuple(point1) or tuple(coord2) == tuple(point2):
                        print('other possibility',coord1, coord2)
                if (tuple(coord2)!=tuple(point1) or tuple(coord2)!=tuple(point2)):
                    if tuple(coord1)==tuple(point1) or tuple(coord1)==tuple(point2):
                        print('other possibility',coord1, coord2)

    # TODO I should return the other pair

 # or ((tuple(coord2)!=tuple(point1) and tuple(coord2)!=tuple(point2)))




