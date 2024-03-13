
def are_ratios_almost_equal(val1, val2, epsilon=0.0001):
    # checks if two values are almost equal (due to rounding errors for example)
    # epsilon is the tolerance
    return abs(val1 - val2) < epsilon

if __name__ == '__main__':
    print(are_ratios_almost_equal(1.6666666666666666,1.6666666666666667)) # True
    print(are_ratios_almost_equal(1.6667,1.6666)) # True
    print(are_ratios_almost_equal(1.667,1.666)) # False