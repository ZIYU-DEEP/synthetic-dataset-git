import numpy as np
from math import sin, cos


def gen_ball(r, mu):
    """
    Generate a set of data points surrounding a point like a ball.
    This function will later be used in main to generate a set of means
    for abnormal test.

    Inputs:
        r: (float) distance between the trained normal and the trained
           abnormal; used as the radius here
        mu: (np.array) a 3d array specifying the mu for trained normal
            or the trained abnormal data

    Returns:
        result: (list) a list a 3d arrays indicating the mean for abnormal
                data to test
    """
    thetas = range(0, 360, 60)
    phis = range(0, 360, 60)
    pairs = [(theta, phi) for theta in thetas for phi in phis]

    result = []
    for pair in pairs:
        theta, phi = pair
        cord = [sin(theta) * cos(phi) * r + mu[0],
                sin(theta) * sin(phi) * r + mu[1],
                cos(theta) * r + mu[2]]
        if cord in result:
            continue
        result.append(cord)

    return result
