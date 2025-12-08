def derivative_quadratic(a):
    h = 1
    left = (a - h) * (a - h)
    right = (a + h) * (a + h)
    slope = (right - left) // (2 * h)

    if slope % 2 == 0:
        return ("even", slope)
    else:
        return ("odd", slope)
