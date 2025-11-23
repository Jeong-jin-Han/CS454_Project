def derivative_quadratic(a):
    # approximate derivative of f(x)=x^2 using finite difference
    h = 1
    left = (a - h) * (a - h)
    right = (a + h) * (a + h)
    slope = (right - left) // (2 * h)

    if slope % 2 == 0:
        return ("even", slope)
    else:
        return ("odd", slope)
