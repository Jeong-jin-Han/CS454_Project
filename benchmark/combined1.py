def combined1(x: int, y: int, z: int) -> int:
    # Needle in the haystack
    if x == 7 and y == 10102 and z == 50:
        v = 100
    else:
        v = x * y - z * z

    # Rugged
    if (v % 6 == 0):
        v += 15
    elif (v % 6 == 1):
        v -= 10
    elif (v % 6 == 2):
        v += 3
    elif (v % 6 == 3):
        v -= 7
    elif (v % 6 == 4):
        v += 1
    else:
        v -= 2

    # Plateau
    if v < 50:
        v = 42
    else:
        v += 5

    return v
