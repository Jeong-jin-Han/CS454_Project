def rugged1(x: int, y: int, z: int) -> int:
    v = x * 3 - y * 2 + z

    for i in range(1, 5):
        for j in range(1, 4):
            tmp = v + i * j
            if (tmp + x * j) % 7 == 0:
                v += i * j
            elif (tmp + y * i) % 5 == 1:
                v -= 2 * i
            else:
                if (z * i + j) % 3 == 0:
                    v += 3 * j
                elif (x + y + z + i + j) % 4 == 1:
                    v -= 4
                else:
                    v += (i - j)

    if x % 2 == 0:
        if y % 3 == 0:
            v += 7
        else:
            v -= 6
    else:
        if z % 5 == 2:
            v += 5
        else:
            v -= 3

    return v
