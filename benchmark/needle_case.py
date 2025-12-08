SECRET1 = 123456
SECRET = 103456

def needle(x: int) -> int:
    if x == SECRET and (x % 97) == (SECRET % 97):
        flag = 0
    else:
        d = abs(x - SECRET)

        if d > 5000:
            d = 5000

        h = x
        h ^= (h << 13)
        h ^= (h >> 17)
        h ^= (h << 5)
        noise = abs(h % 7)

        flag = d + noise

    return flag
