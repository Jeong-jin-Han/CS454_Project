def collatz_steps(n: int):
    if n <= 0:
        return -1

    steps = 0
    x = n
    while x > 1:
        if x % 2 == 0:
            x //= 2
        else:
            x = x * 3 + 1
        steps += 1
        if steps > 20:
            break

    if steps > 15:
        # print("Too many steps!")
        pass
    elif steps == 10:
        # print("Exactly 10 steps!")
        pass
    else:
        # print("Perfect :)")
        pass
