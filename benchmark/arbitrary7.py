def f(a: int, b: int, c: int) -> isinstance:
    val = 0
    for i in range(a):
        val += i * b

    j = c
    while j < b:
        val -= j
        j += 1

    if val > 10:
        if a == 3:
            val += 5
        else:
            val -= 4
    else:
        if b == 1:
            val += 7
        else:
            val -= 8

    return val
