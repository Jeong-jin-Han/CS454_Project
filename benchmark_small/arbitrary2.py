def f(a: int, b: int):
    total = 0
    for i in range(a):
        total += i

    j = b
    while j > 0:
        total -= j
        j -= 1

    if total > 3:
        return total + 5
    elif total == 3:
        return total * 2
    else:
        return total - 7
