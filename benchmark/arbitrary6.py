def f(n: int, k: int) -> int:
    total = 0
    x = n

    while x > 0:
        total += (x // k)
        if x == k:
            total += 5
        x -= 1

    if total < 3:
        return total * 2
    else:
        return total - 3
