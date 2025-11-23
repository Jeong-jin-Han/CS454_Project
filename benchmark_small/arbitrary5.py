def f(x: int, y: int) -> int:
    score = 0

    if x > 1 and not (y < 2):
        score += 3
    else:
        score -= 2

    if (x == 2 and y == 2) or (x == 3 and y != 1):
        score += 10
    else:
        score -= 5

    if not (x + y > 3):
        score += 7
    else:
        score -= 1

    return score
