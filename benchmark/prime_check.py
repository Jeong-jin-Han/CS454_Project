def prime_check(n: int) -> int:
    if n <= 1:
        return -1

    is_prime = 1
    i = 2
    while i * i <= n:
        if n % i == 0:
            is_prime = 0
            break
        i += 1

    if is_prime == 1:
        if n % 4 == 1:
            return 10
        else:
            return 11
    else:
        if n % 3 == 0:
            return 20
        else:
            return 21
