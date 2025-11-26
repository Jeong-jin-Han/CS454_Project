def prime_check(n: int):
    if n <= 1:
        pass
        # return -1

    is_prime = 1
    i = 2
    while i * i <= n:
        if n % i == 0:
            is_prime = 0
            break
        i += 1

    # post-processing to create tricky branches
    if is_prime == 1:
        if n % 4 == 1:
            pass
            # return 10
        else:
            pass
            # return 11
    else:
        if n % 3 == 0:
            pass
            # return 20
        else:
            # return 21
            pass
