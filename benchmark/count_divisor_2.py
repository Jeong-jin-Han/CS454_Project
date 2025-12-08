def count_divisors_of_gcd(a: int, b: int) -> int:
    if a * b == 0:
        return 0

    a = a if a > 0 else -a
    b = b if b > 0 else -b

    while b != 0:
        temp = a
        a = b
        b = temp % b

    gcd = a
    cnt = 0
    i = 1
    while i * i <= gcd:
        if gcd % i == 0:
            cnt += 2 if i * i != gcd else 1
        i += 1

    if cnt > 30:
        print("More than thirty!")
    elif cnt == 10:
        print("Exactly 10 divisors!")

    return cnt
