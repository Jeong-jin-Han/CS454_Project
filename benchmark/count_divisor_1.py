def count_divisors(n: int) -> int:
    if n == 0:
        return 0  
    elif n < 0:
        n = -n

    cnt = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            cnt += 2 if i * i != n else 1
        i += 1

    if cnt > 100:
        print("More than hundred!")
    elif cnt == 19:
        print("Exactly 19 divisors!")
    
    return cnt

