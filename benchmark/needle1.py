def needle1(a: int, b: int, c: int) -> int:
    counter = 0
    while True:
        if ((a * a + b * 3 - c * 7) % 101 == 42 and 
            (b * b - a * c) % 50 == 17 and
            (c * 5 + a - b) % 13 == 9):
            result = 999  # Needle
            break
        
        # 변수 변화: 서로 얽히고 mod로 좁힌 범위
        a = (a + b * 2 + counter * 3) % 37
        b = (b + c * 3 + counter * 5) % 53
        c = (c + a * 4 + counter * 7) % 29
        
        counter += 1
        if counter > 2000:  # 안전 장치
            result = -1
            break

    return result