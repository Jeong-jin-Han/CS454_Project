def triangle(a: int, b: int, c: int):
    if a <= 0 or b <= 0 or c <= 0:
        print("Invalid input")
        return

    if a + b <= c or a + c <= b or b + c <= a:
        print("Not a triangle")
        return

    # actual triangle types with extra branches
    if a == b and b == c:
        print("Equilateral triangle")
    elif a == b or b == c or a == c:
        if (a + b + c) % 2 == 0:
            print("Even isosceles triangle")
        else:
            print("Odd isosceles triangle")
    else:
        if a > 3 and b > 3 and c > 3:
            print("Big scalene triangle")
        else:
            print("Small scalene triangle")
