SECRET_CODE = [10, 20, 30, 40]

def unlock_door(d0: int, d1: int, d2: int, d3: int) -> bool:
    if d0 == SECRET_CODE[0]:
        if d1 == SECRET_CODE[1]:
            if d2 == SECRET_CODE[2]:
                if d3 == SECRET_CODE[3]:
                    return True
    return False

def fitness_digital_lock(d0: int, d1: int, d2: int, d3: int) -> int:

    cost = 0

    if d0 != SECRET_CODE[0]:
        cost += 1000
    if d1 != SECRET_CODE[1]:
        cost += 1000
    if d2 != SECRET_CODE[2]:
        cost += 1000
    if d3 != SECRET_CODE[3]:
        cost += 1000

    return cost
