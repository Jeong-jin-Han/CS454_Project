import time
"mult branch 동시 평가용도"

def busy_wait(duration_sec):
    end = time.time() + duration_sec
    while time.time() < end:
        pass  # 아무것도 안 하지만 CPU는 계속 사용됨

def parallel(n: int):
    busy_wait(10)

    if n == 5:
        pass
    elif n == 10:
        pass
    elif n == 15:
        pass
    elif n == 20:
        pass
    elif n == 25:
        pass
    elif n == 30:
        pass
    elif n == 35:
        pass
    elif n == 40:
        pass 
    elif n == 45:
        pass
    elif n == 50:
        pass
    elif n == 55:
        pass
    elif n == 60:
        pass
    elif n == 65:
        pass
    else:
        pass