TARGET_SUM = 100
TARGET_PROD = 32768

def verify_system_state(x: int, y: int, z: int) -> bool:
    if (x + y + z == TARGET_SUM) and (x * y * z == TARGET_PROD):
        return True
    return False

def fitness_coupled(x: int, y: int, z: int) -> int:
    current_sum = x + y + z
    current_prod = x * y * z

    dist_sum = abs(current_sum - TARGET_SUM)
    dist_prod = abs(current_prod - TARGET_PROD)

    if dist_sum == 0 and dist_prod == 0:
        return 0
    else:
        return 10000 + dist_sum + dist_prod
