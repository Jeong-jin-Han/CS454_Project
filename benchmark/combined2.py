VOLTAGE_MIN = 2000
VOLTAGE_MAX = 2100
TARGET_FREQ = 0xCAFE
MAGIC_KEY = 777
HASH_MOD = 128

def _hash(val: int) -> int:
    val = (val ^ 61) ^ (val >> 16)
    val = (val + (val << 3))
    val = (val ^ (val >> 4))
    val = (val * 0x27d4eb2d)
    val = (val ^ (val >> 15))
    return val & 0xFFFFFFFF

def disarm_doomsday(w: int, x: int, y: int, z: int) -> bool:
    if 20 <= (w // 100) <= 21:

        scrambled = (x ^ (x >> 3)) & 0xFFFF
        if scrambled == TARGET_FREQ:

            if w + x + y == MAGIC_KEY:

                h_val = _hash(w + x + y + z)
                if h_val % HASH_MOD == 0:
                    return True

    return False

def fitness_doomsday(w: int, x: int, y: int, z: int) -> float:
    w_val = w // 100
    if not (20 <= w_val <= 21):
        dist = min(abs(w_val - 20), abs(w_val - 21))
        return 400000 + dist * 1000

    scrambled = (x ^ (x >> 3)) & 0xFFFF
    if scrambled != TARGET_FREQ:
        dist = abs(scrambled - TARGET_FREQ)
        noise = (x & 0b111) * 10
        return 300000 + dist + noise

    target_y = MAGIC_KEY - (w + x)
    if y != target_y:
        return 200000 + abs(y - target_y)

    h_val = _hash(w + x + y + z)
    remainder = h_val % HASH_MOD
    if remainder != 0:
        return 100000 + remainder

    return 0.0
