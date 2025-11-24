# test_ultimate_doomsday.py

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

# 1. [Target Code] 둠스데이 프로토콜
def disarm_doomsday(point: tuple) -> bool:
    w, x, y, z = point
    
    # Level 1: Voltage (Range Check via Quantization)
    # 계측기가 100단위로만 동작한다고 가정 (Integer Division)
    if 20 <= (w // 100) <= 21:
        
        # Level 2: Frequency (Bitwise Logic)
        scrambled = (x ^ (x >> 3)) & 0xFFFF
        if scrambled == TARGET_FREQ:
            
            # Level 3: Security Key (Coupled Logic)
            # w, x값이 확정된 상태에서 y가 맞아야 함
            if w + x + y == MAGIC_KEY:
                
                # Level 4: Proof of Work (Hash)
                h_val = _hash(w + x + y + z)
                if h_val % HASH_MOD == 0:
                    return True # Disarmed!
                    
    return False

# 2. [Fitness Function] Approach Level + Branch Distance
def fitness_doomsday(point: tuple) -> float:
    w, x, y, z = point
    
    # --- Level 1 Check ---
    # w // 100 이 20~21 사이여야 함 (2000~2199)
    w_val = w // 100
    if not (20 <= w_val <= 21):
        # Approach Level 4 (가장 바깥)
        # Distance: 범위 밖 거리
        dist = min(abs(w_val - 20), abs(w_val - 21))
        return 400000 + dist * 1000 # Huge Penalty
        
    # --- Level 2 Check ---
    scrambled = (x ^ (x >> 3)) & 0xFFFF
    if scrambled != TARGET_FREQ:
        # Approach Level 3
        # Rugged distance added
        dist = abs(scrambled - TARGET_FREQ)
        # Add some noise to simulate ruggedness
        noise = (x & 0b111) * 10
        return 300000 + dist + noise
        
    # --- Level 3 Check ---
    target_y = MAGIC_KEY - (w + x)
    if y != target_y:
        # Approach Level 2
        # Coupled Needle distance
        return 200000 + abs(y - target_y)
        
    # --- Level 4 Check ---
    h_val = _hash(w + x + y + z)
    remainder = h_val % HASH_MOD
    if remainder != 0:
        # Approach Level 1
        # Hash Mining (No Gradient)
        return 100000 + remainder # Remainder is just random noise, effectively
        
    # Success
    return 0.0

FITNESS_FUNC = fitness_doomsday
TEST_CONFIG = {
    'dim': 4,
    'start_point': (0, 0, 0, 0),
    'optimal_val': 0.0,
    'threshold': 1e-1,
    'max_iterations': 200, 
    'basin_max_search': 500,
    'max_steps_baseline': 20000
}