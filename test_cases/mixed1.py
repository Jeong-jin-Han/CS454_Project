# test_ultimate_doomsday.py

# ==========================================
# CONSTANTS
# ==========================================
VOLTAGE_MIN = 2000
VOLTAGE_MAX = 2100
TARGET_FREQ = 0xCAFE
MAGIC_KEY = 777
HASH_TARGET_MOD = 128  # 확률 1/128의 해시 충돌을 찾아야 함

def _custom_hash(val: int) -> int:
    """재현 가능한 간단한 비트 믹싱 해시 함수"""
    val = (val ^ 61) ^ (val >> 16)
    val = (val + (val << 3)) 
    val = (val ^ (val >> 4)) 
    val = (val * 0x27d4eb2d) 
    val = (val ^ (val >> 15)) 
    return val & 0xFFFFFFFF

def ultimate_doomsday(point: tuple) -> int:
    """
    [Inputs]
    w: Voltage (Plateau)
    x: Frequency (Rugged)
    y: Security Key (Chained Needle)
    z: Nonce (Hash Mining)
    """
    w, x, y, z = point
    total_cost = 0
    
    # ---------------------------------------------------------
    # 1. Level 1: Voltage Check (Plateau Case)
    # ---------------------------------------------------------
    # 전압이 2000~2100 사이여야 함.
    # 하지만 센서는 100 단위로만 측정됨 (Quantization).
    # 즉, 1000~1099는 모두 같은 값, 1100~1199도 같은 값.
    # 계단형 평지(Staircase Plateau) 형성.
    
    # 거리 계산: 범위 밖이면 페널티, 범위 안이면 0
    if VOLTAGE_MIN <= w <= VOLTAGE_MAX:
        dist_w = 0
    else:
        # 중앙값(2050)과의 거리를 100으로 나눈 몫 -> 계단 형성
        raw_dist = abs(w - 2050)
        dist_w = (raw_dist // 100) * 1000  # 계단 높이 1000
    
    # 만약 전압 구간을 못 맞추면, 뒤의 단계들은 의미가 없음 (Blocking)
    # 하지만 탐색 유도를 위해 뒤의 비용을 약하게 더해줄 순 있으나,
    # 여기서는 '순차적 해결'을 강제하기 위해 w가 틀리면 매우 큰 값을 기본으로 깔아줌.
    if dist_w > 0:
        return dist_w + 500000 

    # ---------------------------------------------------------
    # 2. Level 2: Frequency Sync (Rugged Case)
    # ---------------------------------------------------------
    # w가 해결되었다고 가정하고 x를 평가.
    # x는 비트 연산 노이즈가 섞인 Rugged Landscape.
    
    # 목표: (x ^ (x >> 3)) 값이 TARGET_FREQ와 가까워야 함
    scrambled_x = (x ^ (x >> 3)) & 0xFFFF
    dist_x = abs(scrambled_x - TARGET_FREQ)
    
    # Rugged함에 소소한 노이즈 추가 (홀짝성에 따른 떨림)
    noise = (x & 0b111) * 10 
    dist_x += noise
    
    total_cost += dist_x

    # ---------------------------------------------------------
    # 3. Level 3: Security Key (Coupled Needle)
    # ---------------------------------------------------------
    # x까지 어느 정도 맞아야(거리 < 500) 힌트가 열림.
    # 목표: w + x + y == Magic Number 가 되어야 함 (변수 의존성)
    
    target_sum = MAGIC_KEY + w + x  # 동적 목표
    if dist_x < 500:
        dist_y = abs(y - target_sum)
        # Needle 특성: 정확하지 않으면 페널티가 급격함
        if dist_y > 0:
            dist_y = dist_y + 10000 # 바늘 구멍 밖은 고원
    else:
        dist_y = 50000 # 아직 x가 불안정해서 y를 평가할 수 없음

    total_cost += dist_y

    # ---------------------------------------------------------
    # 4. Level 4: Proof of Work (Hash Mining)
    # ---------------------------------------------------------
    # w, x, y가 모두 안정권(오차 0)에 들어왔을 때만 z(Nonce)를 채굴 가능.
    
    if dist_w == 0 and dist_x == 0 and dist_y == 0:
        # 모든 변수를 섞어서 해시 생성
        # z(Nonce)를 바꿔가며 해시값의 나머지가 0이 되는 경우를 찾아야 함
        combined_seed = w + x + y + z
        hash_val = _custom_hash(combined_seed)
        
        # Modulo 연산으로 Target Hit 여부 판별 (Mining)
        remainder = hash_val % HASH_TARGET_MOD
        
        if remainder == 0:
            return 0 # SUCCESS!
        else:
            # Hash Mining은 Gradient가 없음.
            # 나머지가 1이든 127이든 정답과의 거리는 알 수 없음.
            # 랜덤 서치(Basin Restart) 능력을 극한으로 시험.
            return 1000 + remainder # 약간의 힌트(remainder)를 주긴 하지만 사실상 노이즈
            
    else:
        total_cost += 100000 # 아직 채굴 단계 아님

    return float(total_cost)

FITNESS_FUNC = ultimate_doomsday
TEST_CONFIG = {
    'dim': 4,
    'start_point': (0, 0, 0, 0),
    'optimal_val': 0.0,
    'threshold': 1e-1,
    
    # [전략 포인트]
    # 1. Plateau(w)를 넘기 위해 iteration 필요
    # 2. Rugged(x)를 안정화하기 위해 Gradient Descent 필요
    # 3. Needle(y)을 찾기 위해 정밀 탐색 필요
    # 4. Hash(z)를 찾기 위해 아주 넓은 범위의 Random Restart(Basin Search) 필요
    
    'max_iterations': 200, 
    'basin_max_search': 500, # Hash Mining 확률(1/128)을 뚫으려면 최소 128 이상 탐색해야 함
    'max_steps_baseline': 20000
}