
def calendar_div_1000(timestamp: int):
    # Divisor (Plateau Length): 1000
    # Target: timestamp // 1000 == 1
    # Range: 0 ~ 1000 (Cost = 1.0) -> 1000 (Cost = 0.0)
    
    # 1. Target Check (Gradient Killing)
    # 문자열로 변환하여 나눗셈의 '나머지(Gradient)' 정보를 완벽히 차단합니다.
    # 오직 몫이 1이 되었느냐(성공) 아니냐(실패)만 알 수 있습니다.
    quotient = timestamp // 1000
    
    if quotient == 1:
        return 0.0
    
    # 2. Plateau Area
    # 몫이 1이 아니면(아직 divisor에 도달 못했거나 넘어갔으면)
    # 거대한 평지(Cost 1.0)를 형성합니다.
    return 1.0
