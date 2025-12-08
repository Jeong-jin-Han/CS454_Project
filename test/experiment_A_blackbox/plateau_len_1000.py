
def check_target(val, target):
    # 이 함수 내부는 AST 분석 대상에서 제외되거나, 
    # 호출부에서는 리턴값(Bool)만 보게 됩니다.
    return val == target

def plateau_1000(x: int):
    # Target: 0
    # Initialization: [500, 1000]
    
    target = 0
    
    # 함수 호출로 조건을 숨김 -> Gradient 소멸
    if check_target(x, target):
        return 0.0
    else:
        # 평지에서는 어디에 있든 똑같은 거리(1.0)로 인식됨
        return 1.0
