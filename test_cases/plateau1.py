# test_plateau_lock.py

# 4자리 비밀번호 (각 자리는 0~9 사이 정수라고 가정하지 않고 넓은 범위 허용)
SECRET_CODE = [10, 20, 30, 40]

def digital_lock(inputs: tuple) -> int:
    # inputs: 4개의 정수 (예: (10, 20, 99, 99))
    if len(inputs) != 4:
        return 10000
    
    unmatched_count = 0
    
    # 각 자릿수를 확인
    for i in range(4):
        # [Plateau 원인]
        # 값이 "얼마나 가까운가(Distance)"는 무시하고, "맞았나 틀렸나(Boolean)"만 봄.
        # 즉, 10을 맞춰야 하는데 11을 넣든 100을 넣든 똑같이 "틀림(1점)" 처리.
        # 따라서 정답을 제외한 모든 공간이 평평한 고원(Plateau)임.
        if inputs[i] != SECRET_CODE[i]:
            unmatched_count += 1
            
    # 다 맞으면 0, 하나 틀릴 때마다 1000점씩 페널티
    # 예: 3개 틀리면 3000점. 
    # 3000점에서 2000점으로 내려가려면 우연히 숫자 하나를 정확히 맞춰야 함.
    return unmatched_count * 1000

FITNESS_FUNC = digital_lock
TEST_CONFIG = {
    'dim': 4,
    'start_point': (0, 0, 0, 0),
    'optimal_val': 0,
    'threshold': 0,
    'max_iterations': 100,
    # 정답을 찾기 위해 매우 넓은 범위를 훑어야 함
    # 압축 알고리즘이 "주변을 아무리 봐도 점수가 안 변하네?" 하고 크게 점프해야 함
    'basin_max_search': 500, 
    'max_steps_baseline': 10000
}