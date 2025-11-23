# needle_case.py

SECRET1 = 123456
SECRET = 103456

def needle(x: int) -> int:
    """
    목표 브랜치:
        if x == SECRET and (x % 97) == 13:
            # TARGET
    쪽이라고 가정하고 fitness를 distance 기반으로 만들 수 있습니다.
    """
    if x == SECRET and (x % 97) == (SECRET % 97):
        # TARGET BRANCH
        flag = 0
    else:
        # 대부분의 입력이 여기로 옴
        d = abs(x - SECRET)

        # 멀리서는 전부 같은 값으로 saturate → landscape가 거의 flat
        if d > 5000:
            d = 5000

        # 해시 한 번 섞어서 “거리 정보”도 좀 가려버리기
        h = x
        h ^= (h << 13)
        h ^= (h >> 17)
        h ^= (h << 5)
        noise = abs(h % 7)   # 작지만 랜덤 같은 노이즈

        flag = d + noise

    return flag
