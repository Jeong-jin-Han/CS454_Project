import plateau3

def test_unlock_door_0():
    plateau3.unlock_door(10, -7, 3, 3)

def test_unlock_door_1():
    plateau3.unlock_door(0, -7, 3, 3)

def test_unlock_door_2():
    plateau3.unlock_door(10, 20, 3, 3)

def test_unlock_door_3():
    plateau3.unlock_door(10, -7, 3, 3)

def test_unlock_door_4():
    plateau3.unlock_door(10, 20, 30, 3)

def test_unlock_door_5():
    plateau3.unlock_door(10, 20, 3, 3)

def test_unlock_door_6():
    plateau3.unlock_door(10, 20, 30, 40)

def test_unlock_door_7():
    plateau3.unlock_door(10, 20, 30, 3)

def test_fitness_digital_lock_8():
    plateau3.fitness_digital_lock(0, 94, 0, 422)

def test_fitness_digital_lock_9():
    plateau3.fitness_digital_lock(10, 94, 0, 422)

def test_fitness_digital_lock_10():
    plateau3.fitness_digital_lock(0, 94, 0, 422)

def test_fitness_digital_lock_11():
    plateau3.fitness_digital_lock(0, 20, 0, 422)

def test_fitness_digital_lock_12():
    plateau3.fitness_digital_lock(0, 94, 0, 422)

def test_fitness_digital_lock_13():
    plateau3.fitness_digital_lock(0, 94, 30, 422)

def test_fitness_digital_lock_14():
    plateau3.fitness_digital_lock(0, 94, 0, 422)

def test_fitness_digital_lock_15():
    plateau3.fitness_digital_lock(0, 94, 0, 40)

