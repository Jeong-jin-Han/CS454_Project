import rugged2

def test_verify_system_state_0():
    rugged2.verify_system_state(32, 32, 32)

def test_verify_system_state_1():
    rugged2.verify_system_state(2, 0, 2)

def test_fitness_coupled_2():
    rugged2.fitness_coupled(-16, -14, 146)

def test_fitness_coupled_3():
    rugged2.fitness_coupled(215, -8, 9879)

