import rugged2

def test_verify_system_state_0():
    rugged2.verify_system_state(11, 11, 11)

def test_verify_system_state_1():
    rugged2.verify_system_state(-2, 3, 5)

def test_fitness_coupled_2():
    rugged2.fitness_coupled(16, 16, 128)

def test_fitness_coupled_3():
    rugged2.fitness_coupled(4444, 6833, 7985)

