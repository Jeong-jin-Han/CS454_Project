import rugged2

def test_verify_system_state_0():
    rugged2.verify_system_state(11, 11, 11)

def test_verify_system_state_1():
    rugged2.verify_system_state(2, 0, -3)

def test_fitness_coupled_2():
    rugged2.fitness_coupled(39, 40, 21)

def test_fitness_coupled_3():
    rugged2.fitness_coupled(6232, 5433, 3770)

