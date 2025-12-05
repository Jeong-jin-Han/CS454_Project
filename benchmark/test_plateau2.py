import plateau2

def test_check_schedule_0():
    plateau2.check_schedule(-10)

def test_check_schedule_1():
    plateau2.check_schedule(86365)

def test_check_schedule_2():
    plateau2.check_schedule(-32401)

def test_check_schedule_3():
    plateau2.check_schedule(-10)

def test_fitness_calendar_4():
    plateau2.fitness_calendar(0)

def test_fitness_calendar_5():
    plateau2.fitness_calendar(-10)

def test_fitness_calendar_6():
    plateau2.fitness_calendar(-10)

def test_fitness_calendar_7():
    plateau2.fitness_calendar(-32401)

