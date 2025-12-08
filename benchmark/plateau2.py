def check_schedule(timestamp: int) -> bool:
    if (timestamp // 86400) % 7 == 6:

        if (timestamp % 86400) // 3600 == 14:
            return True

    return False

def fitness_calendar(timestamp: int) -> int:
    fitness = 0

    days_passed = timestamp // 86400
    weekday = days_passed % 7

    if weekday != 6:
        fitness += 10000 + abs(6 - weekday) * 100
    else:
        seconds_in_day = timestamp % 86400
        hour = seconds_in_day // 3600

        if hour != 14:
            fitness += abs(14 - hour) * 100

    return fitness
