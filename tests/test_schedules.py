from marl_lbf.schedules import EpsilonSchedule


def test_epsilon_schedule_stops_at_floor():
    schedule = EpsilonSchedule(start=0.12, minimum=0.1, decay=0.02)

    assert schedule.step() == 0.1
    assert schedule.step() == 0.1
