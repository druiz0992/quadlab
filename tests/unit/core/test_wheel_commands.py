import pytest

from robotlab.core.wheeled import WheelCommands


def test_zeros_and_tuple_conversion():
    commands = WheelCommands([1, 2], [0, 0], [0, 0])
    assert commands.drive_torque_nm == (1.0, 2.0)
    assert WheelCommands.zeros(3) == WheelCommands((0.0,) * 3, (0.0,) * 3, (0.0,) * 3)


def test_validate_reports_the_problem():
    flags = (True, False)
    WheelCommands((1, 0), (0, 0), (0, 0)).validate(flags, flags, flags)
    with pytest.raises(ValueError, match=r"drive_torque_nm\[1\]"):
        WheelCommands((1, 1), (0, 0), (0, 0)).validate(flags, flags, flags)
