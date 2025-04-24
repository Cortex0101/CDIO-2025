import pytest
from Camera.Server import choose_next_ball

# Test function for choosing the next ball
def test_choose_next_ball():
    # Setup for the test
    balls = [(0, 150), (150, 150)]
    current_position = (0, 0)

    # Call the function to choose the next ball
    next_ball = choose_next_ball(balls, current_position)

    # Assert that the closest ball is chosen
    assert next_ball == (0, 150), f"Expected (0, 150), but got {next_ball}"