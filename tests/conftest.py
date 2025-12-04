"""
Pytest configuration additions.

Fixtures defined here are available to any test in Crow.
"""

import os
import sys

import pytest

# Ensure we can import from the main crow/ package


def pytest_addoption(parser):
    """Add handling of specific options for the pytest test runner.

    --runslow: used to run tests marked as slow, which are otherwise not run.
    --integration: used to run only integration tests, which are otherwise not run.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_configure(config):
    """Add new markers that can be set on pytest tests.

    Use the marker `slow` for any test that takes more than a second to run.
    Tests marked as `slow` are not run unless the user requests them by specifying
    the --runslow flag to the pytest program.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Apply our special markers and option handling for pytest."""

    if not config.getoption("--integration"):
        _skip_tests(items, "integration", "need --integration option to run")

    if not config.getoption("--runslow"):
        _skip_tests(items, "slow", "need --runslow option to run")


def _skip_tests(items, keyword, reason):
    """Helper method to skip tests based on a marker name."""

    tests_to_skip = pytest.mark.skip(reason=reason)
    for item in items:
        if keyword in item.keywords:
            item.add_marker(tests_to_skip)


# Fixtures
