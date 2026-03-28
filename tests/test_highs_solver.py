"""Tests for HiGHSSolver implementation."""
import pytest


def test_highs_solver_init():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    assert solver is not None
    assert solver.INF > 1e20


def test_add_binary_var():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    v = solver.add_var(name="x", type="b")
    assert v is not None


def test_add_integer_var():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    v = solver.add_var(name="y", type="i", lb=0, ub=10)
    assert v is not None


def test_add_continuous_var():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    v = solver.add_var(name="z", type="c", lb=0.0, ub=100.0)
    assert v is not None


def test_add_var_invalid_type():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    with pytest.raises(ValueError):
        solver.add_var(name="bad", type="x")
