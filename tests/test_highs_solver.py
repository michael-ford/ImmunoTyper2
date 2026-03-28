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


def test_quicksum():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    x = solver.add_var(name="x", type="c", lb=0, ub=10)
    y = solver.add_var(name="y", type="c", lb=0, ub=10)
    expr = solver.quicksum([2 * x, 3 * y])
    # Should return a highs_linear_expression, not raise
    assert expr is not None


def test_linexpr():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    expr = solver.LinExpr()
    assert expr is not None


def test_add_constr_ge():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    x = solver.add_var(name="x", type="c", lb=0, ub=10)
    y = solver.add_var(name="y", type="c", lb=0, ub=10)
    # Should not raise
    solver.add_constr(x + y >= 5, name="min_sum")


def test_add_constr_eq():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    x = solver.add_var(name="x", type="b")
    y = solver.add_var(name="y", type="b")
    solver.add_constr(x + y == 1, name="one_hot")


def test_operator_overloading_pattern():
    """Test the exact pattern used in models.py: objective = objective + coeff * var"""
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model")
    x = solver.add_var(name="x", type="c", lb=0, ub=10)
    y = solver.add_var(name="y", type="c", lb=0, ub=10)

    # This is the pattern from models.py _build():
    objective = solver.LinExpr()
    objective = objective + 2.5 * x
    objective = objective + 3.0 * y
    assert objective is not None
