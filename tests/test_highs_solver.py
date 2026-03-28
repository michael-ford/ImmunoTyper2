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


def test_solve_minimize():
    """Solve: minimize 2x + 3y subject to x + y >= 5, x,y >= 0, x,y <= 10"""
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_min")
    x = solver.add_var(name="x", type="c", lb=0, ub=10)
    y = solver.add_var(name="y", type="c", lb=0, ub=10)
    solver.add_constr(x + y >= 5, name="min_sum")
    solver.objective = 2 * x + 3 * y
    status, obj_val = solver.solve(time_limit=60, threads=1, log_path="/dev/null")
    assert status in ("optimal", "feasible")
    assert abs(obj_val - 10.0) < 0.01  # optimal: x=5, y=0 -> 10


def test_solve_binary_ilp():
    """Solve a small binary ILP: minimize x + y, x + y == 1, x,y binary"""
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_binary")
    x = solver.add_var(name="x", type="b")
    y = solver.add_var(name="y", type="b")
    solver.add_constr(x + y == 1, name="one_hot")
    solver.objective = x + y
    status, obj_val = solver.solve(time_limit=60, threads=1, log_path="/dev/null")
    assert status in ("optimal", "feasible")
    assert abs(obj_val - 1.0) < 0.01


def test_get_value_binary():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_getval")
    x = solver.add_var(name="x", type="b")
    y = solver.add_var(name="y", type="b")
    solver.add_constr(x + y == 1, name="one_hot")
    solver.objective = x  # minimize x -> x=0, y=1
    solver.solve(time_limit=60, threads=1, log_path="/dev/null")
    assert solver.get_value(x) == False
    assert solver.get_value(y) == True


def test_get_value_integer():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_int")
    x = solver.add_var(name="x", type="i", lb=0, ub=10)
    solver.add_constr(x >= 3, name="lb")
    solver.objective = x
    solver.solve(time_limit=60, threads=1, log_path="/dev/null")
    assert solver.get_value(x) == 3


def test_get_value_continuous():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_cont")
    x = solver.add_var(name="x", type="c", lb=0, ub=10)
    solver.add_constr(x >= 2.5, name="lb")
    solver.objective = x
    solver.solve(time_limit=60, threads=1, log_path="/dev/null")
    assert abs(solver.get_value(x) - 2.5) < 0.001


def test_solve_infeasible():
    """x >= 5 and x <= 3 is infeasible"""
    from immunotyper.solvers import HiGHSSolver, NoSolutionsError
    solver = HiGHSSolver("test_infeasible")
    x = solver.add_var(name="x", type="c", lb=0, ub=3)
    solver.add_constr(x >= 5, name="impossible")
    solver.objective = x
    with pytest.raises(NoSolutionsError):
        solver.solve(time_limit=60, threads=1, log_path="/dev/null")


def test_var_name():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_name")
    x = solver.add_var(name="my_var", type="b")
    assert HiGHSSolver.var_name(x) == "my_var"


def test_update_noop():
    """update() should exist and not crash — HiGHS doesn't need lazy updates"""
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_update")
    solver.update()  # should not raise


def test_change_ub():
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_ub")
    x = solver.add_var(name="x", type="c", lb=0, ub=10)
    solver.changeUb(x, 5)
    solver.add_constr(x >= 0, name="dummy")
    solver.objective = -1 * x  # maximize x by minimizing -x
    solver.solve(time_limit=60, threads=1, log_path="/dev/null")
    assert abs(solver.get_value(x) - 5.0) < 0.01


def test_model_pattern_integration():
    """
    Mimics the exact pattern from ShortReadModelTotalErrorDiscardObj._build():
    - LinExpr() accumulation for objective
    - Binary copy variables with one-hot constraint
    - Read-assignment binary variables
    - Pseudo-abs value constraints for depth error
    - Quicksum over variable lists
    """
    from immunotyper.solvers import HiGHSSolver
    solver = HiGHSSolver("test_model_pattern")

    # Simulate 2 allele candidates, 3 reads
    maxcopy = 4
    num_reads = 3
    num_candidates = 2

    objective = solver.LinExpr()
    objective_array = []

    # Per-candidate: one-hot copy encoding (exactly like models.py constraint 1)
    candidate_copies = []
    candidate_copy_multipliers = []
    for c in range(num_candidates):
        copies = [solver.add_var(type='b', name=f'copies|{x}|cand{c}') for x in range(maxcopy + 1)]
        copy_multiplier = solver.quicksum([i * copy_var for i, copy_var in enumerate(copies)])
        solver.add_constr(solver.quicksum(copies) == 1, name=f"constraint_1-cand{c}")
        candidate_copies.append(copies)
        candidate_copy_multipliers.append(copy_multiplier)

    # Per-read: assignment and discard variables
    read_assignments = {}
    for r in range(num_reads):
        discard_var = solver.add_var(type='b', name=f'discard|read{r}')
        discard_penalty = 5  # simplified
        objective_array.append(discard_penalty * discard_var)

        assignments = []
        for c in range(num_candidates):
            d_var = solver.add_var(type='b', name=f'D|read{r}|cand{c}')
            distance = (r + 1) * (c + 1)  # simplified distance
            objective = objective + distance * d_var
            objective_array.append(distance * d_var)
            assignments.append(d_var)

        # Constraint: assigned to exactly one candidate or discarded
        solver.add_constr(solver.quicksum(assignments) == 1 - discard_var, name=f'assign-read{r}')
        read_assignments[r] = assignments

    # Constraint 3 pattern: enforce assigned reads -> called allele
    for c in range(num_candidates):
        assignment_vars = [read_assignments[r][c] for r in range(num_reads)]
        solver.add_constr(
            solver.quicksum(candidate_copies[c][1:]) * num_reads >= solver.quicksum(assignment_vars),
            name=f"constr_4-cand{c}"
        )

    # Pseudo-abs value constraint pattern (depth error)
    for c in range(num_candidates):
        exp_cov = 10.0
        reads_covering = [read_assignments[r][c] for r in range(num_reads)]
        depth_diff_array = list(reads_covering) + [-exp_cov * candidate_copy_multipliers[c]]

        diff_abs = solver.add_var(name=f'diff_abs_cand{c}', lb=0, type='i')
        solver.add_constr(diff_abs + solver.quicksum(depth_diff_array) >= 0, name=f'abs_1-cand{c}')
        solver.add_constr(diff_abs - solver.quicksum(depth_diff_array) >= 0, name=f'abs_2-cand{c}')

    solver.objective = solver.quicksum(objective_array)
    status, obj_val = solver.solve(time_limit=60, threads=1, log_path="/dev/null")

    assert status in ("optimal", "feasible")

    # Verify one-hot: exactly one copy level selected per candidate
    for c in range(num_candidates):
        copy_values = [solver.get_value(v) for v in candidate_copies[c]]
        assert sum(1 for v in copy_values if v) == 1, f"Candidate {c} copy one-hot violated: {copy_values}"

    # Verify assignment: each read assigned to exactly one candidate or discarded
    for r in range(num_reads):
        assigned = sum(1 for c in range(num_candidates) if solver.get_value(read_assignments[r][c]))
        assert assigned <= 1, f"Read {r} assigned to multiple candidates"
