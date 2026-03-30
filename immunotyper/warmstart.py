"""Greedy heuristic for constructing a warm-start solution to the ImmunoTyper ILP."""

from collections import defaultdict
from statistics import median
from .common import log


def construct_greedy_solution(model, reads, candidates, maxcopy=4):
    """Construct a feasible MIP solution greedily and return column values.

    Strategy:
    1. Assign each read to its best candidate (no reassignment later).
    2. Estimate copies from landmark coverage of assigned reads.
    3. Validate per-candidate constraints, reducing copies if needed.
    4. Reads whose best candidate has copies=0 are discarded.

    This avoids cascading infeasibilities from read reassignment.

    Args:
        model: The built ShortReadModelTotalErrorDiscardObj instance
        reads: List of Read objects (same as passed to model.build)
        candidates: List of Candidate objects (same as passed to model.build)
        maxcopy: Maximum copy number

    Returns:
        list: Column values array of length model._var_count.
    """
    stdev_coeff = model.stdev_coefficient
    min_landmark_depth = model.min_landmark_depth

    # Step 1: Assign each read to its best (lowest distance) candidate — FIXED
    read_assignment = {}
    for read in reads:
        if not hasattr(read, 'assignment') or not read.assignment:
            read_assignment[read.id] = None
            continue
        best_cand_id = min(read.assignment.keys(),
                          key=lambda cid: read.get_distance(cid))
        read_assignment[read.id] = best_cand_id

    # Step 2: Compute per-candidate assigned read sets
    cand_reads = defaultdict(set)
    for read in reads:
        cid = read_assignment.get(read.id)
        if cid is not None:
            cand_reads[cid].add(read.id)

    # Step 3: Estimate copy numbers from assigned-read landmark coverage
    candidate_copies = {}
    for cand in candidates:
        assigned = cand_reads.get(cand.id, set())
        if not assigned:
            candidate_copies[cand.id] = 0
            continue

        ratios = []
        for group in cand.landmark_groups:
            for landmark in group:
                if landmark.exp_cov > 0:
                    covering = sum(1 for r in landmark.reads if r.id in assigned)
                    ratios.append(covering / landmark.exp_cov)

        if ratios:
            med_ratio = median(ratios)
            est_copies = max(0, min(maxcopy, round(med_ratio)))
        else:
            est_copies = 1 if assigned else 0

        candidate_copies[cand.id] = est_copies

    # Step 4: Validate per-candidate constraints, reduce copies if needed
    for cand in candidates:
        est = candidate_copies[cand.id]
        if est == 0:
            continue

        assigned = cand_reads.get(cand.id, set())
        est = _find_valid_copies(cand, est, assigned, stdev_coeff, min_landmark_depth)
        candidate_copies[cand.id] = est

    # Step 5: Reads assigned to candidates with copies=0 become discarded
    for read in reads:
        cid = read_assignment.get(read.id)
        if cid is not None and candidate_copies.get(cid, 0) == 0:
            read_assignment[read.id] = None

    called = {cid for cid, copies in candidate_copies.items() if copies >= 1}
    assigned_count = sum(1 for v in read_assignment.values() if v is not None)
    discarded_count = sum(1 for v in read_assignment.values() if v is None)
    log.info(f"Warm start: {len(called)}/{len(candidates)} candidates called, "
             f"{assigned_count} reads assigned, {discarded_count} discarded")

    # Step 6: Compute depth error variables
    depth_errors = {}
    for cand in candidates:
        est = candidate_copies[cand.id]
        for i, group in enumerate(cand.landmark_groups):
            for j, landmark in enumerate(group):
                covering = sum(1 for r in landmark.reads
                             if read_assignment.get(r.id) == cand.id)
                depth_diff = covering - landmark.exp_cov * est
                depth_errors[(cand.id, i, j)] = max(0, int(round(abs(depth_diff))))

    # Step 7: Build variable name -> value mapping
    var_values = {}

    for cand in candidates:
        est = candidate_copies[cand.id]
        for copy_num in range(maxcopy + 1):
            var_name = f'copies|{copy_num}|{cand.id}'
            var_values[var_name] = 1.0 if copy_num == est else 0.0

    for read in reads:
        assigned_cand = read_assignment.get(read.id)
        discard_name = f'discard|{read.id}'
        var_values[discard_name] = 1.0 if assigned_cand is None else 0.0

        if hasattr(read, 'assignment'):
            for cand_id in read.assignment:
                d_name = f'D|{read.id}|{cand_id}'
                var_values[d_name] = 1.0 if cand_id == assigned_cand else 0.0

    for cand in candidates:
        for i, group in enumerate(cand.landmark_groups):
            for j, landmark in enumerate(group):
                var_name = f'diff_abs_{cand.id}_group{i}_landmark{j}'
                var_values[var_name] = float(depth_errors.get((cand.id, i, j), 0))

    # Step 8: Convert to column-indexed array
    num_cols = model._var_count
    col_values = [0.0] * num_cols

    matched = 0
    for i, (name, vtype, lb, ub) in enumerate(model._var_specs):
        if name in var_values:
            col_values[i] = var_values[name]
            matched += 1

    log.info(f"Warm start: set {matched}/{num_cols} variables")

    obj = _estimate_objective(reads, read_assignment)
    log.info(f"Warm start: estimated objective = {obj:.0f}")

    return col_values


def _find_valid_copies(cand, est, assigned_read_ids, stdev_coeff, min_landmark_depth):
    """Find the highest copy number <= est that satisfies constraints."""
    while est > 0:
        violated = False
        for group in cand.landmark_groups:
            total_error = 0.0
            total_allowed = 0.0
            for landmark in group:
                covering = sum(1 for r in landmark.reads if r.id in assigned_read_ids)
                depth_diff = abs(covering - landmark.exp_cov * est)
                total_error += depth_diff
                total_allowed += stdev_coeff * landmark.stdevs[est]

                min_required = min_landmark_depth * landmark.exp_cov * est
                if covering < min_required:
                    violated = True
                    break

            if violated or total_error > total_allowed:
                violated = True
                break

        if not violated:
            break
        est -= 1
    return est


def _estimate_objective(reads, read_assignment):
    """Estimate the objective value of the greedy solution (for logging)."""
    total = 0.0
    for read in reads:
        assigned_cand = read_assignment.get(read.id)
        if assigned_cand is None:
            if hasattr(read, 'discard_penalty'):
                total += read.discard_penalty
            else:
                total += 6
        else:
            total += read.get_distance(assigned_cand)
    return total
