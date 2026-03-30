"""ILP preprocessing: reduce problem size before solver.

Applied between candidate building and model.build(). Removes reads
with deterministic assignment and prunes low-support candidates.
"""
from .common import log


def preassign_unambiguous_reads(reads, candidates):
    """Remove reads that map to exactly one candidate.

    These reads have deterministic assignment — they contribute a constant
    to the objective and fixed coverage to their single candidate. Removing
    them shrinks the ILP without changing the optimal genotype calls.

    The coverage from preassigned reads is accounted for by adjusting
    landmark.exp_cov: since these reads WILL be assigned, the model only
    needs to match the REMAINING expected coverage with ILP variables.

    Returns (filtered_reads, filtered_candidates, preassigned_counts).
    """
    # Build read -> candidates mapping
    read_candidates = {}  # read_id -> [candidate objects]
    for candidate in candidates:
        for read in candidate.reads:
            if read.id not in read_candidates:
                read_candidates[read.id] = []
            read_candidates[read.id].append(candidate)

    # Identify unambiguous reads (exactly 1 candidate)
    unambiguous_reads = set()
    preassigned = {}  # candidate_id -> count of preassigned reads

    for read_id, cands in read_candidates.items():
        if len(cands) == 1:
            unambiguous_reads.add(read_id)
            cand = cands[0]
            preassigned[cand.id] = preassigned.get(cand.id, 0) + 1

    if not unambiguous_reads:
        log.info("Preprocessing: no unambiguous reads found")
        return reads, candidates, {}

    # Count preassigned reads covering each landmark (for coverage adjustment)
    # This must be done BEFORE removing reads from landmarks
    preassigned_landmark_coverage = {}  # (candidate_id, landmark_pos) -> count
    for candidate in candidates:
        if candidate.id not in preassigned:
            continue
        for group in candidate.landmark_groups:
            for landmark in group:
                count = sum(1 for r in landmark.reads if r.id in unambiguous_reads)
                if count > 0:
                    preassigned_landmark_coverage[(candidate.id, landmark.pos)] = count

    # Remove unambiguous reads from read list and all data structures
    filtered_reads = [r for r in reads if r.id not in unambiguous_reads]

    for candidate in candidates:
        candidate.reads = [r for r in candidate.reads if r.id not in unambiguous_reads]
        if hasattr(candidate, 'reads_dict'):
            candidate.reads_dict = {k: v for k, v in candidate.reads_dict.items()
                                     if k not in unambiguous_reads}
        for group in candidate.landmark_groups:
            for landmark in group:
                landmark.reads = [r for r in landmark.reads if r.id not in unambiguous_reads]
                # Store preassigned coverage as a constant offset for model to use
                key = (candidate.id, landmark.pos)
                landmark.preassigned_coverage = preassigned_landmark_coverage.get(key, 0)

    total_reads = len(reads)
    removed = len(unambiguous_reads)
    log.info(f"Preprocessing: {removed}/{total_reads} unambiguous reads preassigned "
             f"({removed/total_reads*100:.1f}%), {len(filtered_reads)} remain for ILP")
    log.info(f"  {len(preassigned)} candidates have preassigned reads")

    return filtered_reads, candidates, preassigned


def prune_low_support_candidates(reads, candidates, min_reads=3):
    """Remove candidates with very few supporting reads.

    Candidates with fewer than min_reads are unlikely to be called
    (not enough evidence for any copy number > 0). Removing them
    eliminates all their assignment variables and depth error variables.

    Reads that lose all candidates after pruning are also removed.

    Returns (filtered_reads, filtered_candidates).
    """
    pruned_cands = set()
    for candidate in candidates:
        if len(candidate.reads) < min_reads:
            pruned_cands.add(candidate.id)

    if not pruned_cands:
        log.info("Preprocessing: no candidates pruned")
        return reads, candidates

    # Remove pruned candidates
    filtered_candidates = [c for c in candidates if c.id not in pruned_cands]

    # Remove pruned candidate references from reads
    orphaned_reads = set()
    for read in reads:
        if hasattr(read, 'candidates'):
            read.candidates = [c for c in read.candidates if c.id not in pruned_cands]
            if not read.candidates:
                orphaned_reads.add(read.id)

    filtered_reads = [r for r in reads if r.id not in orphaned_reads]

    log.info(f"Preprocessing: pruned {len(pruned_cands)}/{len(candidates)} candidates "
             f"with <{min_reads} reads, {len(orphaned_reads)} orphaned reads removed")
    log.info(f"  {len(filtered_candidates)} candidates, {len(filtered_reads)} reads remain")

    return filtered_reads, filtered_candidates


def prune_distant_pairs(reads, candidates, seq_error_rate=0.02, discard_multiplier=2, max_excess=None):
    """Remove read-candidate pairs where distance exceeds discard penalty.

    If a read would always be cheaper to discard than to assign to a candidate,
    that pair can never appear in an optimal solution. Removing it eliminates
    an assignment variable and reduces nonzeros.

    For each read, also keeps at least the closest candidate to avoid
    creating reads with no candidates.

    Args:
        max_excess: if set, also prune pairs where distance > min_distance + max_excess

    Returns (reads, candidates) with pruned pairs.
    """
    total_pairs_before = sum(len(c.reads) for c in candidates)
    pairs_removed = 0
    orphaned_reads = set()

    # Build read -> [(candidate, distance)] mapping
    read_info = {}
    for candidate in candidates:
        for read in candidate.reads:
            if read.id not in read_info:
                read_info[read.id] = []
            read_info[read.id].append((candidate, read.get_distance(candidate)))

    # For each read, determine which pairs to keep
    reads_to_remove_from = {}  # candidate_id -> set of read_ids to remove
    for read_id, cand_dists in read_info.items():
        if len(cand_dists) <= 1:
            continue  # don't prune single-candidate reads

        # Compute thresholds
        min_dist = min(d for _, d in cand_dists)
        # Use first read object to get alignment length for discard penalty
        read_obj = cand_dists[0][1]  # actually this is distance, need the read
        # Find the actual read object
        read_obj = cand_dists[0][0].reads_dict.get(read_id)
        if read_obj is None:
            continue
        discard_penalty = round(read_obj.primary_mapping.query_alignment_length * seq_error_rate * discard_multiplier)

        # Threshold: keep pairs where distance <= max(discard_penalty, min_dist + max_excess)
        threshold = discard_penalty
        if max_excess is not None:
            threshold = min(threshold, min_dist + max_excess)

        # Always keep at least the closest candidate
        for cand, dist in cand_dists:
            if dist > threshold and dist > min_dist:
                if cand.id not in reads_to_remove_from:
                    reads_to_remove_from[cand.id] = set()
                reads_to_remove_from[cand.id].add(read_id)

    # Apply removals
    for candidate in candidates:
        remove_ids = reads_to_remove_from.get(candidate.id, set())
        if not remove_ids:
            continue
        before = len(candidate.reads)
        candidate.reads = [r for r in candidate.reads if r.id not in remove_ids]
        if hasattr(candidate, 'reads_dict'):
            candidate.reads_dict = {k: v for k, v in candidate.reads_dict.items()
                                     if k not in remove_ids}
        for group in candidate.landmark_groups:
            for landmark in group:
                landmark.reads = [r for r in landmark.reads if r.id not in remove_ids]
        pairs_removed += before - len(candidate.reads)

    # Update read.candidates lists and check for orphaned reads
    for read in reads:
        if hasattr(read, 'candidates'):
            read.candidates = [c for c in read.candidates
                               if read.id not in reads_to_remove_from.get(c.id, set())]
            if not read.candidates:
                orphaned_reads.add(read.id)

    if orphaned_reads:
        reads = [r for r in reads if r.id not in orphaned_reads]

    total_pairs_after = sum(len(c.reads) for c in candidates)
    log.info(f"Pair pruning: {pairs_removed}/{total_pairs_before} pairs removed "
             f"({pairs_removed/total_pairs_before*100:.1f}% reduction), "
             f"{total_pairs_after} remain, {len(orphaned_reads)} orphaned reads removed")

    return reads, candidates


def find_connected_components(reads, candidates):
    """Find connected components in the read-candidate bipartite graph.

    If reads and candidates form disconnected subgraphs, each can be
    solved independently — a much smaller ILP per component.

    Returns list of (reads_subset, candidates_subset) tuples.
    """
    # Build adjacency: candidate_id -> set of read_ids
    candidate_reads = {}
    for candidate in candidates:
        candidate_reads[candidate.id] = set(r.id for r in candidate.reads)

    # Union-find for reads
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    # Union reads that share a candidate
    for candidate in candidates:
        read_ids = [r.id for r in candidate.reads]
        if len(read_ids) > 1:
            for rid in read_ids[1:]:
                union(read_ids[0], rid)
        # Also union the candidate with its first read (for tracking)
        if read_ids:
            union(f"cand_{candidate.id}", read_ids[0])

    # Group by component
    components = {}  # root -> (read_ids, candidate_ids)
    for read in reads:
        root = find(read.id)
        if root not in components:
            components[root] = (set(), set())
        components[root][0].add(read.id)

    for candidate in candidates:
        root = find(f"cand_{candidate.id}")
        if root not in components:
            components[root] = (set(), set())
        components[root][1].add(candidate.id)

    # Build indexed lookups
    reads_by_id = {r.id: r for r in reads}
    candidates_by_id = {c.id: c for c in candidates}

    result = []
    for root, (read_ids, cand_ids) in components.items():
        comp_reads = [reads_by_id[rid] for rid in read_ids if rid in reads_by_id]
        comp_cands = [candidates_by_id[cid] for cid in cand_ids if cid in candidates_by_id]
        if comp_reads and comp_cands:
            result.append((comp_reads, comp_cands))

    return result


def analyze_components(reads, candidates):
    """Analyze the read-candidate graph structure and log statistics."""
    components = find_connected_components(reads, candidates)

    if len(components) <= 1:
        log.info(f"Preprocessing: single connected component "
                 f"({len(reads)} reads, {len(candidates)} candidates) — no decomposition possible")
        return None

    log.info(f"Preprocessing: found {len(components)} connected components!")
    for i, (comp_reads, comp_cands) in enumerate(components):
        log.info(f"  Component {i}: {len(comp_reads)} reads, {len(comp_cands)} candidates")

    return components


def log_read_ambiguity(reads, candidates):
    """Log statistics about read-candidate ambiguity for diagnostics."""
    # Build read -> candidate count mapping
    read_cand_count = {}
    for candidate in candidates:
        for read in candidate.reads:
            read_cand_count[read.id] = read_cand_count.get(read.id, 0) + 1

    # Histogram
    histogram = {}
    for count in read_cand_count.values():
        histogram[count] = histogram.get(count, 0) + 1

    total = len(read_cand_count)
    unambiguous = histogram.get(1, 0)
    log.info(f"Preprocessing stats: {total} reads, "
             f"{unambiguous} unambiguous ({unambiguous/total*100:.1f}% if removed)")

    # Log top-10 buckets
    for n_cands in sorted(histogram.keys())[:10]:
        count = histogram[n_cands]
        log.info(f"  {n_cands} candidates: {count} reads ({count/total*100:.1f}%)")
