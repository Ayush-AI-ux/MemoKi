"""Deliberate misfiling for the Gap 2 experiment. Everything is seeded and deterministic."""
import random

def pick_misfiled(evidence_ids, other_ids, fraction, seed):
    """Choose conversations to misfile. Evidence leaves and all other leaves are sampled SEPARATELY
    (stratified), so that `fraction` of the evidence leaves are really affected. A uniform sample of
    all leaves would barely touch the few hundred evidence leaves and the test would have no power."""
    rng = random.Random(seed)
    chosen = []
    for group in (sorted(evidence_ids), sorted(other_ids)):
        n = int(round(fraction * len(group)))
        chosen.extend(rng.sample(group, n))
    return sorted(chosen)

def plan_moves(tree, conv_ids, seed):
    """For each conversation, pick a leaf-parent under a DIFFERENT top-level branch (a 'wrong branch')."""
    rng = random.Random(seed + 7919)
    parents = tree.leaf_parents()
    branch = {p.uid: tree.top_branch(p).uid for p in parents}
    if len({b for b in branch.values()}) < 2:
        raise ValueError("tree has a single top-level branch: nothing is 'wrong' to misfile into")
    moves = []
    for cid in sorted(conv_ids):
        own = tree.top_branch(tree.leaves[cid].parent).uid
        options = [p for p in parents if branch[p.uid] != own]
        moves.append((cid, rng.choice(options)))
    return moves
