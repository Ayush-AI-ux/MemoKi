"""Memory tree (step 2.3: centroid-only, no cues yet).

Structure: a balanced B-tree-like hierarchy. Leaves are whole conversations (one embedding each).
Internal nodes hold the normalised mean of every leaf beneath them (their `vec`), so a parent is
always a 'more generic description' of its children. All leaves sit at the same depth because the
tree only grows taller when the ROOT splits.

Honest cost accounting: every query-vs-vector similarity (leaf OR internal centroid) is counted as
one embedding comparison, and nodes_visited counts every node that was scored."""
import numpy as np
from .timecue import window_overlaps

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

class Node:
    __slots__ = ("uid", "is_leaf", "conv_id", "vec", "sum", "count", "children", "parent", "_mat", "ents", "tmin", "tmax")

    def __init__(self, uid, is_leaf=False, conv_id=None, vec=None, ents=frozenset(), ts=0.0):
        self.uid, self.is_leaf, self.conv_id = uid, is_leaf, conv_id
        # cue summaries: a parent holds the UNION of its children's entities and the span of their dates,
        # so "entity in node.ents" is guaranteed true for every ancestor of a leaf that has it
        self.ents = ents if is_leaf else set()
        self.tmin, self.tmax = (ts, ts) if is_leaf else (float("inf"), float("-inf"))
        self.vec = vec                                   # unit vector (leaf embedding or normalised centroid)
        self.sum = vec.astype(np.float64) if is_leaf else None
        self.count = 1 if is_leaf else 0
        self.children, self.parent, self._mat = [], None, None

    def child_matrix(self):
        if self._mat is None:
            self._mat = np.stack([c.vec for c in self.children])
        return self._mat

    def refresh(self):
        """Recompute sum/count/vec from children (used after a split)."""
        self.sum = np.sum([c.sum for c in self.children], axis=0)
        self.count = sum(c.count for c in self.children)
        self.vec = _unit(self.sum).astype(np.float32)
        self._mat = None
        self.ents = set().union(*(c.ents for c in self.children))
        self.tmin = min(c.tmin for c in self.children)
        self.tmax = max(c.tmax for c in self.children)

class MemoryTree:
    def __init__(self, max_children: int = 8, min_fill: float = 0.0):
        """min_fill: every node created by a split keeps at least ceil(min_fill * (B+1)) children
        (0 = plain 2-means, which can leave nodes with a single child; must be <= 0.5)."""
        if max_children < 3:
            raise ValueError("max_children must be >= 3")
        if not 0.0 <= min_fill <= 0.5:
            raise ValueError("min_fill must be between 0 and 0.5")
        self.B, self.min_fill = max_children, min_fill
        self.root = None
        self.leaves = {}                 # conv_id -> leaf Node (parent pointers allow later re-filing, Gap 2)
        self.build_comparisons = 0
        self._uid = 0
        self._flat = None                # cached matrix of all leaves (for the flat fallback)
        self._eindex = None              # cached entity -> leaves index (for the entity-lookup recovery)

    def _node(self, **kw):
        self._uid += 1
        return Node(self._uid, **kw)

    # ---------------------------------------------------------------- insertion
    def insert(self, conv_id: str, embedding: np.ndarray, entities=frozenset(), timestamp: float = 0.0):
        if conv_id in self.leaves:
            raise ValueError(f"duplicate conversation id {conv_id}")
        e = _unit(np.asarray(embedding, dtype=np.float32))
        leaf = self._node(is_leaf=True, conv_id=conv_id, vec=e, ents=frozenset(entities), ts=float(timestamp))
        self.leaves[conv_id] = leaf
        self._flat = None
        self._eindex = None
        if self.root is None:
            self.root = self._node()
            self.root.sum, self.root.vec = np.zeros_like(leaf.sum), e.copy()
        node, path = self.root, [self.root]
        while node.children and not node.children[0].is_leaf:
            sims = node.child_matrix() @ e
            self.build_comparisons += len(node.children)
            node = node.children[int(np.argmax(sims))]       # first max wins -> deterministic
            path.append(node)
        node.children.append(leaf)
        leaf.parent = node
        for n in path:                                       # update every ancestor's description
            n.sum = n.sum + leaf.sum
            n.count += 1
            n.vec = _unit(n.sum).astype(np.float32)
            n._mat = None
            n.ents |= leaf.ents
            n.tmin, n.tmax = min(n.tmin, leaf.tmin), max(n.tmax, leaf.tmax)
        for i in range(len(path) - 1, -1, -1):               # split overfull nodes, bottom-up
            if len(path[i].children) <= self.B:
                break
            self._split(path[i], path[i - 1] if i > 0 else None)

    def _split(self, node, parent):
        kids = node.children
        V = np.stack([c.vec for c in kids])
        i1 = int(np.argmin(V @ node.vec))                    # farthest from the centre
        i2 = int(np.argmin(V @ V[i1]))                       # farthest from that
        c1, c2 = V[i1], V[i2]
        for _ in range(5):                                   # 2-means, fixed iterations
            g1 = (V @ c1) >= (V @ c2)
            if g1.all() or (~g1).all():
                break
            n1, n2 = _unit(V[g1].mean(0)), _unit(V[~g1].mean(0))
            if np.allclose(n1, c1) and np.allclose(n2, c2):
                break
            c1, c2 = n1, n2
        g1 = (V @ c1) >= (V @ c2)
        if g1.all() or (~g1).all():                          # degenerate (e.g. identical vectors): halve
            order = np.argsort(-(V @ c1), kind="stable")
            g1 = np.zeros(len(kids), dtype=bool)
            g1[order[: len(kids) // 2]] = True
        m = len(kids)
        min_size = max(1, int(np.ceil(self.min_fill * m)))
        n1 = int(g1.sum())
        target = min(max(n1, min_size), m - min_size)
        if target != n1:                                     # rebalance: move the most borderline children
            order = np.argsort(-((V @ c1) - (V @ c2)), kind="stable")
            g1 = np.zeros(m, dtype=bool)
            g1[order[:target]] = True
        keep = [k for k, f in zip(kids, g1) if f]
        move = [k for k, f in zip(kids, g1) if not f]
        sibling = self._node()
        node.children, sibling.children = keep, move
        for c in keep: c.parent = node
        for c in move: c.parent = sibling
        node.refresh(); sibling.refresh()
        if parent is None:                                   # root split: the only way the tree grows taller
            new_root = self._node()
            new_root.children = [node, sibling]
            node.parent = sibling.parent = new_root
            new_root.refresh()
            self.root = new_root
        else:
            sibling.parent = parent
            parent.children.insert(parent.children.index(node) + 1, sibling)
            parent._mat = None

    # ------------------------------------------------------------------- search
    @staticmethod
    def _passes(node, qc, counters):
        """Cheap cue test: set membership and a range check, no vector arithmetic.
        Cost model: one cue check per query entity looked up, one for the date-range test."""
        if qc.entities:
            counters.cue_checks += len(qc.entities)
            if node.ents.isdisjoint(qc.entities):
                return False
        if qc.window is not None:
            counters.cue_checks += 1
            if not window_overlaps(node.tmin, node.tmax, qc.window, qc.slack_days):
                return False
        return True

    def search(self, query, k, beam, counters, qcues=None):
        """Top-down beam search. With cues: children failing the cue test are dropped BEFORE any
        vector comparison. If no child of the whole level matches (entity unknown to memory, or
        extracted differently), pruning is skipped for that level and ranking uses similarity alone.
        Returns [(conv_id, score)] best first (possibly fewer than k when cues are selective)."""
        if self.root is None or not self.root.children:
            return []
        q = _unit(np.asarray(query, dtype=np.float32))
        active = qcues is not None and qcues.active
        frontier = [self.root]
        while True:
            groups, survivors = [], 0
            for node in frontier:
                counters.nodes_visited += len(node.children)
                if active:
                    mask = [self._passes(c, qcues, counters) for c in node.children]
                    survivors += sum(mask)
                else:
                    mask = None
                groups.append((node, mask))
            relaxed = active and survivors == 0
            if relaxed:
                counters.levels_relaxed += 1
            scored = []
            for node, mask in groups:
                if mask is None or relaxed:
                    kids, sims = node.children, node.child_matrix() @ q
                else:
                    idx = [i for i, m in enumerate(mask) if m]
                    if not idx:
                        continue
                    kids = [node.children[i] for i in idx]
                    sims = node.child_matrix()[idx] @ q
                counters.embedding_comparisons += len(kids)
                scored.extend(zip(sims.tolist(), kids))
            scored.sort(key=lambda t: (-t[0], t[1].uid))
            if scored[0][1].is_leaf:
                return [(n.conv_id, s) for s, n in scored[:k]]
            frontier = [n for _, n in scored[:beam]]

    def flat_search(self, query, k, counters):
        """Brute force over every leaf. Used as the safety fallback; fully counted."""
        if self.root is None:
            return []
        if self._flat is None:
            leaves = list(self.leaves.values())
            self._flat = (np.stack([l.vec for l in leaves]), [l.conv_id for l in leaves])
        mat, ids = self._flat
        sims = mat @ _unit(np.asarray(query, dtype=np.float32))
        counters.embedding_comparisons += len(ids)
        counters.nodes_visited += len(ids)
        top = np.argsort(-sims, kind="stable")[:k]
        return [(ids[i], float(sims[i])) for i in top]

    # ------------------------------------------------- misfiling support (Gap 2)
    def leaf_parents(self):
        """Internal nodes whose children are leaves (where a conversation is 'filed')."""
        out, stack = [], [self.root] if self.root else []
        while stack:
            n = stack.pop()
            if n.children and n.children[0].is_leaf:
                out.append(n)
            else:
                stack.extend(reversed(n.children))
        return sorted(out, key=lambda n: n.uid)

    def top_branch(self, node):
        """The child of the root that contains `node` (the node itself if it is the root)."""
        while node.parent is not None and node.parent is not self.root:
            node = node.parent
        return node

    def move_leaves(self, moves):
        """Batch re-filing: moves = [(conv_id, target_leaf_parent)]. Used to simulate misfiling.
        Structure only changes at the leaf level (depth stays uniform); empty nodes are removed and every
        affected ancestor is recomputed ONCE, so centroids, entity unions and date spans stay truthful.
        A target may end up with more than max_children children (reported, not fixed)."""
        touched = set()
        for cid, target in moves:
            leaf = self.leaves[cid]
            old = leaf.parent
            if old is target:
                continue
            old.children.remove(leaf)
            target.children.append(leaf)
            leaf.parent = target
            touched.update((old, target))
        work = set()
        for n in touched:
            while n is not None:
                work.add(n); n = n.parent
        def depth(n):
            d = 0
            while n.parent is not None:
                n, d = n.parent, d + 1
            return d
        for n in sorted(work, key=depth, reverse=True):          # deepest first
            if not n.children and n is not self.root:
                n.parent.children.remove(n)
            else:
                n.refresh()
        self._flat = None

    def audit_refile(self, margin=0.05, rounds=1):
        """Offline repair. A leaf is re-filed when another leaf-group's centroid is closer to it than
        its OWN group's centroid computed WITHOUT it, by more than `margin` (cosine). Runs once, offline:
        it costs (leaves x leaf-groups) comparisons and adds nothing at query time. Returns statistics."""
        stats = {"moved": [], "comparisons": 0}
        for _ in range(rounds):
            parents = self.leaf_parents()
            if len(parents) < 2:
                break
            P = np.stack([p.vec for p in parents])
            pos = {p.uid: i for i, p in enumerate(parents)}
            leaves = list(self.leaves.values())
            S = np.stack([l.vec for l in leaves]) @ P.T
            stats["comparisons"] += S.size
            moves = []
            for r, leaf in enumerate(leaves):
                own = leaf.parent; i = pos[own.uid]
                if len(own.children) > 1:
                    S[r, i] = float(_unit(own.sum - leaf.sum).astype(np.float32) @ leaf.vec)   # leave-one-out
                else:
                    S[r, i] = 1.0                                   # a one-leaf group is never 'wrong'
                j = int(np.argmax(S[r]))
                if j != i and S[r, j] - S[r, i] > margin:
                    moves.append((leaf.conv_id, parents[j]))
            if not moves:
                break
            self.move_leaves(moves)
            stats["moved"].extend(c for c, _ in moves)
        return stats

    def _entity_index(self):
        if self._eindex is None:
            idx = {}
            for leaf in self.leaves.values():
                for e in leaf.ents:
                    idx.setdefault(e, []).append(leaf)
            self._eindex = idx
        return self._eindex

    def entity_flat_search(self, query, k, counters, qcues):
        """Recovery tier B: look the query's entities up in a flat entity index (and apply the date
        window), then score only those candidates exactly. Returns [] if the query has no entity or
        nothing matches (so the caller keeps its original answer)."""
        if self.root is None or not qcues.entities:
            return []
        counters.cue_checks += len(qcues.entities)
        cands = {}
        for e in qcues.entities:
            for leaf in self._entity_index().get(e, ()):
                cands[leaf.uid] = leaf
        if qcues.window is not None:
            kept = {}
            for u, leaf in cands.items():
                counters.cue_checks += 1
                if window_overlaps(leaf.tmin, leaf.tmax, qcues.window, qcues.slack_days):
                    kept[u] = leaf
            cands = kept
        if not cands:
            return []
        leaves = [cands[u] for u in sorted(cands)]
        sims = np.stack([l.vec for l in leaves]) @ _unit(np.asarray(query, dtype=np.float32))
        counters.embedding_comparisons += len(leaves)
        counters.nodes_visited += len(leaves)
        top = np.argsort(-sims, kind="stable")[:k]
        return [(leaves[i].conv_id, float(sims[i])) for i in top]

    # ----------------------------------------------------------------- inspection
    def iter_leaves(self):
        stack = [self.root] if self.root else []
        while stack:
            n = stack.pop()
            if n.is_leaf: yield n
            else: stack.extend(reversed(n.children))

    def stats(self):
        if self.root is None:
            return {"leaves": 0, "height": 0, "internal_nodes": 0}
        depths, internal, branch = set(), 0, []
        stack = [(self.root, 1)]
        while stack:
            n, d = stack.pop()
            if n.is_leaf: depths.add(d)
            else:
                internal += 1; branch.append(len(n.children))
                stack.extend((c, d + 1) for c in n.children)
        return {"leaves": len(self.leaves), "height": max(depths) - 1, "internal_nodes": internal,
                "uniform_depth": len(depths) == 1, "mean_children": round(float(np.mean(branch)), 2),
                "max_children": max(branch), "min_children": min(branch)}

    def to_dict(self, max_depth=None):
        """Plain nested dict (for the dashboard / server later)."""
        def rec(n, d):
            if n.is_leaf: return {"id": n.uid, "conv_id": n.conv_id}
            out = {"id": n.uid, "count": n.count}
            if max_depth is not None and d >= max_depth: out["collapsed"] = True
            else: out["children"] = [rec(c, d + 1) for c in n.children]
            return out
        return rec(self.root, 1) if self.root else {}
