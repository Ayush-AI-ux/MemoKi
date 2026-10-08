"""Memory tree (step 2.3: centroid-only, no cues yet).

Structure: a balanced B-tree-like hierarchy. Leaves are whole conversations (one embedding each).
Internal nodes hold the normalised mean of every leaf beneath them (their `vec`), so a parent is
always a 'more generic description' of its children. All leaves sit at the same depth because the
tree only grows taller when the ROOT splits.

Honest cost accounting: every query-vs-vector similarity (leaf OR internal centroid) is counted as
one embedding comparison, and nodes_visited counts every node that was scored."""
import numpy as np

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

class Node:
    __slots__ = ("uid", "is_leaf", "conv_id", "vec", "sum", "count", "children", "parent", "_mat")

    def __init__(self, uid, is_leaf=False, conv_id=None, vec=None):
        self.uid, self.is_leaf, self.conv_id = uid, is_leaf, conv_id
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

    def _node(self, **kw):
        self._uid += 1
        return Node(self._uid, **kw)

    # ---------------------------------------------------------------- insertion
    def insert(self, conv_id: str, embedding: np.ndarray):
        if conv_id in self.leaves:
            raise ValueError(f"duplicate conversation id {conv_id}")
        e = _unit(np.asarray(embedding, dtype=np.float32))
        leaf = self._node(is_leaf=True, conv_id=conv_id, vec=e)
        self.leaves[conv_id] = leaf
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
    def search(self, query, k, beam, counters):
        """Top-down beam search. Returns [(conv_id, score)] best first."""
        if self.root is None or not self.root.children:
            return []
        q = _unit(np.asarray(query, dtype=np.float32))
        frontier = [self.root]
        while True:
            scored = []
            for node in frontier:
                sims = node.child_matrix() @ q
                counters.embedding_comparisons += len(node.children)
                counters.nodes_visited += len(node.children)
                scored.extend(zip(sims.tolist(), node.children))
            scored.sort(key=lambda t: (-t[0], t[1].uid))
            if scored[0][1].is_leaf:
                return [(n.conv_id, s) for s, n in scored[:k]]
            frontier = [n for _, n in scored[:beam]]

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
