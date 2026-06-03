#!/usr/bin/env python3
"""Enumerate local p=7 BT-tree polymer diagnostics around a fixed edge.

This script is a finite, deterministic diagnostic for the DASHI Yang-Mills
carrier lane.  It is not a constructive-QFT proof and does not promote Clay
Yang-Mills.

Plaquette model
---------------
Plaquettes are represented by triples ``(depth, position, edge_type)``.  The
triple denotes the oriented edge from the tree vertex ``(depth, position)`` to
``(depth + 1, position * (p + 1) + edge_type)`` in the local rooted
Bruhat-Tits sampler for ``p = 7``.  The local valency parameter is therefore
``p + 1 = 8``.  Two plaquettes are adjacent when their oriented edges share a
tree endpoint.  The fixed edge is ``e0 = (0, 0, 0)``.

Enumeration
-----------
For diameter ``D in {1, 2}``, the script enumerates minimal connected
plaquette polymers containing ``e0`` whose graph diameter is exactly ``D``.
Minimal means cardinality ``D + 1`` for this smoke diagnostic: diameter-1
polymers are adjacent pairs ``{e0, e1}``; diameter-2 polymers are connected
triples containing ``e0`` with graph diameter two.  This is the finite local
enumerator requested for the p=7 lane, not an all-subsets cluster expansion.

Carrier action
--------------
The concrete action is a deliberately simple positive local carrier model:

  S_carrier(Gamma) =
      plaquette_tension * |Gamma|
    + boundary_tension  * boundary_edges(Gamma) / (p + 1)
    + depth_penalty     * depth_span(Gamma)
    + type_penalty      * distinct_edge_types(Gamma)
    + diameter_penalty  * diameter(Gamma)

with parameters:

  plaquette_tension = 0.42
  boundary_tension  = 0.18
  depth_penalty     = 0.07
  type_penalty      = 0.03
  diameter_penalty  = 0.05

These constants are chosen to make the smoke diagnostic numerically stable and
monotone in obvious local-complexity features.  They are not fitted physical
parameters.  ``kp_ratio < 1`` is only a finite diagnostic-safe result for this
model, not a Clay proof.

Output CSV columns:

  diameter,count,z_sum,weighted_sum,beta,kp_ratio,promotion_status
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

P = 7
VALENCY = P + 1
E0 = (0, 0, 0)
BETAS = (6.0, 7.69, 10.13, 13.64, 16.7)
A_WEIGHT = 0.5
PROMOTION_STATUS = "NO_PROMOTION"


Plaquette = tuple[int, int, int]


@dataclass(frozen=True)
class ActionParameters:
    plaquette_tension: float = 0.42
    boundary_tension: float = 0.18
    depth_penalty: float = 0.07
    type_penalty: float = 0.03
    diameter_penalty: float = 0.05


@dataclass(frozen=True)
class PolymerStats:
    size: int
    diameter: int
    boundary_edges: int
    depth_span: int
    distinct_edge_types: int
    action: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/ym_p7_polymer_enumerator"),
        help="directory for ym_p7_polymer_enumerator.csv",
    )
    return parser.parse_args()


def endpoints(edge: Plaquette) -> tuple[tuple[int, int], tuple[int, int]]:
    depth, position, edge_type = edge
    if depth < 0:
        raise ValueError(f"depth must be nonnegative, got {edge!r}")
    if edge_type < 0 or edge_type >= VALENCY:
        raise ValueError(f"edge_type must be in [0,{VALENCY}), got {edge!r}")
    source = (depth, position)
    target = (depth + 1, position * VALENCY + edge_type)
    return source, target


def outgoing(vertex: tuple[int, int]) -> Iterator[Plaquette]:
    depth, position = vertex
    if depth < 0:
        return
    for edge_type in range(VALENCY):
        yield (depth, position, edge_type)


def incoming(vertex: tuple[int, int]) -> Plaquette | None:
    depth, position = vertex
    if depth <= 0:
        return None
    parent_position, edge_type = divmod(position, VALENCY)
    return (depth - 1, parent_position, edge_type)


def neighbors(edge: Plaquette) -> set[Plaquette]:
    """Plaquettes sharing a source or target endpoint with ``edge``."""
    source, target = endpoints(edge)
    out: set[Plaquette] = set()
    for vertex in (source, target):
        parent = incoming(vertex)
        if parent is not None:
            out.add(parent)
        out.update(outgoing(vertex))
    out.discard(edge)
    return out


def ball(center: Plaquette, radius: int) -> set[Plaquette]:
    seen = {center}
    frontier = {center}
    for _ in range(radius):
        nxt: set[Plaquette] = set()
        for edge in frontier:
            nxt.update(neighbors(edge))
        nxt.difference_update(seen)
        seen.update(nxt)
        frontier = nxt
    return seen


def graph_distance(a: Plaquette, b: Plaquette, allowed: set[Plaquette] | None = None) -> int:
    if a == b:
        return 0
    frontier = {a}
    seen = {a}
    distance = 0
    while frontier:
        distance += 1
        nxt: set[Plaquette] = set()
        for edge in frontier:
            for nb in neighbors(edge):
                if allowed is not None and nb not in allowed:
                    continue
                if nb == b:
                    return distance
                if nb not in seen:
                    seen.add(nb)
                    nxt.add(nb)
        frontier = nxt
    raise RuntimeError(f"disconnected plaquette graph for {a!r}, {b!r}")


def pairwise_distances(vertices: list[Plaquette]) -> dict[tuple[Plaquette, Plaquette], int]:
    distances: dict[tuple[Plaquette, Plaquette], int] = {}
    allowed = set(vertices)
    for i, a in enumerate(vertices):
        for b in vertices[i:]:
            d = graph_distance(a, b, allowed)
            distances[(a, b)] = d
            distances[(b, a)] = d
    return distances


def is_connected(polymer: frozenset[Plaquette]) -> bool:
    if not polymer:
        return False
    start = next(iter(polymer))
    seen = {start}
    frontier = {start}
    while frontier:
        nxt: set[Plaquette] = set()
        for edge in frontier:
            nxt.update(neighbors(edge).intersection(polymer))
        nxt.difference_update(seen)
        seen.update(nxt)
        frontier = nxt
    return seen == set(polymer)


def diameter_of(polymer: frozenset[Plaquette], distances: dict[tuple[Plaquette, Plaquette], int]) -> int:
    vertices = list(polymer)
    diameter = 0
    for i, a in enumerate(vertices):
        for b in vertices[i + 1 :]:
            diameter = max(diameter, distances[(a, b)])
    return diameter


def boundary_edge_count(polymer: frozenset[Plaquette]) -> int:
    boundary = 0
    for edge in polymer:
        for nb in neighbors(edge):
            if nb not in polymer:
                boundary += 1
    return boundary


def carrier_action(polymer: frozenset[Plaquette], diameter: int, params: ActionParameters) -> PolymerStats:
    depths = [edge[0] for edge in polymer]
    edge_types = {edge[2] for edge in polymer}
    boundary = boundary_edge_count(polymer)
    depth_span = max(depths) - min(depths)
    action = (
        params.plaquette_tension * len(polymer)
        + params.boundary_tension * boundary / VALENCY
        + params.depth_penalty * depth_span
        + params.type_penalty * len(edge_types)
        + params.diameter_penalty * diameter
    )
    return PolymerStats(
        size=len(polymer),
        diameter=diameter,
        boundary_edges=boundary,
        depth_span=depth_span,
        distinct_edge_types=len(edge_types),
        action=action,
    )


def enumerate_polymers(exact_diameter: int) -> list[frozenset[Plaquette]]:
    """Enumerate minimal connected polymers containing E0 with exact diameter."""
    if exact_diameter not in (1, 2):
        raise ValueError("this diagnostic enumerates exact diameters 1 and 2")

    vertices = sorted(ball(E0, exact_diameter))
    distances = pairwise_distances(vertices)
    polymers: list[frozenset[Plaquette]] = []

    if exact_diameter == 1:
        for v in vertices:
            if v == E0:
                continue
            polymer = frozenset((E0, v))
            if is_connected(polymer) and diameter_of(polymer, distances) == 1:
                polymers.append(polymer)
        return polymers

    others = [v for v in vertices if v != E0]
    for i, a in enumerate(others):
        for b in others[i + 1 :]:
            polymer = frozenset((E0, a, b))
            if not is_connected(polymer):
                continue
            if diameter_of(polymer, distances) == exact_diameter:
                polymers.append(polymer)
    return polymers


def aggregate_rows(params: ActionParameters) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for diameter in (1, 2):
        polymers = enumerate_polymers(diameter)
        stats = [carrier_action(polymer, diameter, params) for polymer in polymers]
        for beta in BETAS:
            z_values = [math.exp(-beta * stat.action) - 1.0 for stat in stats]
            z_sum = sum(z_values)
            weighted_sum = sum(abs(z) * math.exp(A_WEIGHT * stat.size) for z, stat in zip(z_values, stats))
            kp_ratio = weighted_sum
            rows.append(
                {
                    "diameter": str(diameter),
                    "count": str(len(polymers)),
                    "z_sum": f"{z_sum:.12g}",
                    "weighted_sum": f"{weighted_sum:.12g}",
                    "beta": f"{beta:.12g}",
                    "kp_ratio": f"{kp_ratio:.12g}",
                    "promotion_status": PROMOTION_STATUS,
                }
            )
    return rows


def write_csv(out_dir: Path, rows: list[dict[str, str]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ym_p7_polymer_enumerator.csv"
    fieldnames = ["diameter", "count", "z_sum", "weighted_sum", "beta", "kp_ratio", "promotion_status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    args = parse_args()
    params = ActionParameters()
    rows = aggregate_rows(params)
    out_path = write_csv(args.out_dir, rows)
    print(f"wrote {out_path}")
    print("promotion_status=NO_PROMOTION; kp_ratio<1 is finite diagnostic safe only, not Clay proof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
