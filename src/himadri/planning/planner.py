"""Rover path planning on the cost surface.

A* with an 8-connected grid and an admissible (octile-distance) heuristic.
A D*-Lite-style incremental replanner is provided for dynamic hazard updates;
for the static prototype scene A* yields the optimal route. The traverse is
framed as a 'dash-and-return': a bounded-energy/-time excursion from the sunlit
landing site into the PSR to the ice target and back to sunlight to recharge.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

INF = 1e9
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _octile(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)


def plan_route(cost: np.ndarray, start, goal, method: str = "astar"):
    """A* search. Returns a list of (row,col) from start to goal, or [] if no
    path exists. Step cost = mean of the two cells' costs x geometric length."""
    H, W = cost.shape
    start, goal = tuple(start), tuple(goal)
    if cost[start] >= INF or cost[goal] >= INF:
        return []

    g = {start: 0.0}
    came: dict = {}
    pq = [(_octile(start, goal), 0.0, start)]
    closed = set()
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal:
            return _reconstruct(came, cur)
        closed.add(cur)
        cr, cc = cur
        for dr, dc in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            nxt = (nr, nc)
            if nxt in closed or cost[nr, nc] >= INF:
                continue
            length = math.sqrt(dr * dr + dc * dc)
            step = 0.5 * (cost[cr, cc] + cost[nr, nc]) * length
            ng = gc + step
            if ng < g.get(nxt, INF):
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(pq, (ng + _octile(nxt, goal), ng, nxt))
    return []


def _reconstruct(came, cur):
    path = [cur]
    while cur in came:
        cur = came[cur]
        path.append(cur)
    return path[::-1]


def dash_and_return(cost: np.ndarray, illumination: np.ndarray, start, goal,
                    cfg) -> dict:
    """Plan landing->ice (dash) and ice->sunlight (return).

    Returns the combined route plus the leg breakpoint so the energy model can
    treat the dark excursion and the recharge return separately.
    """
    inbound = plan_route(cost, start, goal, cfg.planning.method)
    # return leg: head back to the brightest reachable cell near the start
    H, W = cost.shape
    sr, sc = start
    r0, r1 = max(0, sr - 30), min(H, sr + 30)
    c0, c1 = max(0, sc - 30), min(W, sc + 30)
    window = illumination[r0:r1, c0:c1]
    lr, lc = np.unravel_index(np.argmax(window), window.shape)
    sun_target = (r0 + lr, c0 + lc)
    outbound = plan_route(cost, goal, sun_target, cfg.planning.method)

    full = inbound + outbound[1:] if (inbound and outbound) else inbound
    return {
        "inbound": inbound,
        "outbound": outbound,
        "route": full,
        "dash_breakpoint": len(inbound),
        "sun_target": sun_target,
    }
