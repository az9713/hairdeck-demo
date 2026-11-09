"""Cards and chains of a HairDeck result -> arrays in the avatar's Blender world. NumPy only, so Blender's Python runs it too.

Mesh rule follows hairdeck/cards/mesh.py and hairdeck/infer/glb.py (4 vertices per control point, Bishop frames), with the
hidden inner caps left out: only the root cap and the tip cap stay.
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('hd_sim', os.path.join(HERE, 'HairDeck', 'hairdeck', 'spring', 'sim.py'))
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


def body():
    return json.load(open(os.path.join(HERE, 'assets', 'avatar', 'body.json')))


def to_world(p, b):
    """HairDeck head frame -> Blender world of the body: hairdeck.vrm.prep.head_to_body, then prep.to_blender."""
    q = (np.asarray(p, np.float64) - np.asarray(b['b'])) / b['k']
    f = b['fit']
    a = (q - np.array(f['t'])) / f['s']
    return np.stack([f['sx'] * a[..., 0], f['sz'] * a[..., 2], a[..., 1]], -1)


def scale(b):
    """Metres of the body's Blender world per metre of the head frame."""
    return 1.0 / (b['k'] * b['fit']['s'])


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def tangents(P):
    n = len(P)
    if n < 5:
        return np.repeat(_unit(P[-1] - P[0])[None], n, 0)
    st = np.array([-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12])
    T = np.empty_like(P)
    for i in range(n):
        lo = min(max(i - 2, 0), n - 5)
        T[i] = _unit((st[:, None] * P[lo:lo + 5]).sum(0))
    return T


def frames(P, n0):
    """Width directions along the card: Bishop transport of the root direction n0 (hairdeck/cards/frames.py)."""
    T = tangents(P)
    N = np.empty_like(P)
    N[0] = _unit(n0 - (n0 @ T[0]) * T[0])
    for i in range(1, len(P)):
        ax = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(ax)
        m = N[i - 1]
        if s > 1e-9:
            k = ax / s
            c = np.clip(T[i - 1] @ T[i], -1, 1)
            m = m * c + np.cross(k, m) * s + k * (k @ m) * (1 - c)
        N[i] = _unit(m - (m @ T[i]) * T[i])
    return T, N


def card_verts(seq, shape):
    """(n, 8) card in world metres -> (4n, 3) ring vertices."""
    P, w, th = seq[:, :3], seq[:, 3:4], seq[:, 4:5]
    T, N = frames(P, seq[0, 5:8])
    g = _unit(T - (T * N).sum(-1, keepdims=True) * N)
    B = np.cross(N, g)
    up, dn = (3.0, 6.0) if shape == 'triangle' else (2.0, 2.0)
    return np.stack([P + w / 2 * N, P + th / up * B, P - w / 2 * N, P - th / dn * B], 1).reshape(-1, 3)


def card_faces(n):
    """Triangles of one card with n rings: side walls, root cap, tip cap. No inner caps."""
    f = [[0, 2, 1], [0, 3, 2]]
    for i in range(1, n):
        c, p = 4 * i, 4 * i - 4
        f += [[c + k, p + k, c + (k + 1) % 4] for k in range(4)]
        f += [[p + (k + 1) % 4, c + (k + 1) % 4, p + k] for k in (0, 3, 2, 1)]
    c = 4 * (n - 1)
    f += [[c, c + 1, c + 2], [c, c + 2, c + 3]]
    return np.array(f, np.int64)


def card_colors(seq, colors, k=1.0):
    """(4n, 3) scene-linear vertex colours. colors (B, 3) sRGB sit at the centres of B equal parts of the card's arc length
    (hairdeck/cards/mesh.py card_vertex_colors). k is the per-card brightness that keeps neighbouring cards visible."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seq[:, :3], axis=0), axis=1))]
    v = d / max(d[-1], 1e-9)
    x = (np.arange(len(colors)) + 0.5) / len(colors)
    c = np.clip(np.stack([np.interp(v, x, colors[:, i]) for i in range(3)], -1), 0, 1)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) * k
    return np.repeat(c, 4, axis=0)


# most forward y of the posed shirt per height, |x| < 0.11, measured by blender/t_chest.py
CHEST_Z = np.array([1.01, 1.19, 1.22, 1.25, 1.28, 1.31, 1.34])
CHEST_Y = np.array([-0.132, -0.131, -0.103, -0.080, -0.059, -0.039, -0.011])


def on_chest(p, gap=0.012):
    """Move points that float in front of the chest back to gap metres from it. The image gives no depth, so the front
    locks of a long style stand out straight from the neck: their tips were 4.6 cm from the shirt (long, seed 1)."""
    q = np.array(p, np.float64)
    below = q[..., 2] < CHEST_Z[-1]
    q[..., 1] = np.where(below, np.maximum(q[..., 1], np.interp(q[..., 2], CHEST_Z, CHEST_Y) - gap), q[..., 1])
    return q


def load(folder, seed=0, b=None, chest=False):
    """-> cards: list of dict(seq (n, 8) world, shape, colors (16, 3) sRGB), chains: list of dict(J, stiff, drag, grav).
    chest=True applies on_chest to cards and chains, so both keep the same rest pose."""
    b = b or body()
    s = scale(b)
    cards = []
    # the width direction turns with the frame change: same axis swap as positions, without the translation
    f = b['fit']
    for c in json.load(open(os.path.join(folder, f'cards_seed{seed}.json'))):
        q = np.asarray(c['seq'], np.float64)
        if q.ndim != 2 or len(q) < 3 or not np.isfinite(q).all():
            continue
        out = q.copy()
        out[:, :3] = to_world(q[:, :3], b)
        if chest:
            out[:, :3] = on_chest(out[:, :3])
        out[:, 3:5] = q[:, 3:5] * s
        out[:, 5:8] = np.stack([f['sx'] * q[:, 5], f['sz'] * q[:, 7], q[:, 6]], -1)
        cards.append(dict(seq=out, shape=c['shape'], colors=np.asarray(c.get('colors', [[0.3, 0.2, 0.15]]), np.float64)))
    chains = []
    p = os.path.join(folder, f'chains_seed{seed}.json')
    if os.path.exists(p):
        for c in json.load(open(p)):
            J = to_world(np.asarray(c['joints'], np.float64), b)
            J = sim.clean(on_chest(J) if chest else J)
            if len(J) >= 2:
                chains.append(dict(J=J, stiff=float(c['stiffness']), drag=0.4, grav=np.zeros(3)))
    return cards, chains


if __name__ == '__main__':  # self-check: faces of a 5-ring card, 8 side triangles per gap, root cap and tip cap
    assert len(card_faces(5)) == 2 + 8 * 4 + 2
    assert card_faces(5).max() == 19
    # on_chest: a floating point moves back, a point on the back or above the shirt stays
    assert abs(on_chest([0.08, -0.177, 1.18])[1] - (-0.143)) < 0.002
    assert on_chest([0.0, 0.2, 1.1])[1] == 0.2 and on_chest([0.0, -0.09, 1.45])[1] == -0.09
    print('ok')
