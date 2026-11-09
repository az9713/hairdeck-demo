"""Regrowth: every frame is a pure function of its index, so a stopped render resumes by skipping existing files.

blender -b --factory-startup -P film.py -- stills 96,288,500 [WIDTH]   -> out/tests/film/fNNNNN.png
blender -b --factory-startup -P film.py -- render START END [WIDTH]    -> out/frames/fNNNNN.png, existing files skipped
Delete out/sim/*.npz after a change to head(), wind() or the colliders.
"""
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'blender')]
import rig
import scene

FPS, DUR = 24, 90
N = FPS * DUR
PIV = np.array([0.0, 0.0247, 1.3863])  # head joint
sim = rig.sim
CUT_Z = 1.365  # chin height: the Face mesh ends at z = 1.352
LOSS = (37.0, 48.5)
GROW = (63.0, 77.0)


def ss(a, b, t):
    x = min(max((t - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


# --- performance ---------------------------------------------------------------------------------------------------
def head(t):
    """World rotation of the head at time t. She faces -Y. Yaw < 0 looks to screen left, pitch > 0 looks down."""
    yaw = -38 * ss(8.5, 11, t) * (1 - ss(14, 15.8, t)) - 30 * ss(80, 82.5, t) * (1 - ss(86, 88.5, t))
    pitch = 20 * ss(19.5, 21.5, t) * (1 - ss(25.2, 26, t)) + 6 * ss(26, 28, t) * (1 - ss(60, 70, t)) + 8 * ss(38, 48, t) * (1 - ss(52, 60, t))
    yaw += 1.2 * math.sin(.5 * t)
    pitch += .6 * math.sin(.37 * t + 1)
    roll = .8 * math.sin(.29 * t)
    a = Matrix.Rotation(math.radians(yaw), 3, 'Z') @ Matrix.Rotation(math.radians(pitch), 3, 'X') @ Matrix.Rotation(math.radians(roll), 3, 'Y')
    return np.array(a)


WIND_DIR = np.array([.9, .35, .05])  # toward screen right and behind her, so the hair does not cross her face


def wind(t):
    """Force amount: gusts in spring, calm in the middle, none in winter."""
    a = .3 * (ss(0, 2, t) * (1 - ss(14, 17, t)) + ss(78, 80, t)) + .1 * ss(15, 17, t) * (1 - ss(34, 36, t)) + .16 * ss(35, 38, t) * (1 - ss(47, 50, t)) + .06 * ss(62, 66, t) * (1 - ss(78, 80, t))
    return a * (.65 + .35 * math.sin(.9 * t) * math.sin(.37 * t + 2))


BLINKS = [3.5, 7.2, 12.4, 18.2, 23.1, 39.3, 44.0, 47.2, 64.5, 71.3, 76.4, 80.2, 86.8]


def face(t):
    """Shape key values."""
    close = max([0.0] + [1 - abs(t - b) / .09 for b in BLINKS])
    close = max(close, ss(55.5, 56.2, t) * (1 - ss(56.9, 57.6, t)))  # the slow blink in winter
    joy = ss(81, 83.5, t)
    sad = ss(20, 23, t) * (1 - ss(60, 70, t))
    return {'Fcl_EYE_Close': close * (1 - joy * .0), 'Fcl_BRW_Sorrow': .75 * sad, 'Fcl_MTH_Sorrow': .35 * sad,
            'Fcl_MTH_Fun': .8 * joy, 'Fcl_BRW_Joy': .6 * joy}


# start, end, azimuth start/end, distance start/end
SHOTS = [(0, 16, 22, 28, 2.15, 1.75), (16, 26, -22, -18, 1.35, 1.2), (26, 36, 198, 190, 1.35, 1.2), (36, 50, 14, 20, 1.25, 1.05),
         (50, 62, 8, 4, .92, .8), (62, 78, 4, 18, .8, 1.15), (78, 90, -24, -14, 1.55, 1.35)]


def camera(t):
    for t0, t1, a0, a1, d0, d1 in SHOTS:
        if t < t1 or t1 == DUR:
            x = (t - t0) / (t1 - t0)
            x = x * x * (3 - 2 * x) * .3 + x * .7
            d = d0 + (d1 - d0) * x
            return a0 + (a1 - a0) * x, d, 1.63 - .16875 * d  # target height keeps the top of the head near the frame top


# plate weights: spring -> summer -> autumn -> winter -> thaw -> spring
PLATES = ['spring', 'summer', 'autumn', 'winter', 'thaw']
FADES = [(14, 18), (34.5, 37.5), (47, 52), (60, 66), (75, 80)]


# --- simulation -----------------------------------------------------------------------------------------------------
def colliders(t):
    """(centre, radius): skull, chest, shoulders. Only the skull turns with the head."""
    R = head(t)
    return [(R @ (np.array([0, .012, 1.47]) - PIV) + PIV, .10), (np.array([0, -.005, 1.19]), .125), (np.array([.135, .01, 1.27]), .06), (np.array([-.135, .01, 1.27]), .06)]


def simulate(chains, t0, t1, path):
    """Copy of sim.simulate with a wind force and sphere colliders. A joint that starts inside a sphere ignores that sphere."""
    if os.path.exists(path):
        z = np.load(path)
        return [(z[f'h{i}'], z[f'r{i}']) for i in range(len(chains))]
    dt = 1 / 72  # 3 steps per film frame; sim.py uses 60 Hz
    f0, f1 = int(t0 * FPS), int(t1 * FPS)
    rng = np.random.default_rng(5)
    st = []
    C0 = colliders(t0)
    for c in chains:
        J = c['J']
        d = np.diff(J, axis=0)
        L = np.linalg.norm(d, axis=1)
        mask = np.array([[np.linalg.norm(p - s) > r for s, r in C0] for p in J[1:]])
        st.append(dict(L=L, D=d / L[:, None], cur=J[1:].copy(), prev=J[1:].copy(), mask=mask, ph=rng.uniform(0, 6.28),
                       h=np.zeros((f1 - f0, len(L), 3)), r=np.zeros((f1 - f0, len(L), 3, 3))))
    for s in range((f1 - f0) * 3):
        t = t0 + s * dt
        R, S, W = head(t), colliders(t), wind(t)
        f, store = s // 3, s % 3 == 0
        for c, q in zip(chains, st):
            Wp, hd = R, R @ (c['J'][0] - PIV) + PIV
            for k in range(len(q['L'])):
                rest = Wp @ q['D'][k]
                w = WIND_DIR * W * (1 + .45 * math.sin(2.3 * t + q['ph'] + k * .8))
                nxt = q['cur'][k] + (q['cur'][k] - q['prev'][k]) * (1 - c['drag']) + rest * (c['stiff'] * dt) + w * dt
                for i, (ce, ra) in enumerate(S):
                    v = nxt - ce
                    n = np.linalg.norm(v)
                    if q['mask'][k, i] and n < ra:
                        nxt = ce + v / max(n, 1e-9) * ra
                v = nxt - hd
                n = np.linalg.norm(v)
                v = v / n if n > 1e-12 else rest
                nxt = hd + v * q['L'][k]
                q['prev'][k], q['cur'][k] = q['cur'][k], nxt
                Wk = sim.rotdiff(rest, v) @ Wp
                if store:
                    q['h'][f, k], q['r'][f, k] = hd, Wk
                Wp, hd = Wk, nxt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f'h{i}': q['h'] for i, q in enumerate(st)}, **{f'r{i}': q['r'] for i, q in enumerate(st)})
    return [(q['h'], q['r']) for q in st]


# --- card animation -------------------------------------------------------------------------------------------------
def trim(X, s, f, tip):
    """One card: X (4n, 3) deformed rings, s (n,) arc fraction of each ring, f visible fraction, tip taper length as a fraction.
    Rings beyond f collapse onto the point at f. Topology stays fixed. Rings near f narrow toward that point."""
    n = len(s)
    Y = X.reshape(n, 4, 3).copy()
    if f <= 0:
        Y[:] = Y[0].mean(0)
        return Y.reshape(-1, 3)
    C = Y.mean(1)
    e = np.stack([np.interp(f, s, C[:, i]) for i in range(3)])
    w = np.clip((f - s) / max(tip, 1e-4), 0, 1)[:, None, None]
    Y = C[:, None] + (Y - C[:, None]) * w
    Y[s >= f] = e
    return Y.reshape(-1, 3)


class Style:
    def __init__(self, d, name, seed, chest, t0, t1):
        folder = os.path.join(ROOT, 'assets', 'final', name)
        self.cards, self.chains = rig.load(folder, seed, chest=chest)
        self.V, self.s, self.off, self.len = [], [], [0], []
        for c in self.cards:
            v = rig.card_verts(c['seq'], c['shape'])
            p = c['seq'][:, :3]
            a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            self.V.append(v)
            self.s.append(a / max(a[-1], 1e-9))
            self.len.append(a[-1])
            self.off.append(self.off[-1] + len(v))
        self.bind = sim.bind(self.V, self.chains, .17)  # W: width of the avatar's face in metres
        self.V0 = np.concatenate(self.V)
        self.f0 = int(t0 * FPS)
        self.sim = simulate(self.chains, t0, t1, os.path.join(ROOT, 'out', 'sim', name + '.npz'))
        self.nf = int(t1 * FPS) - self.f0
        self.obj = scene.add_hair(d, self.cards, 'hair_' + name)
        self.fall = None

    def deform(self, f):
        i = min(max(f - self.f0, 0), self.nf - 1)
        s = [(h[i:i + 1], r[i:i + 1]) for h, r in self.sim]
        R = head(f / FPS)[None]
        return sim.deform(self.V0, self.bind, self.chains, s, R, np.zeros((1, 3)), PIV)[0]

    def part(self, X, j):
        return X[self.off[j]:self.off[j + 1]]


def falling(S, c0, tau, seed):
    """A piece S (m, 3) tau seconds after it detached: terminal-velocity fall, outward drift, sway, tumble."""
    g = np.random.default_rng(seed)
    vt, om, ph = g.uniform(.5, .8), g.uniform(.2, .6), g.uniform(0, 6.28)
    ax = g.normal(size=3)
    ax /= np.linalg.norm(ax)
    drop = vt * (tau - .3 * (1 - math.exp(-tau / .3)))
    rad = c0[:2] / max(np.linalg.norm(c0[:2]), .05)
    gx = (1 - math.exp(-tau / .5))
    off = np.array([rad[0] * .08 * gx + .035 * math.sin(1.6 * tau + ph) + .02 * tau, rad[1] * .08 * gx - .01 * tau, -drop])
    R = np.array(Matrix.Rotation(om * (tau - .5 * gx), 3, Vector(ax)))
    return (S - c0) @ R.T + c0 + off


class Long(Style):
    def __init__(self, d):
        super().__init__(d, 'long', 1, True, 0, 50.5)
        n = len(self.cards)
        g = np.random.default_rng(7)
        self.cut_t, self.cut_r, self.cut_f = np.full(n, 1e9), np.zeros(n, int), np.ones(n)
        tips = np.array([c['seq'][-1, :3] for c in self.cards])
        # the first cards to fall: long cards in front and at the sides, visible from the front in shot B
        early = [j for j in np.argsort(tips[:, 1]) if self.len[j] > .3][:40]
        early = list(g.choice(early, 7, replace=False))
        self.loss = np.zeros(n)
        rest = [j for j in range(n) if j not in early]
        # front hair first, so her face stays clear while the rest falls
        score = np.array([g.uniform(0, 1) - 1.5 * (tips[j, 1] < -.02) for j in rest])
        for k, i in enumerate(np.argsort(score)):
            self.loss[rest[i]] = LOSS[0] + (LOSS[1] - LOSS[0]) * (k / max(len(rest) - 1, 1)) ** .85
        for k, j in enumerate(early):
            self.loss[j] = 17.2 + k * 1.15
        # the cut moves across the back from screen left to right, 28 s to 31.5 s
        for j, c in enumerate(self.cards):
            if self.loss[j] < 28:
                continue
            p = c['seq'][:, :3]
            z = CUT_Z + g.uniform(-.006, .006)
            below = np.nonzero(p[:, 2] < z)[0]
            if len(below) and below[0] >= 2:
                r = below[0]
                self.cut_r[j], self.cut_f[j] = r, self.s[j][r]
                self.cut_t[j] = 28 + 3.5 * (.5 + p[r, 0] / .22) + g.uniform(-.12, .12)
        self.cut_t = np.where(self.cut_t < 1e8, np.clip(self.cut_t, 28, 31.8), self.cut_t)
        drop = {j: int(self.cut_r[j]) for j in range(n) if self.cut_t[j] < 1e8}
        self.fall = scene.add_hair(d, self.cards, 'fall_long', drop)
        self.snap = {}
        for t in set(self.cut_t[self.cut_t < 1e8]) | set(self.loss):
            f = int(t * FPS)
            if f not in self.snap:
                self.snap[f] = self.deform(f)
        print('LONG cards', n, 'cuts', len(drop), 'early', early)

    def update(self, f):
        t = f / FPS
        X = self.deform(f)
        Y, Z = X.copy(), np.tile(np.array([0, 0, -5.0]), (len(X), 1))  # unused fall vertices wait below the floor
        for j in range(len(self.cards)):
            a, b = self.off[j], self.off[j + 1]
            s = self.s[j]
            lost, cut = t >= self.loss[j], t >= self.cut_t[j]
            if lost:
                Y[a:b] = trim(Y[a:b], s, 0.0, 1)
            elif cut:
                Y[a:b] = trim(Y[a:b], s, self.cut_f[j], .02)
            n = len(s)
            r = self.cut_r[j] if self.cut_t[j] < 1e8 else n
            if cut:  # the tip piece: rings r..n-1
                S = self.snap[int(self.cut_t[j] * FPS)][a:b][r * 4:]
                tau = t - self.cut_t[j]
                if tau < 6:
                    Z[a + r * 4:b] = falling(S, S.mean(0), tau, 1000 + j)
            if lost:  # the root piece: rings 0..r-1
                S = self.snap[int(self.loss[j] * FPS)][a:b][:r * 4]
                tau = t - self.loss[j]
                if tau < 6:
                    Z[a:a + r * 4] = falling(S, S.mean(0), tau, 2000 + j)
        return Y, Z


class Curly(Style):
    def __init__(self, d):
        super().__init__(d, 'curly', 3, False, 62, 90)
        g = np.random.default_rng(11)
        self.start = GROW[0] + 1.5 + g.uniform(0, 3, len(self.cards))
        self.span = g.uniform(GROW[1] - GROW[0] - 6.5, GROW[1] - GROW[0] - 4.5, len(self.cards))

    def update(self, f):
        t = f / FPS
        X = self.deform(f)
        for j in range(len(self.cards)):
            x = min(max((t - self.start[j]) / self.span[j], 0), 1)
            if x >= 1:
                continue
            a, b = self.off[j], self.off[j + 1]
            x = x * x * (3 - 2 * x)
            # visible length L(t) in metres -> fraction. Short early hair stays narrow, like stubble.
            X[a:b] = trim(X[a:b], self.s[j], x, .05 / self.len[j])
        return X, None


# --- world ---------------------------------------------------------------------------------------------------------
def set_verts(obj, X):
    obj.data.vertices.foreach_set('co', X.ravel())
    obj.data.update()


def make_plate(sc):
    """A camera-facing plane behind her. Five textures in a chain of mixes, driven by value nodes."""
    m = bpy.data.materials.new('plate')
    m.use_nodes = True
    t = m.node_tree
    t.nodes.clear()
    em, out = t.nodes.new('ShaderNodeEmission'), t.nodes.new('ShaderNodeOutputMaterial')
    tex = []
    for p in PLATES:
        n = t.nodes.new('ShaderNodeTexImage')
        n.image = bpy.data.images.load(os.path.join(ROOT, 'assets', 'plates', p + '.png'))
        n.extension = 'EXTEND'
        tex.append(n)
    sock, vals = tex[0].outputs[0], []
    for src in tex[1:] + [tex[0]]:
        mx, v = t.nodes.new('ShaderNodeMix'), t.nodes.new('ShaderNodeValue')
        mx.data_type = 'RGBA'
        t.links.new(v.outputs[0], mx.inputs[0])
        t.links.new(sock, mx.inputs['A'])
        t.links.new(src.outputs[0], mx.inputs['B'])
        sock = mx.outputs['Result']
        vals.append(v)
    t.links.new(sock, em.inputs[0])
    t.links.new(em.outputs[0], out.inputs[0])
    me = bpy.data.meshes.new('plate')
    me.from_pydata([(-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0)], [], [(0, 1, 2, 3)])
    me.uv_layers.new().data.foreach_set('uv', [0, 0, 1, 0, 1, 1, 0, 1])
    me.materials.append(m)
    o = bpy.data.objects.new('plate', me)
    sc.collection.objects.link(o)
    return o, vals


class Particles:
    """Petals, leaves and snow in camera space: position is a function of time, no state. Colour per group is fixed."""
    GROUPS = [('petal', 70, (.005, .008), ((1, .78, .84), (1, .88, .9)), lambda t: ss(0, 2, t) * (1 - ss(13, 17, t)) + ss(78, 81, t), .1),
              ('leaf', 90, (.008, .014), ((.86, .36, .06), (.78, .2, .05), (.92, .58, .1)), lambda t: ss(35.5, 38, t) * (1 - ss(49, 52, t)), .16),
              ('snow', 170, (.002, .004), ((1, 1, 1), (.92, .96, 1)), lambda t: ss(48, 52, t) * (1 - ss(61, 65, t)), .09)]

    def __init__(self, sc):
        g = np.random.default_rng(3)
        self.p = []
        V, F, C, k = [], [], [], 0
        for name, n, sz, cols, env, speed in self.GROUPS:
            self.p.append(dict(x=g.uniform(0, 1, n), y=g.uniform(0, 1, n), d=g.uniform(.25, 1.5, n), size=g.uniform(*sz, n), v=speed * g.uniform(.7, 1.3, n),
                               ph=g.uniform(0, 6.28, n), om=g.uniform(.6, 2.2, n), env=env, name=name))
            for i in range(n):
                F.append((k, k + 1, k + 2, k + 3))
                C += [cols[g.integers(len(cols))]] * 4
                k += 4
        V = np.zeros((k, 3))
        me = bpy.data.meshes.new('particles')
        me.from_pydata(V.tolist(), [], F)
        a = me.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
        C = np.array(C) ** 2.2
        a.data.foreach_set('color', np.c_[C, np.ones(len(C))].ravel())
        m = bpy.data.materials.new('particles')
        m.use_nodes = True
        t = m.node_tree
        t.nodes.clear()
        col, em, out = t.nodes.new('ShaderNodeVertexColor'), t.nodes.new('ShaderNodeEmission'), t.nodes.new('ShaderNodeOutputMaterial')
        col.layer_name = 'col'
        t.links.new(col.outputs[0], em.inputs[0])
        t.links.new(em.outputs[0], out.inputs[0])
        me.materials.append(m)
        self.obj = bpy.data.objects.new('particles', me)
        sc.collection.objects.link(self.obj)

    def update(self, t, cam, dist):
        M = np.array(cam.matrix_world)
        out = []
        for q in self.p:
            e = q['env'](t)
            # active share: particles with a low phase rank appear first
            on = (q['ph'] / 6.28) < e
            depth = dist + q['d']
            hw, hh = depth * .3 * 1.1, depth * .16875 * 1.15
            y = 1 - ((q['y'] + q['v'] * t * 2.2 / (2 * hh)) % 1)
            x = (q['x'] + .012 * t) % 1 + .02 * np.sin(q['om'] * t * .6 + q['ph'])
            c = np.stack([(x * 2 - 1) * hw, (y * 2 - 1) * hh, -depth], 1)
            ang = q['om'] * t + q['ph']
            fl = np.abs(np.cos(q['om'] * t * .7 + q['ph'])) * .8 + .2
            sz = q['size'] * on
            long_ = {'leaf': 1.8, 'petal': 1.4, 'snow': 1.0}[q['name']]
            u = np.stack([np.cos(ang), np.sin(ang)], 1) * (sz * long_)[:, None]
            v = np.stack([-np.sin(ang), np.cos(ang)], 1) * (sz * fl)[:, None]
            quad = np.stack([c.copy() for _ in range(4)], 1)
            sgn = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # a pointed shape, not a square
            for i, (a, b) in enumerate(sgn):
                quad[:, i, :2] += a * u + b * v
            out.append(quad.reshape(-1, 3))
        P = np.concatenate(out)
        W = P @ M[:3, :3].T + M[:3, 3]
        set_verts(self.obj, W)


class Film:
    def __init__(self, res):
        self.d = d = scene.build(res=res)
        sc = self.sc = bpy.context.scene
        sc.eevee.taa_render_samples = 16
        self.long, self.curly = Long(d), Curly(d)
        self.plate, self.vals = make_plate(sc)
        self.part = Particles(sc)
        arm = d['arm']
        self.pb = arm.pose.bones[scene.HEAD_BONE]
        self.base = self.pb.matrix.copy()
        self.mw = arm.matrix_world.copy()
        self.keys = bpy.data.objects['Face'].data.shape_keys.key_blocks
        self.cam = d['cam']

    def set(self, f):
        t = f / FPS
        R = Matrix(head(t).tolist()).to_4x4()
        P = Vector(PIV.tolist())
        self.pb.matrix = self.mw.inverted() @ (Matrix.Translation(P) @ R @ Matrix.Translation(-P)) @ self.mw @ self.base
        for k, v in face(t).items():
            self.keys[k].value = v
        az, dist, z = camera(t)
        scene.aim(self.cam, (PIV[0], PIV[1], z), az, dist)
        a, e = math.radians(az + 18), math.radians(30)  # key light stays on the camera side, also in the back view
        self.d['sun'].rotation_euler = Vector((-math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), -math.sin(e))).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        M = self.cam.matrix_world
        D = dist + 2.6  # plate distance from the camera
        self.plate.matrix_world = M @ Matrix.Translation((0, 0, -D)) @ Matrix.Diagonal((1.2 * 2 * D * .3, 1.2 * 2 * D * .16875, 1, 1))  # 1.2: the wall bands above and below the window stay out of frame
        for v, (a, b) in zip(self.vals, FADES):
            v.outputs[0].default_value = ss(a, b, t)
        self.part.update(t, self.cam, dist)
        self.d['cap'].default_value = (1 - ss(36.5, 41, t)) if t < 55 else ss(63, 71, t)  # the cap returns first and reads as short new hair
        lo, cu = t < 50.4, t >= 62.5
        self.long.obj.hide_render = not lo
        self.long.fall.hide_render = not (16.9 < t < 55)
        self.curly.obj.hide_render = not cu
        if lo or t < 55:
            Y, Z = self.long.update(min(f, int(50.3 * FPS)))
            if lo:
                set_verts(self.long.obj, Y)
            set_verts(self.long.fall, Z)
        if cu:
            set_verts(self.curly.obj, self.curly.update(f)[0])

    def render(self, f, path):
        self.set(f)
        self.sc.render.filepath = path
        bpy.ops.render.render(write_still=True)


def main():
    a = sys.argv[sys.argv.index('--') + 1:]
    if a[0] == 'stills':
        W = int(a[2]) if len(a) > 2 else 960
        film = Film((W, W * 9 // 16))
        os.makedirs(os.path.join(ROOT, 'out', 'tests', 'film'), exist_ok=True)
        for f in [int(x) for x in a[1].split(',')]:
            t = time.time()
            film.render(f, os.path.join(ROOT, 'out', 'tests', 'film', f'f{f:05d}.png'))
            print('FRAME', f, round(time.time() - t, 2), flush=True)
    else:
        W = int(a[3]) if len(a) > 3 else 1920
        film = Film((W, W * 9 // 16))
        out = os.path.join(ROOT, 'out', 'frames')
        os.makedirs(out, exist_ok=True)
        for f in range(int(a[1]), min(int(a[2]), N)):
            p = os.path.join(out, f'f{f:05d}.png')
            if os.path.exists(p) and os.path.getsize(p) > 0:
                continue
            t = time.time()
            film.render(f, p + '.tmp.png')
            os.replace(p + '.tmp.png', p)
            print('FRAME', f, round(time.time() - t, 2), flush=True)
    print('DONE')


main()
