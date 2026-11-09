"""Shared Blender scene of Regrowth: the avatar without its hair, arms down, toon materials, outline, camera, sun, backdrop."""
import math
import os
import shutil

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD_BONE = 'J_Bip_C_Head'


def toon_group():
    """Lit colour = texture; shade colour = texture x shade tint. Sharp two-step ramp on a diffuse closure."""
    g = bpy.data.node_groups.new('Toon', 'ShaderNodeTree')
    g.interface.new_socket('Color', in_out='INPUT', socket_type='NodeSocketColor')
    g.interface.new_socket('Shade', in_out='INPUT', socket_type='NodeSocketColor')
    g.interface.new_socket('Alpha', in_out='INPUT', socket_type='NodeSocketFloat')
    g.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    n, l = g.nodes, g.links
    i, o = n.new('NodeGroupInput'), n.new('NodeGroupOutput')
    dif, s2r, ramp = n.new('ShaderNodeBsdfDiffuse'), n.new('ShaderNodeShaderToRGB'), n.new('ShaderNodeValToRGB')
    ramp.color_ramp.interpolation = 'CONSTANT'
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].position = 0.22
    bw, mix = n.new('ShaderNodeRGBToBW'), n.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    em, tr, ms = n.new('ShaderNodeEmission'), n.new('ShaderNodeBsdfTransparent'), n.new('ShaderNodeMixShader')
    l.new(dif.outputs[0], s2r.inputs[0])
    l.new(s2r.outputs[0], bw.inputs[0])
    l.new(bw.outputs[0], ramp.inputs[0])
    l.new(ramp.outputs[0], mix.inputs[0])
    l.new(i.outputs['Shade'], mix.inputs['A'])
    l.new(i.outputs['Color'], mix.inputs['B'])
    l.new(mix.outputs['Result'], em.inputs[0])
    l.new(i.outputs['Alpha'], ms.inputs[0])
    l.new(tr.outputs[0], ms.inputs[1])
    l.new(em.outputs[0], ms.inputs[2])
    l.new(ms.outputs[0], o.inputs[0])
    return g


def tint(socket, rgb, tree):
    m = tree.nodes.new('ShaderNodeMix')
    m.data_type = 'RGBA'
    m.blend_type = 'MULTIPLY'
    m.inputs[0].default_value = 1.0
    tree.links.new(socket, m.inputs['A'])
    m.inputs['B'].default_value = (*rgb, 1)
    return m.outputs['Result']


def forehead_skin():
    """Scene-linear colour of the Face texture at the top 10% of the skin vertices, where the scalp meets the forehead.
    Image.pixels holds the file's sRGB values, and a Mix node input takes linear values, so the colour is converted."""
    obj = next(o for o in bpy.data.objects if o.type == 'MESH' and any('Face_00_SKIN' in m.name for m in o.data.materials if m))
    mesh = obj.data
    idx = next(i for i, m in enumerate(mesh.materials) if m and 'Face_00_SKIN' in m.name)
    img = next(x for x in mesh.materials[idx].node_tree.nodes if x.type == 'TEX_IMAGE').image
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    rgb = px.reshape(h, w, 4)[..., :3]
    uv = mesh.uv_layers[0].data
    loops = [(mesh.vertices[mesh.loops[l].vertex_index].co.z, uv[l].uv) for p in mesh.polygons if p.material_index == idx for l in p.loop_indices]
    top = np.quantile([z for z, _ in loops], 0.9)
    s = np.array([rgb[int(q.y % 1 * h) % h, int(q.x % 1 * w) % w] for z, q in loops if z >= top])
    c = np.median(s, axis=0)
    print('forehead sRGB', c, 'samples', len(s))
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def scalp_value():
    """The head's painted skullcap uses the HairBack material. A value node fades it to skin colour.
    Returns the value socket (1 = painted cap visible, 0 = bare scalp)."""
    skin = forehead_skin()
    t = next(m for m in bpy.data.materials if 'HairBack' in m.name).node_tree
    em = next(x for x in t.nodes if x.type == 'EMISSION')
    src = em.inputs['Color'].links[0].from_socket
    val = t.nodes.new('ShaderNodeValue')
    val.name = 'cap'
    val.outputs[0].default_value = 0.0
    mix = t.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    t.links.new(val.outputs[0], mix.inputs[0])
    mix.inputs['A'].default_value = (*skin, 1)
    t.links.new(src, mix.inputs['B'])
    t.links.new(mix.outputs['Result'], em.inputs['Color'])
    return val.outputs[0]


def toon_avatar(group):
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        t = mat.node_tree
        em = next((x for x in t.nodes if x.type == 'EMISSION'), None)
        if not em or not em.inputs['Color'].is_linked or not em.outputs[0].is_linked:
            continue
        src = em.inputs['Color'].links[0].from_socket
        gn = t.nodes.new('ShaderNodeGroup')
        gn.node_tree = group
        flat = any(k in mat.name for k in ('Eye', 'Brow', 'FaceMouth'))
        skin = 'SKIN' in mat.name or 'HairBack' in mat.name  # the bare scalp has to shade like the forehead
        shade = (1, 1, 1) if flat else ((1.0, 0.93, 0.91) if skin else (0.78, 0.76, 0.86))
        t.links.new(src, gn.inputs['Color'])
        t.links.new(tint(src, shade, t), gn.inputs['Shade'])
        gn.inputs['Alpha'].default_value = 1.0  # the material's own mix shaders keep handling texture alpha
        target = em.outputs[0].links[0].to_socket
        t.links.new(gn.outputs[0], target)


def outline_material():
    m = bpy.data.materials.new('Outline')
    m.use_nodes = True
    t = m.node_tree
    t.nodes.clear()
    e, o = t.nodes.new('ShaderNodeEmission'), t.nodes.new('ShaderNodeOutputMaterial')
    e.inputs[0].default_value = (0.05, 0.035, 0.05, 1)
    t.links.new(e.outputs[0], o.inputs[0])
    m.use_backface_culling = True
    return m


def add_outline(obj, mat, thick):
    """Inverted hull: a Solidify shell with flipped normals, front faces culled by the material."""
    obj.data.materials.append(mat)
    s = obj.modifiers.new('outline', 'SOLIDIFY')
    s.thickness = thick
    s.offset = 1
    s.use_flip_normals = True
    s.material_offset = len(obj.data.materials) - 1
    s.use_rim = False


def add_hair(d, cards, name='hair', drop=None):
    """All cards as one mesh with vertex colours, toon material and outline. Per-card brightness: 0.78..1, fixed seed (glb.py).
    drop: {card index: ring r} removes the side walls between rings r-1 and r, so the two parts of that card can separate."""
    import rig
    rng = np.random.default_rng(0)
    V, F, C, off = [], [], [], 0
    for j, c in enumerate(cards):
        V.append(rig.card_verts(c['seq'], c['shape']))
        f = rig.card_faces(len(c['seq']))
        if drop and j in drop:  # card_faces order: 2 root cap triangles, then 8 triangles per gap
            a = 2 + 8 * (drop[j] - 1)
            f = np.delete(f, range(a, a + 8), axis=0)
        F.append(f + off)
        C.append(rig.card_colors(c['seq'], c['colors'], float(np.clip(rng.normal(0.93, 0.07), 0.78, 1.0))))
        off += len(V[-1])
    V, F, C = np.concatenate(V), np.concatenate(F), np.concatenate(C)
    m = bpy.data.meshes.new(name)
    m.from_pydata(V.tolist(), [], F.tolist())
    m.update()
    m.polygons.foreach_set('use_smooth', [True] * len(m.polygons))
    a = m.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
    a.data.foreach_set('color', np.c_[C, np.ones(len(C))].ravel())
    mat = bpy.data.materials.get('hair_toon')
    if not mat:
        mat = bpy.data.materials.new('hair_toon')
        mat.use_nodes = True
        t = mat.node_tree
        t.nodes.clear()
        col, gn, o = t.nodes.new('ShaderNodeVertexColor'), t.nodes.new('ShaderNodeGroup'), t.nodes.new('ShaderNodeOutputMaterial')
        col.layer_name = 'col'
        gn.node_tree = d['toon']
        t.links.new(col.outputs[0], gn.inputs['Color'])
        t.links.new(tint(col.outputs[0], (0.55, 0.52, 0.62), t), gn.inputs['Shade'])
        gn.inputs['Alpha'].default_value = 1.0
        t.links.new(gn.outputs[0], o.inputs[0])
    m.materials.append(mat)
    h = bpy.data.objects.new(name, m)
    bpy.context.scene.collection.objects.link(h)
    add_outline(h, d['outline'], 0.0012)
    return h


def outline_with_alpha(obj, base, thick):
    """Outline of a textured mesh. Each material slot gets its own shell material, which cuts the shell with that slot's
    texture alpha. A shell without alpha shows through faces that the texture hides (the dark wedges at the collar)."""
    n = len(obj.data.materials)
    for m in list(obj.data.materials):
        tex = next((x for x in m.node_tree.nodes if x.type == 'TEX_IMAGE' and x.outputs['Alpha'].is_linked), None) if m and m.use_nodes else None
        if not tex:
            obj.data.materials.append(base)
            continue
        c = base.copy()
        c.name = 'Outline_' + m.name
        t = c.node_tree
        im, gt, tr, mx = t.nodes.new('ShaderNodeTexImage'), t.nodes.new('ShaderNodeMath'), t.nodes.new('ShaderNodeBsdfTransparent'), t.nodes.new('ShaderNodeMixShader')
        im.image = tex.image
        gt.operation = 'GREATER_THAN'
        gt.inputs[1].default_value = 0.5
        em = next(x for x in t.nodes if x.type == 'EMISSION')
        out = next(x for x in t.nodes if x.type == 'OUTPUT_MATERIAL')
        t.links.new(im.outputs['Alpha'], gt.inputs[0])
        t.links.new(gt.outputs[0], mx.inputs[0])
        t.links.new(tr.outputs[0], mx.inputs[1])
        t.links.new(em.outputs[0], mx.inputs[2])
        t.links.new(mx.outputs[0], out.inputs[0])
        obj.data.materials.append(c)
    s = obj.modifiers.new('outline', 'SOLIDIFY')
    s.thickness = thick
    s.offset = 1
    s.use_flip_normals = True
    s.material_offset = n
    s.use_rim = False


def world_rotate_bone(arm, name, axis, angle):
    """Pose a bone by a rotation about its head, with the axis given in world space."""
    pb = arm.pose.bones[name]
    mw = arm.matrix_world
    head = mw @ pb.head
    R = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, Vector(axis)) @ Matrix.Translation(-head)
    pb.matrix = mw.inverted() @ R @ mw @ pb.matrix
    bpy.context.view_layer.update()


def build(res=(1920, 1080)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    glb = os.path.join(ROOT, 'assets', 'avatar', 'avatar_copy.glb')
    if not os.path.exists(glb):
        shutil.copy(os.path.join(ROOT, 'assets', 'avatar', 'avatar.vrm'), glb)
    bpy.ops.import_scene.gltf(filepath=glb)
    for o in list(bpy.data.objects):
        if o.type == 'MESH' and (o.name.startswith('Hair') or o.name.startswith('Icosphere')):
            bpy.data.objects.remove(o)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    arm.rotation_mode = 'XYZ'
    arm.rotation_euler[2] = math.pi  # faces -Y, the frame of hairdeck.vrm.prep.to_blender
    bpy.context.view_layer.update()
    world_rotate_bone(arm, 'J_Bip_L_UpperArm', (0, 1, 0), math.radians(68) * side(arm, 'J_Bip_L_UpperArm'))
    world_rotate_bone(arm, 'J_Bip_R_UpperArm', (0, 1, 0), math.radians(68) * side(arm, 'J_Bip_R_UpperArm'))
    print('materials', [m.name for m in bpy.data.materials])
    cap = scalp_value()
    group = toon_group()
    toon_avatar(group)
    outline = outline_material()
    for o in bpy.data.objects:
        if o.type == 'MESH':
            outline_with_alpha(o, outline, 0.0016)
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.render.image_settings.file_format = 'PNG'
    w = bpy.data.worlds.new('w')
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes['Background'].inputs[1].default_value = 0.0
    cam = bpy.data.cameras.new('cam')
    cam.lens = 60
    co = bpy.data.objects.new('cam', cam)
    sc.collection.objects.link(co)
    sc.camera = co
    sd = bpy.data.lights.new('sun', 'SUN')
    sd.energy = 3.0
    sd.angle = math.radians(3)
    sd.use_shadow = False  # the outline shell would shade the whole body; the toon ramp uses the surface angle only
    so = bpy.data.objects.new('sun', sd)
    sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(65), 0, math.radians(35))
    return dict(arm=arm, cam=co, sun=so, toon=group, outline=outline, cap=cap)


def side(arm, name):
    """+1 when the bone lies on the +X side of the world, else -1."""
    return 1.0 if (arm.matrix_world @ arm.pose.bones[name].head).x > 0 else -1.0


def aim(cam, target, azimuth_deg, distance, elev_deg=0.0):
    a, e = math.radians(azimuth_deg), math.radians(elev_deg)
    d = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * distance
    cam.location = Vector(target) + d
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat('-Z', 'Y').to_euler()


def head_pos(arm):
    return arm.matrix_world @ arm.pose.bones[HEAD_BONE].head
