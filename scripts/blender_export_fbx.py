"""
Blender FBX export script for UniRig.

Run with:
    <blender_path> --background --python blender_export_fbx.py -- --params_file <path>

Creates an armature with optional mesh and skinning in Blender, then exports to FBX.
"""

import bpy
from mathutils import Vector
import sys
import os
import json
import numpy as np
from collections import defaultdict


def clean_bpy():
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)


def make_armature(
    vertices,
    joints,
    skin,
    parents,
    names,
    faces=None,
    extrude_size=0.03,
    group_per_vertex=-1,
    add_root=False,
    do_not_normalize=False,
    use_extrude_bone=True,
    use_connect_unique_child=True,
    extrude_from_parent=True,
    tails=None,
):
    # make collection
    collection = bpy.data.collections.new('new_collection')
    bpy.context.scene.collection.children.link(collection)

    # make mesh
    if vertices is not None:
        mesh = bpy.data.meshes.new('mesh')
        if faces is None:
            faces = []
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        # make object from mesh
        object = bpy.data.objects.new('character', mesh)

        # add object to scene collection
        collection.objects.link(object)

    # deselect mesh
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.data.armatures.get('Armature')
    edit_bones = armature.edit_bones

    J = joints.shape[0]
    if tails is None:
        tails = joints.copy()
        tails[:, 2] += extrude_size
    connects = [False for _ in range(J)]
    children = defaultdict(list)
    for i in range(1, J):
        children[parents[i]].append(i)
    if tails is not None:
        if use_extrude_bone:
            for i in range(J):
                if len(children[i]) != 1 and extrude_from_parent and i != 0:
                    pjoint = joints[parents[i]]
                    joint = joints[i]
                    d = joint - pjoint
                    if np.linalg.norm(d) < 0.000001:
                        d = np.array([0., 0., 1.])
                    else:
                        d = d / np.linalg.norm(d)
                    tails[i] = joint + d * extrude_size
        if use_connect_unique_child:
            for i in range(J):
                if len(children[i]) == 1:
                    child = children[i][0]
                    tails[i] = joints[child]
                if parents[i] is not None and len(children[parents[i]]) == 1:
                    connects[i] = True

    if add_root:
        bone_root = edit_bones.get('Bone')
        bone_root.name = 'Root'
        bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
    else:
        bone_root = edit_bones.get('Bone')
        bone_root.name = names[0]
        bone_root.head = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
        bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2] + extrude_size))

    def extrude_bone(
        edit_bones,
        name,
        parent_name,
        head,
        tail,
        connect,
    ):
        bone = edit_bones.new(name)
        bone.head = Vector((head[0], head[1], head[2]))
        bone.tail = Vector((tail[0], tail[1], tail[2]))
        bone.name = name
        parent_bone = edit_bones.get(parent_name)
        bone.parent = parent_bone
        bone.use_connect = connect
        assert not np.isnan(head).any(), f"nan found in head of bone {name}"
        assert not np.isnan(tail).any(), f"nan found in tail of bone {name}"

    for i in range(J):
        if add_root is False and i == 0:
            continue
        edit_bones = armature.edit_bones
        pname = 'Root' if parents[i] is None else names[parents[i]]
        extrude_bone(edit_bones, names[i], pname, joints[i], tails[i], connects[i])
    for i in range(J):
        bone = edit_bones.get(names[i])
        bone.head = Vector((joints[i, 0], joints[i, 1], joints[i, 2]))
        bone.tail = Vector((tails[i, 0], tails[i, 1], tails[i, 2]))

    if vertices is None or skin is None:
        return
    # must set to object mode to enable parent_set
    bpy.ops.object.mode_set(mode='OBJECT')
    objects = bpy.data.objects
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob = objects['character']
    arm = bpy.data.objects['Armature']
    ob.select_set(True)
    arm.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    vis = []
    for x in ob.vertex_groups:
        vis.append(x.name)
    # sparsify
    argsorted = np.argsort(-skin, axis=1)
    vertex_group_reweight = skin[np.arange(skin.shape[0])[..., None], argsorted]
    if group_per_vertex == -1:
        group_per_vertex = vertex_group_reweight.shape[-1]
    if not do_not_normalize:
        vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[..., None]

    for v, w in enumerate(skin):
        for ii in range(group_per_vertex):
            i = argsorted[v, ii]
            if i >= J:
                continue
            n = names[i]
            if n not in vis:
                continue
            ob.vertex_groups[n].add([v], vertex_group_reweight[v, ii], 'REPLACE')


def main():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_file', type=str, required=True)
    args = parser.parse_args(argv)

    with open(args.params_file, 'r') as f:
        params = json.load(f)

    data_file = params['data_file']
    with np.load(data_file, allow_pickle=True) as data:
        joints = np.array(data['joints'])
        vertices = np.array(data['vertices']) if 'vertices' in data else None
        skin = np.array(data['skin']) if 'skin' in data else None
        faces = np.array(data['faces']) if 'faces' in data else None
        tails = np.array(data['tails']) if 'tails' in data else None

    path = params['path']
    names = params['names']
    parents = params['parents']
    parents = [None if p == -1 else p for p in parents]

    extrude_size = params.get('extrude_size', 0.03)
    group_per_vertex = params.get('group_per_vertex', -1)
    add_root = params.get('add_root', False)
    do_not_normalize = params.get('do_not_normalize', False)
    use_extrude_bone = params.get('use_extrude_bone', True)
    use_connect_unique_child = params.get('use_connect_unique_child', True)
    extrude_from_parent = params.get('extrude_from_parent', True)

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    clean_bpy()
    make_armature(
        vertices=vertices,
        joints=joints,
        skin=skin,
        parents=parents,
        names=names,
        faces=faces,
        extrude_size=extrude_size,
        group_per_vertex=group_per_vertex,
        add_root=add_root,
        do_not_normalize=do_not_normalize,
        use_extrude_bone=use_extrude_bone,
        use_connect_unique_child=use_connect_unique_child,
        extrude_from_parent=extrude_from_parent,
        tails=tails,
    )
    bpy.ops.export_scene.fbx(filepath=path, check_existing=False, add_leaf_bones=False)
    print(f"Exported FBX to {path}")


if __name__ == '__main__':
    main()
