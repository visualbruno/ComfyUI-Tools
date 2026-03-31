"""
Blender extraction script for UniRig.

Run with:
    <blender_path> --background --python blender_extract.py -- --params_file <path_to_params.json>

Extracts mesh and armature data from 3D model files using Blender's Python API
and saves intermediate numpy arrays for further processing outside of Blender.
"""

import bpy
import sys
import os
import json
import numpy as np


def clean_bpy():
    """Remove all data in bpy."""
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception as e:
        print(f"Warning: Could not purge orphans: {e}")

    data_types = [
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.cameras,
        bpy.data.collections,
        bpy.data.curves,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.materials,
        bpy.data.meshes,
        bpy.data.objects,
        bpy.data.textures,
        bpy.data.worlds,
        bpy.data.node_groups
    ]

    for data_collection in data_types:
        try:
            for item in data_collection:
                try:
                    data_collection.remove(item)
                except Exception as e:
                    print(f"Warning: Could not remove {item.name} from {data_collection}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {data_collection}: {e}")

    import gc
    gc.collect()


def load(filepath):
    """Load a 3D model file into Blender and return the armature (or None)."""
    old_objs = set(bpy.context.scene.objects)

    if not os.path.exists(filepath):
        raise ValueError(f'File {filepath} does not exist !')

    try:
        if filepath.endswith(".vrm"):
            bpy.ops.preferences.addon_enable(module='vrm')
            bpy.ops.import_scene.vrm(
                filepath=filepath,
                use_addon_preferences=True,
                extract_textures_into_folder=False,
                make_new_texture_folder=False,
                set_shading_type_to_material_on_import=False,
                set_view_transform_to_standard_on_import=True,
                set_armature_display_to_wire=True,
                set_armature_display_to_show_in_front=True,
                set_armature_bone_shape_to_default=True,
                disable_bake=True,
            )
        elif filepath.endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=filepath)
        elif filepath.endswith(".fbx") or filepath.endswith(".FBX"):
            bpy.ops.import_scene.fbx(filepath=filepath, ignore_leaf_bones=False, use_image_search=False)
        elif filepath.endswith(".glb") or filepath.endswith(".gltf"):
            bpy.ops.import_scene.gltf(filepath=filepath, import_pack_images=False)
        elif filepath.endswith(".dae"):
            bpy.ops.wm.collada_import(filepath=filepath)
        elif filepath.endswith(".blend"):
            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                data_to.objects = data_from.objects
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
        else:
            raise ValueError(f"not supported type {filepath}")
    except:
        raise ValueError(f"failed to load {filepath}")

    armature = [x for x in set(bpy.context.scene.objects)-old_objs if x.type=="ARMATURE"]
    if len(armature)==0:
        return None
    if len(armature)>1:
        raise ValueError(f"multiple armatures found")
    armature = armature[0]

    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in bpy.data.armatures[0].edit_bones:
        bone.roll = 0.

    bpy.ops.object.mode_set(mode='OBJECT')
    armature.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    return armature


def get_arranged_bones(armature):
    """Get bones in BFS order sorted by position."""
    matrix_world = armature.matrix_world
    arranged_bones = []
    root = armature.pose.bones[0]
    while root.parent is not None:
        root = root.parent
    Q = [root]
    rot = np.array(matrix_world)[:3, :3]

    while len(Q) != 0:
        b = Q.pop(0)
        arranged_bones.append(b)
        children = []
        for cb in b.children:
            head = rot @ np.array(b.head)
            children.append((cb, head[0], head[1], head[2]))
        children = sorted(children, key=lambda x: (x[3], x[1], x[2]))
        _c = [x[0] for x in children]
        Q = _c + Q
    return arranged_bones


def process_mesh(arranged_bones=None):
    """Extract mesh data from the Blender scene."""
    meshes = []
    for v in bpy.data.objects:
        if v.type == 'MESH':
            meshes.append(v)

    if arranged_bones is not None:
        index = {}
        for (id, pbone) in enumerate(arranged_bones):
            index[pbone.name] = id

    _dict_mesh = {}
    _dict_skin = {}
    if arranged_bones is not None:
        total_bones = len(arranged_bones)
    else:
        total_bones = None
    for obj in meshes:
        m = np.array(obj.matrix_world)
        matrix_world_rot = m[:3, :3]
        matrix_world_bias = m[:3, 3]
        rot = matrix_world_rot
        total_vertices = len(obj.data.vertices)
        vertex = np.zeros((4, total_vertices))
        vertex_normal = np.zeros((total_vertices, 3))
        if total_bones is not None:
            skin_weight = np.zeros((total_vertices, total_bones))
        else:
            skin_weight = None
        obj_verts = obj.data.vertices
        faces = []
        normals = []

        for v in obj_verts:
            vertex_normal[v.index] = rot @ np.array(v.normal)
            vv = rot @ v.co
            vv = np.array(vv) + matrix_world_bias
            vertex[0:3, v.index] = vv
            vertex[3][v.index] = 1

        for polygon in obj.data.polygons:
            edges = polygon.edge_keys
            nodes = []
            adj = {}
            for edge in edges:
                if adj.get(edge[0]) is None:
                    adj[edge[0]] = []
                adj[edge[0]].append(edge[1])
                if adj.get(edge[1]) is None:
                    adj[edge[1]] = []
                adj[edge[1]].append(edge[0])
                nodes.append(edge[0])
                nodes.append(edge[1])
            normal = polygon.normal
            nodes = list(set(sorted(nodes)))
            first = nodes[0]
            loop = []
            now = first
            vis = {}
            while True:
                loop.append(now)
                vis[now] = True
                if vis.get(adj[now][0]) is None:
                    now = adj[now][0]
                elif vis.get(adj[now][1]) is None:
                    now = adj[now][1]
                else:
                    break
            for (second, third) in zip(loop[1:], loop[2:]):
                faces.append((first + 1, second + 1, third + 1))
                normals.append(rot @ normal)

        obj_group_names = [g.name for g in obj.vertex_groups]
        if arranged_bones is not None:
            for bone in arranged_bones:
                if bone.name not in obj_group_names:
                    continue
                gidx = obj.vertex_groups[bone.name].index
                bone_verts = [v for v in obj_verts if gidx in [g.group for g in v.groups]]
                for v in bone_verts:
                    which = [id for id in range(len(v.groups)) if v.groups[id].group==gidx]
                    w = v.groups[which[0]].weight
                    assert(0 <= v.index < total_vertices)
                    vv = rot @ v.co
                    vv = np.array(vv) + matrix_world_bias
                    vertex[0:3, v.index] = vv
                    vertex[3][v.index] = 1
                    skin_weight[v.index, index[bone.name]] = w

        correct_faces = []
        for (i, face) in enumerate(faces):
            normal = normals[i]
            v0 = face[0] - 1
            v1 = face[1] - 1
            v2 = face[2] - 1
            v = np.cross(
                vertex[:3, v1] - vertex[:3, v0],
                vertex[:3, v2] - vertex[:3, v0],
            )
            if (v*normal).sum() > 0:
                correct_faces.append(face)
            else:
                correct_faces.append((face[0], face[2], face[1]))
        if len(correct_faces) > 0:
            _dict_mesh[obj.name] = {
                'vertex': vertex,
                'face': correct_faces,
            }
            if skin_weight is not None:
                _dict_skin[obj.name] = {
                    'skin': skin_weight,
                }

    vertex = np.concatenate([_dict_mesh[name]['vertex'] for name in _dict_mesh], axis=1)[:3, :].transpose()

    total_faces = 0
    now_bias = 0
    for name in _dict_mesh:
        total_faces += len(_dict_mesh[name]['face'])
    faces = np.zeros((total_faces, 3), dtype=np.int64)
    tot = 0
    for name in _dict_mesh:
        f = np.array(_dict_mesh[name]['face'], dtype=np.int64)
        faces[tot:tot+f.shape[0]] = f + now_bias
        now_bias += _dict_mesh[name]['vertex'].shape[1]
        tot += f.shape[0]

    skin = None
    if arranged_bones is not None and len(_dict_skin) > 0:
        skin = np.concatenate([
            _dict_skin[d]['skin'] for d in _dict_skin
        ], axis=0)

    return vertex, faces, skin


def process_armature(armature, arranged_bones):
    """Extract armature/skeleton data from Blender."""
    matrix_world = armature.matrix_world
    index = {}

    for (id, pbone) in enumerate(arranged_bones):
        index[pbone.name] = id

    root = armature.pose.bones[0]
    while root.parent is not None:
        root = root.parent
    m = np.array(matrix_world.to_4x4())
    scale_inv = np.linalg.inv(np.diag(matrix_world.to_scale()))
    rot = m[:3, :3]
    bias = m[:3, 3]

    s = []
    bpy.ops.object.editmode_toggle()
    edit_bones = armature.data.edit_bones

    J = len(arranged_bones)
    joints = np.zeros((J, 3), dtype=np.float32)
    tails = np.zeros((J, 3), dtype=np.float32)
    parents = []
    name_to_id = {}
    names = []
    matrix_local_stack = np.zeros((J, 4, 4), dtype=np.float32)
    for (id, pbone) in enumerate(arranged_bones):
        name = pbone.name
        names.append(name)
        matrix_local = np.array(pbone.bone.matrix_local)
        use_inherit_rotation = pbone.bone.use_inherit_rotation
        if use_inherit_rotation == False:
            print(f"Warning: use_inherit_rotation of bone {name} is False !")
        head = rot @ matrix_local[0:3, 3] + bias
        s.append(head)
        edit_bone = edit_bones.get(name)
        tail = rot @ np.array(edit_bone.tail) + bias

        name_to_id[name] = id
        joints[id] = head
        tails[id] = tail
        parents.append(None if pbone.parent not in arranged_bones else name_to_id[pbone.parent.name])
        # remove scale part
        matrix_local[:, 3:4] = m @ matrix_local[:, 3:4]
        matrix_local[:3, :3] = scale_inv @ matrix_local[:3, :3]
        matrix_local_stack[id] = matrix_local
    bpy.ops.object.editmode_toggle()

    return joints, tails, parents, names, matrix_local_stack


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

    files = params['files']
    tot = 0

    for file_entry in files:
        input_file = file_entry[0]
        output_dir = file_entry[1]

        clean_bpy()

        try:
            print(f"Now processing {input_file}...")
            armature = load(input_file)

            print('save to:', output_dir)
            os.makedirs(output_dir, exist_ok=True)

            if armature is not None:
                arranged_bones = get_arranged_bones(armature)
            else:
                arranged_bones = None

            vertices, faces, skin = process_mesh(arranged_bones)

            if armature is not None:
                joints, tails, parents, names, matrix_local = process_armature(armature, arranged_bones)
            else:
                joints = tails = parents = names = matrix_local = None

            # Save intermediate data as numpy arrays
            save_dict = {
                'vertices': vertices,
                'faces': faces,
            }
            if skin is not None:
                save_dict['skin'] = skin
            if joints is not None:
                save_dict['joints'] = joints
            if tails is not None:
                save_dict['tails'] = tails
            if matrix_local is not None:
                save_dict['matrix_local'] = matrix_local
            if parents is not None:
                # Encode None as -1 for numpy serialization
                save_dict['parents'] = np.array(
                    [p if p is not None else -1 for p in parents], dtype=np.int64
                )
            if names is not None:
                save_dict['names'] = np.array(names)

            intermediate_file = os.path.join(output_dir, '_blender_intermediate.npz')
            np.savez(intermediate_file, **save_dict)
            print(f"Saved intermediate data to {intermediate_file}")
            tot += 1

        except ValueError as e:
            print(f"ValueError: {str(e)}")
        except RuntimeError as e:
            print(f"RuntimeError: {str(e)}")
        except TimeoutError as e:
            print("TimeoutError: Processing timed out")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

    print(f"{tot} files extracted by Blender successfully")


if __name__ == '__main__':
    main()
