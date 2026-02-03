import bpy
import json
import os
from mathutils import Vector

# SMPL constants for mapping (from provided logic)
SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
]

SMPL_TREE = {
    0: [1, 2, 3], 1: [4], 2: [5], 3: [6], 4: [7], 5: [8], 6: [9],
    7: [10], 8: [11], 9: [12, 13, 14], 12: [15], 13: [16], 14: [17],
    16: [18], 17: [19], 18: [20], 19: [21]
}

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
PHALANX_TYPES = ['1', '2', '3']

def map_bones_to_smpl(current_bone, smpl_idx, mapping):
    mapping[current_bone.name] = SMPL_JOINT_NAMES[smpl_idx]
    children = current_bone.children
    smpl_children_indices = SMPL_TREE.get(smpl_idx, [])
    if not children or not smpl_children_indices: return

    if len(smpl_children_indices) > 1:
        sorted_children = sorted(children, key=lambda b: b.tail_local[0])
        if smpl_idx == 0: 
            targets = [2, 3, 1] 
            for i, target_smpl in enumerate(targets):
                if i < len(sorted_children): map_bones_to_smpl(sorted_children[i], target_smpl, mapping)
        elif smpl_idx == 9:
            targets = [14, 12, 13]
            for i, target_smpl in enumerate(targets):
                if i < len(sorted_children): map_bones_to_smpl(sorted_children[i], target_smpl, mapping)
    else:
        map_bones_to_smpl(children[0], smpl_children_indices[0], mapping)

def get_finger_mapping(wrist_bone, side_prefix):
    finger_map = {}
    finger_roots = wrist_bone.children
    if not finger_roots: return finger_map
    sorted_fingers = sorted(finger_roots, key=lambda b: b.tail_local[2])
    for i, root_bone in enumerate(sorted_fingers):
        if i < len(FINGER_NAMES):
            finger_label = FINGER_NAMES[i]
            current_joint = root_bone
            for p_idx in PHALANX_TYPES:
                name = f"{side_prefix}_{finger_label}{p_idx}"
                finger_map[current_joint.name] = name
                if current_joint.children: current_joint = current_joint.children[0]
                else: break
    return finger_map

def main():
    input_path = os.getenv('INPUT_MESH')
    output_path = os.getenv('OUTPUT_MESH')

    if not input_path or not output_path:
        print("Error: Environment variables not set.")
        return

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=input_path)
    
    full_mapping = {}
    armature_obj = None

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature_obj = obj
            break

    if not armature_obj: return

    # Generate and apply bone renaming
    root_bone = armature_obj.data.bones[0]
    while root_bone.parent: root_bone = root_bone.parent
    map_bones_to_smpl(root_bone, 0, full_mapping)
    
    for bone in armature_obj.data.bones:
        mapped_name = full_mapping.get(bone.name, "")
        if "Wrist" in mapped_name:
            side = "L" if "L_Wrist" in mapped_name else "R"
            full_mapping.update(get_finger_mapping(bone, side))

    for old_name, new_name in full_mapping.items():
        if old_name in armature_obj.data.bones:
            armature_obj.data.bones[old_name].name = new_name

    # Selection logic: Armature + Linked Meshes
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Select if parented or has Armature Modifier pointing to our armature
            if obj.parent == armature_obj or any(m.type == 'ARMATURE' and m.object == armature_obj for m in obj.modifiers):
                obj.select_set(True)
                # Ensure materials are nodes-based for export
                for slot in obj.material_slots:
                    if slot.material:
                        slot.material.use_nodes = True

    # --- UPDATED EXPORT FOR TEXTURES ---
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=True,
        object_types={'ARMATURE', 'MESH'},
        # These two settings are key for textures:
        path_mode='COPY', 
        embed_textures=True
    )

    print(f"Exported armature, mesh, and embedded textures to: {output_path}")

if __name__ == "__main__":
    main()