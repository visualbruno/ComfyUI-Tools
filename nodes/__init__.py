# ComfyUI-Meshlib Nodes
# This module exports all node classes for ComfyUI registration

from .blender_nodes import (
    VisualBrunoToolsFBXRenameToSMPL,
)

# Export all node classes
NODE_CLASS_MAPPINGS = {
    # Blender Nodes
    "VisualBrunoToolsFBXRenameToSMPL": VisualBrunoToolsFBXRenameToSMPL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Blender Nodes
    "VisualBrunoToolsFBXRenameToSMPL": "BlenderTools - FBX Rename to SMPL",
}
