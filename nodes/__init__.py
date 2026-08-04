# ComfyUI-Meshlib Nodes
# This module exports all node classes for ComfyUI registration

from .blender_nodes import (
    VisualBrunoToolsFBXRenameToSMPL,
)

from .threed_nodes import (
    VisualBrunoToolsProjectionMultiViewTexturing,
    VisualBrunoToolsMeshSimplify,
    VisualBrunoToolsMeshSimplifyTrellis2,
)

from .image_nodes import (
    VisualBrunoToolsCropImageAlpha,
)

# Export all node classes
NODE_CLASS_MAPPINGS = {
    # Blender Nodes
    "VisualBrunoToolsFBXRenameToSMPL": VisualBrunoToolsFBXRenameToSMPL,
    
    # 3d Nodes
    "VisualBrunoToolsProjectionMultiViewTexturing": VisualBrunoToolsProjectionMultiViewTexturing,
    "VisualBrunoToolsMeshSimplify": VisualBrunoToolsMeshSimplify,
    "VisualBrunoToolsMeshSimplifyTrellis2": VisualBrunoToolsMeshSimplifyTrellis2,
    
    # Image Nodes
    "VisualBrunoToolsCropImageAlpha": VisualBrunoToolsCropImageAlpha,    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Blender Nodes
    "VisualBrunoToolsFBXRenameToSMPL": "BlenderTools - FBX Rename to SMPL",
    
    # 3d Nodes
    "VisualBrunoToolsProjectionMultiViewTexturing": "3d - Projection MultiView Texturing",
    "VisualBrunoToolsMeshSimplify": "3d - Simplify Trimesh using meshoptimizer",
    "VisualBrunoToolsMeshSimplifyTrellis2": "3d - Simplify Trellis2 Mesh using mesh optimizer",
    
    # Blender Nodes
    "VisualBrunoToolsCropImageAlpha": "Image - Crop Image with Alpha",    
}
