# ComfyUI-Meshlib Nodes
# This module exports all node classes for ComfyUI registration

from .blender_nodes import (
    VisualBrunoToolsFBXRenameToSMPL,
)

from .unirig_nodes import (
    VisualBrunoToolsUniRigModelLoader,
    VisualBrunoToolsUniRigSkeletonPrediction
)

from .threed_nodes import (
    VisualBrunoToolsProjectionMultiViewTexturing,
    VisualBrunoToolsMeshSimplify
)

# Export all node classes
NODE_CLASS_MAPPINGS = {
    # Blender Nodes
    "VisualBrunoToolsFBXRenameToSMPL": VisualBrunoToolsFBXRenameToSMPL,
    
    # UniRig Nodes
    "VisualBrunoToolsUniRigModelLoader": VisualBrunoToolsUniRigModelLoader,
    "VisualBrunoToolsUniRigSkeletonPrediction": VisualBrunoToolsUniRigSkeletonPrediction,
    
    # 3d Nodes
    "VisualBrunoToolsProjectionMultiViewTexturing": VisualBrunoToolsProjectionMultiViewTexturing,
    "VisualBrunoToolsMeshSimplify": VisualBrunoToolsMeshSimplify,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Blender Nodes
    "VisualBrunoToolsFBXRenameToSMPL": "BlenderTools - FBX Rename to SMPL",
    
    # UniRig Nodes
    "VisualBrunoToolsUniRigModelLoader": "UniRig - Model Loader",
    "VisualBrunoToolsUniRigSkeletonPrediction": "UniRig - Skeleton Prediction",
    
    # 3d Nodes
    "VisualBrunoToolsProjectionMultiViewTexturing": "3d - Projection MultiView Texturing",
    "VisualBrunoToolsMeshSimplify": "VisualBrunoToolsMeshSimplify",    
}
