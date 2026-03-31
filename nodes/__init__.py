# ComfyUI-Meshlib Nodes
# This module exports all node classes for ComfyUI registration

from .blender_nodes import (
    VisualBrunoToolsFBXRenameToSMPL,
)

from .unirig_nodes import (
    VisualBrunoToolsUniRigModelLoader,
    VisualBrunoToolsUniRigSkeletonPrediction,
    VisualBrunoToolsUniRigSkinningWeightPrediction,
)

from .threed_nodes import (
    VisualBrunoToolsProjectionMultiViewTexturing,
    VisualBrunoToolsMeshSimplify,
    VisualBrunoToolsMeshSimplifyTrellis2,
)

# Export all node classes
NODE_CLASS_MAPPINGS = {
    # Blender Nodes
    "VisualBrunoToolsFBXRenameToSMPL": VisualBrunoToolsFBXRenameToSMPL,
    
    # UniRig Nodes
    "VisualBrunoToolsUniRigModelLoader": VisualBrunoToolsUniRigModelLoader,
    "VisualBrunoToolsUniRigSkeletonPrediction": VisualBrunoToolsUniRigSkeletonPrediction,
    "VisualBrunoToolsUniRigSkinningWeightPrediction": VisualBrunoToolsUniRigSkinningWeightPrediction,
    
    # 3d Nodes
    "VisualBrunoToolsProjectionMultiViewTexturing": VisualBrunoToolsProjectionMultiViewTexturing,
    "VisualBrunoToolsMeshSimplify": VisualBrunoToolsMeshSimplify,
    "VisualBrunoToolsMeshSimplifyTrellis2": VisualBrunoToolsMeshSimplifyTrellis2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Blender Nodes
    "VisualBrunoToolsFBXRenameToSMPL": "BlenderTools - FBX Rename to SMPL",
    
    # UniRig Nodes
    "VisualBrunoToolsUniRigModelLoader": "UniRig - Model Loader",
    "VisualBrunoToolsUniRigSkeletonPrediction": "UniRig - Skeleton Prediction",
    "VisualBrunoToolsUniRigSkinningWeightPrediction": "UniRig - Skinning Weight Prediction",
    
    # 3d Nodes
    "VisualBrunoToolsProjectionMultiViewTexturing": "3d - Projection MultiView Texturing",
    "VisualBrunoToolsMeshSimplify": "3d - Simplify Trimesh using meshoptimizer",
    "VisualBrunoToolsMeshSimplifyTrellis2": "3d - Simplify Trellis2 Mesh using mesh optimizer",
}
