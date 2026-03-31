import trimesh as Trimesh
import numpy as np
from PIL import Image
import ctypes
import os
import sys
import copy
import torch

file_directory = os.path.dirname(os.path.abspath(__file__))
libs_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)),'libs')

if sys.platform == 'win32':
    lib_path = os.path.join(libs_directory, 'meshoptimizer.dll')
else:
    lib_path = os.path.join(libs_directory, 'libmeshoptimizer.so')

meshopt = ctypes.CDLL(lib_path)

## Simplification
# 2. Define C-types function signatures
meshopt.meshopt_simplify.argtypes = [
    ctypes.POINTER(ctypes.c_uint32),  # destination indices
    ctypes.POINTER(ctypes.c_uint32),  # source indices
    ctypes.c_size_t,                  # index_count
    ctypes.POINTER(ctypes.c_float),   # vertex_positions
    ctypes.c_size_t,                  # vertex_count
    ctypes.c_size_t,                  # vertex_positions_stride
    ctypes.c_size_t,                  # target_index_count
    ctypes.c_float,                   # target_error
    ctypes.c_uint32,                  # options
    ctypes.POINTER(ctypes.c_float)    # result_error
]
meshopt.meshopt_simplify.restype = ctypes.c_size_t

meshopt.meshopt_optimizeVertexFetch.argtypes = [
    ctypes.c_void_p,                  # destination vertices
    ctypes.POINTER(ctypes.c_uint32),  # indices (used as in & out)
    ctypes.c_size_t,                  # index_count
    ctypes.c_void_p,                  # source vertices
    ctypes.c_size_t,                  # vertex_count
    ctypes.c_size_t                   # vertex_size
]
meshopt.meshopt_optimizeVertexFetch.restype = ctypes.c_size_t

# 3. Define Enum Flags
meshopt_SimplifyLockBorder = 1 << 0
meshopt_SimplifySparse = 1 << 1
meshopt_SimplifyErrorAbsolute = 1 << 2
meshopt_SimplifyPrune = 1 << 3
meshopt_SimplifyRegularize = 1 << 4
meshopt_SimplifyPermissive = 1 << 5
meshopt_SimplifyRegularizeLight  = 1 << 6

def simplify_mesh(vertices, faces, target_face_count, target_error=1e-2, options = 0):
    """
    Simplifies a 3D mesh by reducing its face count.
    
    :param vertices: Nx3 numpy array of vertex positions (float32)
    :param faces: Mx3 numpy array of face indices (uint32)
    :param target_face_count: How many faces you want remaining
    :param target_error: Allowable geometric error relative to mesh extents
    :return: A tuple of (compact_vertices, compact_faces)
    """
    # Ensure we are working with contiguous C-compatible memory arrays
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(faces.flatten(), dtype=np.uint32)
    
    vertex_count = len(vertices)
    index_count = len(indices)
    target_index_count = target_face_count * 3
    
    # 1. SIMPLIFY
    # Allocate destination array for indices (worst case scenario: same size as original)
    dest_indices = np.zeros(index_count, dtype=np.uint32)
    result_error = ctypes.c_float(0.0)
    
    #combined_options = meshopt_SimplifyLockBorder | meshopt_SimplifyPrune
    
    # Call to meshopt_simplify
    new_index_count = meshopt.meshopt_simplify(
        dest_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        index_count,
        vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        vertex_count,
        12, # Stride: 3 floats * 4 bytes
        target_index_count,
        target_error,
        options, # Applying your desired option flag mapping boundaries
        ctypes.byref(result_error)
    )
    
    # Slice the result down to the actual simplified index count
    simplified_indices = dest_indices[:new_index_count]
    
    # 2. COMPACT VERTEX BUFFER
    # At this point, `simplified_indices` points to uncompacted original `vertices`. 
    # Allocate memory for the filtered/compacted vertices
    dest_vertices = np.zeros_like(vertices)
    
    new_vertex_count = meshopt.meshopt_optimizeVertexFetch(
        dest_vertices.ctypes.data_as(ctypes.c_void_p),
        simplified_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), # Modified in-place!
        new_index_count,
        vertices.ctypes.data_as(ctypes.c_void_p),
        vertex_count,
        12 # Size of each vertex: 3 floats * 4 bytes
    )
    
    # Subslice only the remaining used portion
    compact_vertices = dest_vertices[:new_vertex_count]
    compact_faces = simplified_indices.reshape(-1, 3)
    
    return compact_vertices, compact_faces

#Optimization
# Setup C-types signatures for the functions
meshopt.meshopt_generateVertexRemap.argtypes = [
    ctypes.POINTER(ctypes.c_uint32), # destination (remap table)
    ctypes.POINTER(ctypes.c_uint32), # indices
    ctypes.c_size_t,                 # index_count
    ctypes.c_void_p,                 # vertices
    ctypes.c_size_t,                 # vertex_count
    ctypes.c_size_t                  # vertex_size
]
meshopt.meshopt_generateVertexRemap.restype = ctypes.c_size_t

meshopt.meshopt_remapIndexBuffer.argtypes = [
    ctypes.POINTER(ctypes.c_uint32), # destination
    ctypes.POINTER(ctypes.c_uint32), # indices
    ctypes.c_size_t,                 # index_count
    ctypes.POINTER(ctypes.c_uint32)  # remap
]

meshopt.meshopt_remapVertexBuffer.argtypes = [
    ctypes.c_void_p,                 # destination
    ctypes.c_void_p,                 # vertices
    ctypes.c_size_t,                 # vertex_count
    ctypes.c_size_t,                 # vertex_size
    ctypes.POINTER(ctypes.c_uint32)  # remap
]

meshopt.meshopt_optimizeVertexCache.argtypes = [
    ctypes.POINTER(ctypes.c_uint32), # destination
    ctypes.POINTER(ctypes.c_uint32), # indices
    ctypes.c_size_t,                 # index_count
    ctypes.c_size_t                  # vertex_count
]

meshopt.meshopt_optimizeOverdraw.argtypes = [
    ctypes.POINTER(ctypes.c_uint32), # destination
    ctypes.POINTER(ctypes.c_uint32), # indices
    ctypes.c_size_t,                 # index_count
    ctypes.POINTER(ctypes.c_float),  # vertex_positions
    ctypes.c_size_t,                 # vertex_count
    ctypes.c_size_t,                 # vertex_positions_stride
    ctypes.c_float                   # threshold
]

meshopt.meshopt_optimizeVertexFetch.argtypes = [
    ctypes.c_void_p,                 # destination
    ctypes.POINTER(ctypes.c_uint32), # indices (in/out)
    ctypes.c_size_t,                 # index_count
    ctypes.c_void_p,                 # vertices
    ctypes.c_size_t,                 # vertex_count
    ctypes.c_size_t                  # vertex_size
]
meshopt.meshopt_optimizeVertexFetch.restype = ctypes.c_size_t

def weld_mesh(vertices: np.ndarray, indices: np.ndarray):
    """
    Welds identical vertices together by remapping the vertex and index buffers.
    This should be run before simplifying a mesh to ensure continuous topology.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(indices.flatten(), dtype=np.uint32)

    vertex_count = len(vertices)
    index_count = len(indices)
    vertex_size = vertices.dtype.itemsize * vertices.shape[1] 

    # Get raw memory pointers using ctypes
    vertices_ptr = vertices.ctypes.data_as(ctypes.c_void_p)
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    remap = (ctypes.c_uint32 * vertex_count)()
    unique_vert_count = meshopt.meshopt_generateVertexRemap(
        remap, indices_ptr, index_count, vertices_ptr, vertex_count, vertex_size
    )

    remapped_indices = (ctypes.c_uint32 * index_count)()
    meshopt.meshopt_remapIndexBuffer(remapped_indices, indices_ptr, index_count, remap)

    remapped_vertices = np.zeros((unique_vert_count, vertices.shape[1]), dtype=vertices.dtype)
    remapped_vertices_ptr = remapped_vertices.ctypes.data_as(ctypes.c_void_p)
    meshopt.meshopt_remapVertexBuffer(remapped_vertices_ptr, vertices_ptr, vertex_count, vertex_size, remap)

    final_indices = np.ctypeslib.as_array(remapped_indices).reshape(-1, 3)
    return remapped_vertices, final_indices

def optimize_mesh(vertices: np.ndarray, indices: np.ndarray):
    """
    Optimizes a 3D mesh for GPU rendering.
    vertices: numpy array of float32, shape (N, 3) (or structurally matching XYZ...)
    indices: numpy array of uint32, shape (M,)
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(indices.flatten(), dtype=np.uint32)

    vertex_count = len(vertices)
    index_count = len(indices)
    vertex_size = vertices.dtype.itemsize * vertices.shape[1] # e.g., 12 bytes for 3 floats

    # Get raw memory pointers using ctypes
    vertices_ptr = vertices.ctypes.data_as(ctypes.c_void_p)
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    # --- Step 1: Remap (Remove Duplicates) ---
    print('Removing duplicates ...')
    remap = (ctypes.c_uint32 * vertex_count)()
    unique_vert_count = meshopt.meshopt_generateVertexRemap(
        remap, indices_ptr, index_count, vertices_ptr, vertex_count, vertex_size
    )

    # Apply remap to indices and vertices
    remapped_indices = (ctypes.c_uint32 * index_count)()
    meshopt.meshopt_remapIndexBuffer(remapped_indices, indices_ptr, index_count, remap)

    remapped_vertices = np.zeros((unique_vert_count, vertices.shape[1]), dtype=vertices.dtype)
    remapped_vertices_ptr = remapped_vertices.ctypes.data_as(ctypes.c_void_p)
    meshopt.meshopt_remapVertexBuffer(remapped_vertices_ptr, vertices_ptr, vertex_count, vertex_size, remap)

    # Update state for next steps
    current_indices = remapped_indices
    current_vertices = remapped_vertices
    current_vert_count = unique_vert_count

    # --- Step 2: Vertex Cache Optimization ---
    print('Vertex cache optimization ...')
    vcache_indices = (ctypes.c_uint32 * index_count)()
    meshopt.meshopt_optimizeVertexCache(
        vcache_indices, current_indices, index_count, current_vert_count
    )

    # --- Step 3: Overdraw Optimization ---
    print('Overdraw optimization ...')
    overdraw_indices = (ctypes.c_uint32 * index_count)()
    positions_ptr = current_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # 1.05 = allows cache hit rate to drop by roughly 5% to gain more overdraw reduction
    threshold = 1.05 
    meshopt.meshopt_optimizeOverdraw(
        overdraw_indices, vcache_indices, index_count, 
        positions_ptr, current_vert_count, vertex_size, threshold
    )

    # --- Step 4: Vertex Fetch Optimization ---
    print('Vertex fetch optimization ...')
    final_vertices = np.zeros_like(current_vertices)
    final_vertices_ptr = final_vertices.ctypes.data_as(ctypes.c_void_p)
    
    # This edits `overdraw_indices` IN-PLACE to match the new vertex layout!
    meshopt.meshopt_optimizeVertexFetch(
        final_vertices_ptr, overdraw_indices, index_count,
        current_vertices.ctypes.data_as(ctypes.c_void_p), current_vert_count, vertex_size
    )

    # Convert ctypes index array back to numpy
    final_indices = np.ctypeslib.as_array(overdraw_indices)
    final_indices = final_indices.reshape(-1, 3)

    return final_vertices, final_indices

class VisualBrunoToolsProjectionMultiViewTexturing:
    """
    Apply texture to mesh by projecting multiple view images.
    
    Uses angle-weighted blending: each surface receives texture from all views
    that can "see" it, weighted by how directly the surface faces each camera.
    
    Camera angles (Y-up coordinate system):
    - Azimuth: rotation around Y axis
      - 0° = front (looking in -Z direction)
      - 90° = left (looking in -X direction)  
      - 180° = back (looking in +Z direction)
      - 270° = right (looking in +X direction)
    - Elevation: rotation around X axis
      - 0° = horizontal
      - 90° = top (looking in -Y direction, from above)
      - -90° = bottom (looking in +Y direction, from below)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 8192}),
                "blend_texture": ("BOOLEAN", {"default":True}),
                "blend_exponent": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.5}),
                "ortho_scale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.01}),
                "norm_size": ("FLOAT",{"default":1.15, "min":0.0, "max":9.99, "step":0.01}),
                "fill_holes": ("BOOLEAN",{"default":True}),
                "max_hole_size": ("INT",{"default":10,"min":0,"max":99999,"step":1}),
            },
            "optional": {
                # Standard views
                "front_image": ("IMAGE",),   # az=0, el=0
                "back_image": ("IMAGE",),    # az=180, el=0
                "left_image": ("IMAGE",),    # az=90, el=0
                "right_image": ("IMAGE",),   # az=270, el=0
                "top_image": ("IMAGE",),     # az=0, el=90
                "bottom_image": ("IMAGE",),  # az=0, el=-90
                # Custom views
                "custom_images": ("IMAGE",),
                "custom_azimuths": ("STRING", {"default": ""}),
                "custom_elevations": ("STRING", {"default": ""}),
                "camera_config": ("HY3DCAMERA",),
            }
        }
    
    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("trimesh", "base_color", "metallic_roughness",)
    FUNCTION = "process"
    CATEGORY = "VisualBrunoTools/3d"
    OUTPUT_NODE = True
    
    def process(
        self,
        trimesh,
        texture_size,
        blend_texture,
        blend_exponent,
        ortho_scale,
        norm_size,
        fill_holes,
        max_hole_size,
        baseColorTexture = None,
        front_image=None,
        back_image=None,
        left_image=None,
        right_image=None,
        top_image=None,
        bottom_image=None,
        custom_images=None,
        custom_azimuths="",
        custom_elevations="",
        camera_config = None
    ):
        from ..scripts.texture_projection_multiview import texture_mesh_with_multiview
        
        # Collect views
        images = []
        azimuths = []
        elevations = []
        
        # Standard views with their camera angles
        standard_views = [
            (front_image, 0, 0, "front"),
            (back_image, 180, 0, "back"),
            (left_image, 90, 0, "left"),
            (right_image, 270, 0, "right"),
            (top_image, 0, 90, "top"),
            (bottom_image, 0, -90, "bottom"),
        ]
        
        for img, az, el, name in standard_views:
            if img is not None:
                images.append(self._tensor_to_pil(img))
                azimuths.append(az)
                elevations.append(el)
                print(f"[MultiView] Added {name} view (az={az}, el={el})")
        
        # Custom views
        if custom_images is not None:
            custom_az_list = self._parse_angles(custom_azimuths)
            custom_el_list = self._parse_angles(custom_elevations)
            
            if custom_az_list and custom_el_list:
                num_custom = min(len(custom_az_list), len(custom_el_list), int(custom_images.shape[0]))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(custom_az_list[i])
                    elevations.append(custom_el_list[i])
                    print(f"[MultiView] Added custom view {i+1} (az={custom_az_list[i]}, el={custom_el_list[i]})")
            elif camera_config:
                selected_camera_azims = camera_config["selected_camera_azims"]
                selected_camera_elevs = camera_config["selected_camera_elevs"]
                #ortho_scale = camera_config["ortho_scale"]             

                num_custom = min(len(selected_camera_azims), len(selected_camera_elevs), int(custom_images.shape[0]))
                for i in range(num_custom):
                    images.append(self._tensor_to_pil(custom_images[i:i+1]))
                    azimuths.append(selected_camera_azims[i])
                    elevations.append(selected_camera_elevs[i])
                    print(f"[MultiView] Added custom view {i+1} (az={selected_camera_azims[i]}, el={selected_camera_elevs[i]})")                
        
        if len(images) == 0:
            raise ValueError("No input images provided! Please connect at least one image.")
        
        print(f"[MultiView] Total views: {len(images)}")
        print(f"[MultiView] Azimuths: {azimuths}")
        print(f"[MultiView] Elevations: {elevations}")

        trimesh_obj, base_color, mr = texture_mesh_with_multiview(
            trimesh,
            images,
            azimuths,
            elevations,
            texture_size=texture_size,
            blend_exponent=blend_exponent,
            ortho_scale=ortho_scale,
            blend_texture=blend_texture,
            fill_holes=fill_holes,
            norm_size=norm_size,
            max_hole_size=max_hole_size
        )
        
        return (trimesh_obj, pil2tensor(base_color), pil2tensor(mr))
    
    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI IMAGE tensor to PIL."""
        if len(tensor.shape) == 4:
            arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return []
            
class VisualBrunoToolsMeshSimplify:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_count": ("INT",{"default":100000,"min":1,"max":10000000,"step":1}),
                "weld_vertices": ("BOOLEAN",{"default":True}),
                "lock_border": ("BOOLEAN",{"default":True}),
                "sparse": ("BOOLEAN",{"default":False}),
                "error_absolute": ("BOOLEAN",{"default":False}),
                "prune": ("BOOLEAN",{"default":False}),
                "regularize": ("BOOLEAN",{"default":False}),
                "permissive": ("BOOLEAN",{"default":False}),
                "regularize_light": ("BOOLEAN",{"default":False}),
            },
        }
    
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "VisualBrunoTools/3d"
    OUTPUT_NODE = True
    
    def process(
        self,
        trimesh,
        target_face_count,
        weld_vertices,
        lock_border,
        sparse,
        error_absolute,
        prune,
        regularize,
        permissive,
        regularize_light
    ):
        mesh = trimesh.copy()
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        print(f"Original: {len(vertices)} vertices, {len(faces)} faces")
        
        if weld_vertices:
            welded_vertices, welded_faces = weld_mesh(vertices, faces)
            print(f"Welded: {len(welded_vertices)} vertices, {len(welded_faces)} faces")
        else:
            welded_vertices = vertices
            welded_faces = faces
            
        options = 0
        if lock_border:
            options = options | meshopt_SimplifyLockBorder
        if sparse:
            options = options | meshopt_SimplifySparse
        if error_absolute:
            options = options | meshopt_SimplifyErrorAbsolute
        if prune:
            options = options | meshopt_SimplifyPrune
        if regularize:
            options = options | meshopt_SimplifyRegularize
        if permissive:
            options = options | meshopt_SimplifyPermissive
        if regularize_light:
            options = options | meshopt_SimplifyRegularizeLight
            
        new_vertices, new_faces = simplify_mesh(welded_vertices, welded_faces, target_face_count=target_face_count, options = options)
        
        print(f"Simplified (after weld): {len(new_vertices)} vertices, {len(new_faces)} faces")
        
        mesh.faces = new_faces
        mesh.vertices = new_vertices
        
        return (mesh,)
        
class VisualBrunoToolsMeshSimplifyTrellis2:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_face_count": ("INT",{"default":100000,"min":1,"max":10000000,"step":1}),
                "weld_vertices": ("BOOLEAN",{"default":True}),
                "lock_border": ("BOOLEAN",{"default":True}),
                "sparse": ("BOOLEAN",{"default":False}),
                "error_absolute": ("BOOLEAN",{"default":False}),
                "prune": ("BOOLEAN",{"default":False}),
                "regularize": ("BOOLEAN",{"default":False}),
                "permissive": ("BOOLEAN",{"default":False}),
                "regularize_light": ("BOOLEAN",{"default":False}),
            },
        }
    
    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "VisualBrunoTools/3d"
    OUTPUT_NODE = True
    
    def process(
        self,
        mesh,
        target_face_count,
        weld_vertices,
        lock_border,
        sparse,
        error_absolute,
        prune,
        regularize,
        permissive,
        regularize_light
    ):
        mesh_copy = copy.deepcopy(mesh)
        
        vertices = mesh_copy.vertices.cpu().numpy()
        faces = mesh_copy.faces.cpu().numpy()
        
        print(f"Original: {len(vertices)} vertices, {len(faces)} faces")
        
        if weld_vertices:
            welded_vertices, welded_faces = weld_mesh(vertices, faces)
            print(f"Welded: {len(welded_vertices)} vertices, {len(welded_faces)} faces")
        else:
            welded_vertices = vertices
            welded_faces = faces
            
        options = 0
        if lock_border:
            options = options | meshopt_SimplifyLockBorder
        if sparse:
            options = options | meshopt_SimplifySparse
        if error_absolute:
            options = options | meshopt_SimplifyErrorAbsolute
        if prune:
            options = options | meshopt_SimplifyPrune
        if regularize:
            options = options | meshopt_SimplifyRegularize
        if permissive:
            options = options | meshopt_SimplifyPermissive
        if regularize_light:
            options = options | meshopt_SimplifyRegularizeLight
            
        new_vertices, new_faces = simplify_mesh(welded_vertices, welded_faces, target_face_count=target_face_count, options = options)
        
        print(f"Simplified (after weld): {len(new_vertices)} vertices, {len(new_faces)} faces")
        
        mesh_copy.faces = torch.from_numpy(new_faces).float()
        mesh_copy.vertices = torch.from_numpy(new_vertices).float()
        
        return (mesh_copy,)        