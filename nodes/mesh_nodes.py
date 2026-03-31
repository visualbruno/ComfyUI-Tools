import ctypes
import numpy as np
import os
import sys
import trimesh

file_directory = os.path.dirname(os.path.abspath(__file__))
libs_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)),'libs')

if sys.platform == 'win32':
    lib_path = os.path.join(lib_path, 'meshoptimizer.dll')
else:
    lib_path = os.path.join(lib_path, 'libmeshoptimizer.so')

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

def simplify_mesh(vertices, faces, target_face_count, target_error=1e-2):
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
    
    combined_options = meshopt_SimplifyLockBorder | meshopt_SimplifyPrune
    
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
        combined_options, # Applying your desired option flag mapping boundaries
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

# ==============================================================
# EXAMPLE USAGE:
# ==============================================================
if __name__ == "__main__":
    mesh = trimesh.load('AnimeGirl.glb',force='mesh')
    sample_vertices = mesh.vertices
    sample_faces = mesh.faces

    print(f"Original: {len(sample_vertices)} vertices, {len(sample_faces)} faces")
    
    new_vertices, new_faces = optimize_mesh(sample_vertices, sample_faces)
    
    print(f"Optimized: {len(new_vertices)} vertices, {len(new_faces)} faces")    
    
    new_vertices, new_faces = simplify_mesh(new_vertices, new_faces, target_face_count=10000)
    
    print(f"Simplified: {len(new_vertices)} vertices, {len(new_faces)} faces")
    
    mesh.faces = new_faces
    mesh.vertices = new_vertices
    
    mesh.export('AnimeGirl_20K_optimized.obj',file_type='obj')

    welded_vertices, welded_faces = weld_mesh(sample_vertices, sample_faces)
    print(f"Welded: {len(welded_vertices)} vertices, {len(welded_faces)} faces")

    new_vertices, new_faces = simplify_mesh(welded_vertices, welded_faces, target_face_count=10000)

    print(f"Simplified (after weld): {len(new_vertices)} vertices, {len(new_faces)} faces")

    mesh.faces = new_faces
    mesh.vertices = new_vertices

    mesh.export('AnimeGirl_20K_welded_simplified.obj',file_type='obj')
