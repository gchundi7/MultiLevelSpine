import gmsh
import sys
import os
import numpy as np
from stl import mesh
import pymeshlab
from sklearn.linear_model import RANSACRegressor
import warnings
from scipy.spatial import KDTree

def decimate_stl(input_stl, target_reduction):
    ms = pymeshlab.MeshSet()
    temp_input = "temp_input.stl"
    input_stl.save(temp_input)
    
    ms.load_new_mesh(temp_input)
    initial_faces = ms.current_mesh().face_number()
    target_faces = int(initial_faces * (1 - target_reduction))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    
    temp_decimated = "temp_decimated.stl"
    ms.save_current_mesh(temp_decimated)
    decimated_mesh = mesh.Mesh.from_file(temp_decimated)
    
    os.remove(temp_input)
    os.remove(temp_decimated)
    
    return decimated_mesh

def generate_mesh(input_stl_mesh, target_elements=None):
    bbox_min = np.min(input_stl_mesh.points.reshape(-1, 3), axis=0)
    bbox_max = np.max(input_stl_mesh.points.reshape(-1, 3), axis=0)
    characteristic_length = np.mean(bbox_max - bbox_min)
    mesh_size = characteristic_length / (target_elements ** (1/3))

    gmsh.initialize()
    gmsh.model.add("Model")

    temp_stl = "temp.stl"
    input_stl_mesh.save(temp_stl)
    
    try:
        gmsh.merge(temp_stl)
    finally:
        os.remove(temp_stl)

    angle = 40
    gmsh.model.mesh.classifySurfaces(angle * np.pi / 180., True, True, 180 * np.pi / 180.)
    gmsh.model.mesh.createGeometry()

    surfaces = gmsh.model.getEntities(2)
    surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
    volume = gmsh.model.geo.addVolume([surface_loop])
    
    gmsh.model.geo.synchronize()
    
    gmsh.model.addPhysicalGroup(3, [volume], 1)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 2)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        print(f"Error during mesh generation: {str(e)}")
        return None, None, None

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    
    if not elem_types:
        print("No tetrahedral elements generated")
        return None, None, None
        
    tetras = elem_node_tags[0].reshape(-1, 4) - 1

    return node_coords, tetras, node_tags

def remove_overlapping_elements(disc_nodes, disc_elements, vertebra_nodes, vertebra_elements, tolerance=1e-3):
    """Remove disc elements that overlap with vertebra elements"""
    
    # Create KD-tree for vertebra nodes
    vertebra_tree = KDTree(vertebra_nodes)
    
    # Get centroids of disc elements
    disc_centroids = np.mean(disc_nodes[disc_elements], axis=1)
    
    # Find nearest vertebra nodes for each disc centroid
    distances, _ = vertebra_tree.query(disc_centroids)
    
    # Keep elements whose centroids are not too close to vertebra nodes
    keep_mask = distances > tolerance
    
    return disc_elements[keep_mask]

def save_vtk(filename, nodes, elements):
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {len(nodes)} float\n")
        for node in nodes:
            f.write(f"{node[0]} {node[1]} {node[2]}\n")
            
        f.write(f"CELLS {len(elements)} {len(elements) * 5}\n")
        for elem in elements:
            f.write(f"4 {' '.join(map(str, elem))}\n")
            
        f.write(f"CELL_TYPES {len(elements)}\n")
        for _ in elements:
            f.write("10\n")

def save_combined_vtk(filename, meshes):
    total_nodes = sum(mesh[0].shape[0] for mesh in meshes)
    total_elements = sum(mesh[1].shape[0] for mesh in meshes)
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Combined Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {total_nodes} float\n")
        for nodes, _ in meshes:
            for node in nodes:
                f.write(f"{node[0]} {node[1]} {node[2]}\n")
        
        f.write(f"CELLS {total_elements} {total_elements * 5}\n")
        node_offset = 0
        for nodes, elements in meshes:
            for elem in elements:
                adjusted_elem = [x + node_offset for x in elem]
                f.write(f"4 {' '.join(map(str, adjusted_elem))}\n")
            node_offset += nodes.shape[0]
        
        f.write(f"CELL_TYPES {total_elements}\n")
        for _, elements in meshes:
            for _ in range(elements.shape[0]):
                f.write("10\n")

def main():
    if len(sys.argv) < 6:
        print("Usage: python multilevel.py T12.stl L1.stl L2.stl disc1.stl disc2.stl output_prefix")
        sys.exit(1)

    vertebra_files = sys.argv[1:4]
    disc_files = sys.argv[4:6]
    output_prefix = sys.argv[6]
    
    vertebra_target = 10000  # Elements for vertebrae
    disc_target = 20000     # More elements for discs

    vertebrae_meshes = []
    disc_meshes = []
    all_meshes = []
    
    # Generate vertebrae meshes
    for i, v_file in enumerate(vertebra_files):
        print(f"Processing vertebra {i+1}")
        stl_mesh = mesh.Mesh.from_file(v_file)
        decimated = decimate_stl(stl_mesh, 0.75)
        nodes, elements, _ = generate_mesh(decimated, vertebra_target)
        
        if nodes is None:
            print(f"Failed to generate mesh for vertebra {i+1}")
            continue
            
        vertebrae_meshes.append((nodes, elements))
        all_meshes.append((nodes, elements))
        save_vtk(f"{output_prefix}_vertebra_{i+1}.vtk", nodes, elements)

    # Generate disc meshes
    for i, disc_file in enumerate(disc_files):
        print(f"Processing disc {i+1}")
        disc_stl = mesh.Mesh.from_file(disc_file)
        nodes, elements, _ = generate_mesh(disc_stl, disc_target)
        
        if nodes is None:
            print(f"Failed to generate mesh for disc {i+1}")
            continue
            
        # Remove elements that overlap with adjacent vertebrae
        elements = remove_overlapping_elements(nodes, elements, 
                                            vertebrae_meshes[i][0], vertebrae_meshes[i][1])
        elements = remove_overlapping_elements(nodes, elements, 
                                            vertebrae_meshes[i+1][0], vertebrae_meshes[i+1][1])
        
        disc_meshes.append((nodes, elements))
        all_meshes.append((nodes, elements))
        save_vtk(f"{output_prefix}_disc_{i+1}.vtk", nodes, elements)

    # Save combined mesh
    save_combined_vtk(f"{output_prefix}_combined.vtk", all_meshes)
    print("Mesh generation complete")

if __name__ == "__main__":
    main()