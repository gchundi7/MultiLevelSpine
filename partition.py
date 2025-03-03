import gmsh
import sys
import math
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import scipy
from scipy.interpolate import interp1d

@dataclass
class DiscRegions:
    """Configuration class for disc region thresholds"""
    nucleus_threshold: float = 0.3
    layer1_threshold: float = 0.4
    layer2_threshold: float = 0.5
    layer3_threshold: float = 0.6
    layer4_threshold: float = 0.7
    layer5_threshold: float = 0.8
    layer6_threshold: float = 0.9
    layer7_threshold: float = 1.0

    def get_region(self, normalized_radius: float) -> str:
        """Determine which region a point belongs to based on its normalized radius"""
        # Handle edge cases
        if normalized_radius < 0:
            return "nucleus"
        if normalized_radius > 1:
            return "annulus_layer7"
            
        # Normal threshold checks
        if normalized_radius < self.nucleus_threshold:
            return "nucleus"
        elif normalized_radius < self.layer1_threshold:
            return "annulus_layer1" 
        elif normalized_radius < self.layer2_threshold:
            return "annulus_layer2"
        elif normalized_radius < self.layer3_threshold:
            return "annulus_layer3"
        elif normalized_radius < self.layer4_threshold:
            return "annulus_layer4"
        elif normalized_radius < self.layer5_threshold:
            return "annulus_layer5"
        elif normalized_radius < self.layer6_threshold:
            return "annulus_layer6"
        else:
            return "annulus_layer7"

class MeshProcessor:
    def __init__(self, input_file: str, regions: DiscRegions = None):
        self.input_file = input_file
        self.regions = regions or DiscRegions()
        self.model = None
        self.coords = None
        self.elements = None
        self.com = None
        self.max_radius = 0
        self.region_elements: Dict[str, List[int]] = {
            "nucleus": [],
            "annulus_layer1": [],
            "annulus_layer2": [],
            "annulus_layer3": [],
            "annulus_layer4": [],
            "annulus_layer5": [],
            "annulus_layer6": [],
            "annulus_layer7": []
        }

    def create_physical_groups(self):
        """Create physical groups for mesh regions ensuring unique elements"""
        # Base tags for each region
        base_tags = {
            "nucleus": 1000,
            "annulus_layer1": 2000,
            "annulus_layer2": 3000,
            "annulus_layer3": 4000,
            "annulus_layer4": 5000,
            "annulus_layer5": 6000,
            "annulus_layer6": 7000,
            "annulus_layer7": 8000
        }
        
        for region, elements in self.region_elements.items():
            if not elements:
                print(f"\nWarning: No elements for region {region}")
                continue
            
            print(f"\n{region}:")
            print(f"Original elements: {len(elements)}")
            
            try:
                # Create physical group with all elements at once
                self.model.addPhysicalGroup(3, elements, 
                                        tag=base_tags[region],
                                        name=region)
                print(f"Successfully created physical group {region} with {len(elements)} elements")
            except Exception as e:
                print(f"Error creating physical group {region}: {str(e)}")

    def get_radial_distance(self, x: float, y: float, z: float, com: np.ndarray, use_z: bool = False) -> float:
        """Calculate radial distance from a point to the center of mass"""
        dx = x - com[0]
        dy = y - com[1]
        if use_z:
            dz = z - com[2]
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        return math.sqrt(dx*dx + dy*dy)
    
    def get_boundary_points(self) -> Dict[str, List[Tuple[float, float]]]:
        """Sample points along the boundary of each region"""
        boundary_points = {region: [] for region in self.region_elements.keys()}
        
        # Find boundary nodes by checking element faces
        node_faces = {}
        for i in range(0, len(self.elements[1]), 4):
            nodes = self.elements[1][i:i+4]
            faces = [
                tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[3]]))
            ]
            for face in faces:
                node_faces[face] = node_faces.get(face, 0) + 1
        
        # Faces appearing once are boundary faces
        boundary_nodes = set()
        for face, count in node_faces.items():
            if count == 1:
                boundary_nodes.update(face)
                
        print(f"Found {len(boundary_nodes)} boundary nodes")
        
        # Process boundary nodes
        for node in boundary_nodes:
            coords = self.coords[node-1]
            r = self.get_radial_distance(coords[0], coords[1], coords[2], self.com)
            normalized_r = r / self.max_radius
            region = self.regions.get_region(normalized_r)
            boundary_points[region].append((coords[0], coords[1]))
            
        return boundary_points
    
    def smooth_annulus_boundaries(self, boundaries: Dict[str, List[Tuple[float, float]]],
                              smoothing_factor: float = 0.1, n_points: int = 360) -> Dict[str, List[Tuple[float, float]]]:
        """
        Smooth the boundaries for each annulus layer using a spline-based method.
        
        Parameters:
        boundaries: A dictionary mapping each region (e.g., 'annulus_layer4', etc.)
                    to a list of (x, y) tuples that define its boundary.
        smoothing_factor: Parameter controlling the amount of smoothing.
                            Higher values result in smoother boundaries.
        n_points: Number of points to sample in the smoothed boundary.
        
        Returns:
        A new dictionary with the smoothed boundary points for each region.
        """
        import numpy as np
        from scipy.interpolate import UnivariateSpline

        smoothed = {}
        # We'll use the center of mass as the reference center.
        center = self.com

        for region, pts in boundaries.items():
            if len(pts) < 3:
                # Not enough points to smooth – leave as-is.
                smoothed[region] = pts
                continue

            pts_arr = np.array(pts)
            # Convert to polar coordinates relative to center.
            dx = pts_arr[:, 0] - center[0]
            dy = pts_arr[:, 1] - center[1]
            angles = np.arctan2(dy, dx)
            # Normalize angles to be between 0 and 2pi.
            angles = np.mod(angles, 2 * np.pi)
            radii = np.sqrt(dx**2 + dy**2)

            # Sort the points by angle.
            sort_idx = np.argsort(angles)
            angles = angles[sort_idx]
            radii = radii[sort_idx]

            # Append the first point (with angle increased by 2pi) for periodicity.
            angles_ext = np.concatenate((angles, [angles[0] + 2 * np.pi]))
            radii_ext = np.concatenate((radii, [radii[0]]))

            # Create a smoothing spline for the radial distance.
            spline = UnivariateSpline(angles_ext, radii_ext, s=smoothing_factor * len(angles_ext))
            angles_new = np.linspace(angles_ext[0], angles_ext[-1], n_points)
            radii_new = spline(angles_new)

            # Convert back to Cartesian coordinates.
            x_new = center[0] + radii_new * np.cos(angles_new)
            y_new = center[1] + radii_new * np.sin(angles_new)
            smoothed[region] = list(zip(x_new, y_new))

        return smoothed

    def get_disc_boundary(self) -> List[Tuple[float, float]]:
        # Find all boundary faces first
        node_faces = {}
        for i in range(0, len(self.elements[1]), 4):
            nodes = self.elements[1][i:i+4]
            faces = [tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                    tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                    tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                    tuple(sorted([nodes[1], nodes[2], nodes[3]]))]
            for face in faces:
                node_faces[face] = node_faces.get(face, 0) + 1
        
        # Get boundary nodes
        boundary_nodes = set()
        for face, count in node_faces.items():
            if count == 1:  # Boundary face
                mean_z = np.mean([self.coords[n-1][2] for n in face])
                if abs(mean_z - self.com[2]) < 0.5:  # Points near center z-plane
                    boundary_nodes.update(face)
        
        # Convert to XY coordinates and sort by angle
        boundary_points = []
        for node in boundary_nodes:
            coords = self.coords[node-1]
            boundary_points.append((coords[0], coords[1]))
        
        print(f"Found {len(boundary_points)} boundary points on outer edge")
        return boundary_points
    
    def create_proportional_boundaries(self, outer_boundary: List[Tuple[float, float]]) -> Dict[str, List[Tuple[float, float]]]:
        """Create inner boundaries that maintain the shape of the outer boundary"""
        boundaries = {region: [] for region in self.region_elements.keys()}
        boundaries['annulus_layer7'] = outer_boundary
        
        # Get centroid of outer boundary
        centroid = np.mean(outer_boundary, axis=0)
        
        # Create scaled versions for each layer
        scale_factors = {
            'annulus_layer6': 0.9,
            'annulus_layer5': 0.8,
            'annulus_layer4': 0.7,
            'annulus_layer3': 0.6,
            'annulus_layer2': 0.5,
            'annulus_layer1': 0.4,
            'nucleus': 0.3
        }
        
        for region, scale in scale_factors.items():
            boundaries[region] = [(
                centroid[0] + (x - centroid[0]) * scale,
                centroid[1] + (y - centroid[1]) * scale
            ) for x, y in outer_boundary]
        
        return boundaries
    
    def interpolate_boundary(self, boundary_points: Dict[str, List[Tuple[float, float]]]) -> Dict[str, List[Tuple[float, float]]]:
        """Interpolate between boundary points to create smoother region transitions"""
        from scipy.interpolate import interp1d
        
        interpolated = {}
        for region, points in boundary_points.items():
            if len(points) < 3:
                interpolated[region] = points
                continue
                
            # Convert to polar coordinates for better interpolation
            theta = np.arctan2([p[1] - self.com[1] for p in points], 
                            [p[0] - self.com[0] for p in points])
            r = [np.sqrt((p[0] - self.com[0])**2 + (p[1] - self.com[1])**2) for p in points]
            
            # Sort by angle
            sort_idx = np.argsort(theta)
            theta = theta[sort_idx]
            r = np.array(r)[sort_idx]
            
            # Create periodic interpolation
            theta = np.append(theta, theta[0] + 2*np.pi)
            r = np.append(r, r[0])
            
            f = interp1d(theta, r, kind='cubic')
            
            # Generate more points
            theta_new = np.linspace(min(theta), max(theta), 100)
            r_new = f(theta_new)
            
            # Convert back to cartesian
            interpolated[region] = [(r*np.cos(t) + self.com[0], r*np.sin(t) + self.com[1]) 
                                for r, t in zip(r_new, theta_new)]
        
        return interpolated
    
    def get_anatomical_boundary(self) -> List[Tuple[float, float, float]]:
        """Get the actual anatomical boundary of the disc"""
        print("Finding boundary faces...")
        node_faces = {}
        total_elements = len(self.elements[1]) // 4
        for i in range(0, len(self.elements[1]), 4):
            if i % 1000 == 0:
                print(f"Processing element {i//4}/{total_elements}")
            nodes = self.elements[1][i:i+4]
            faces = [
                tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[3]]))
            ]
            for face in faces:
                node_faces[face] = node_faces.get(face, 0) + 1

        # Get boundary nodes - now keeping all surface nodes
        boundary_nodes = set()
        for face, count in node_faces.items():
            if count == 1:  # This is a boundary face
                boundary_nodes.update(face)

        # Convert to coordinates maintaining all dimensions
        boundary_points = []
        for node in boundary_nodes:
            coords = self.coords[node-1]
            boundary_points.append((coords[0], coords[1], coords[2]))

        return boundary_points

    def point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm to determine if point is inside polygon"""
        x, y = point
        inside = False
        
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
                
        return inside

    def validate_mesh(self) -> None:
        """Validate mesh data for processing"""
        if self.coords is None or len(self.coords) == 0:
            raise ValueError("Mesh contains no nodes")
        
        if self.elements is None or len(self.elements[0]) == 0:
            raise ValueError("Mesh contains no tetrahedral elements")

        # Get all entities in the model
        entities = self.model.getEntities(3)  # Get all 3D entities
        entity_tags = set(tag for dim, tag in entities)
        
        # Print diagnostic information
        print("\nMesh Diagnostics:")
        print("-" * 50)
        print(f"Number of nodes: {len(self.coords)}")
        print(f"Number of tetrahedral elements: {len(self.elements[0])}")
        print(f"Element tag range: {min(self.elements[0])} to {max(self.elements[0])}")
        print(f"Number of 3D entities: {len(entities)}")
        print(f"Entity tag range: {min(entity_tags) if entity_tags else 'N/A'} to {max(entity_tags) if entity_tags else 'N/A'}")
        
        # Check element tag continuity
        element_tags = set(self.elements[0])
        gaps = []
        prev_tag = min(element_tags)
        for tag in sorted(element_tags)[1:]:
            if tag - prev_tag > 1:
                gaps.append((prev_tag, tag))
            prev_tag = tag
        
        if gaps:
            print("\nFound gaps in element numbering:")
            for start, end in gaps[:5]:  # Show first 5 gaps
                print(f"Gap between {start} and {end}")
            if len(gaps) > 5:
                print(f"... and {len(gaps)-5} more gaps")
        
        # Check for degenerate elements
        degenerate_count = 0
        for i in range(0, len(self.elements[1]), 4):
            nodes = self.elements[1][i:i+4]
            if len(set(nodes)) != 4:
                degenerate_count += 1
        
        if degenerate_count > 0:
            print(f"\nFound {degenerate_count} degenerate elements")

    def calculate_element_volumes(self) -> Dict[str, List[float]]:
        """Calculate volumes for elements in each region"""
        volumes = {region: [] for region in self.region_elements.keys()}
        
        for region, elements in self.region_elements.items():
            for elem_tag in elements:
                # Get node coordinates for this element
                elem_nodes = self.model.mesh.getElement(elem_tag)[1]
                node_coords = [self.coords[node_idx-1] for node_idx in elem_nodes]
                
                # Calculate tetrahedron volume
                matrix = np.vstack([np.array(node_coords[1:]) - np.array(node_coords[0])])
                volume = abs(np.linalg.det(matrix)) / 6.0
                volumes[region].append(volume)
                
        return volumes

    def verify_partitioning(self):
        """Verify that the mesh partitioning was successful with corrected element counting"""
        print("\nVerification Report:")
        print("-" * 50)
        
        # Get physical groups
        physical_groups = self.model.getPhysicalGroups()
        print(f"\nPhysical Groups Found: {len(physical_groups)}")
        
        for dim, tag in physical_groups:
            name = self.model.getPhysicalName(dim, tag)
            entities = self.model.getEntitiesForPhysicalGroup(dim, tag)
            print(f"\nGroup: {name}")
            print(f"Dimension: {dim}")
            print(f"Tag: {tag}")
            print(f"Number of entities: {len(self.region_elements[name])}")
            
            # Get elements directly from our stored region elements
            print(f"Total elements in group: {len(self.region_elements[name])}")
        
        # Cross-reference with original partitioning
        print("\nCross-referencing with original partitioning:")
        for region, elements in self.region_elements.items():
            print(f"{region}: {len(elements)} elements assigned during partitioning")

        # Verify total elements adds up correctly
        total_elements = sum(len(elements) for elements in self.region_elements.values())
        print(f"\nTotal elements across all regions: {total_elements}")
        print(f"Expected total elements: {len(self.elements[0])}")
        assert total_elements == len(self.elements[0]), "Element count mismatch"
        # Add volume-based validation
        total_elements = len(self.elements[0])
        print("\nVolume-based element distribution:")
        for region, elements in self.region_elements.items():
            percentage = len(elements) / total_elements * 100
            threshold = region.split('_')[-1] if '_' in region else 'nucleus'
            print(f"{region}: {percentage:.1f}% elements ({len(elements)} elements)")

        print("\nExpected distribution:")
        print(f"nucleus: 30%")
        print(f"Each annulus layer: ~10%")
        return len(physical_groups) == 8  # Updated to reflect 8 regions (nucleus + 7 layers)

    def print_region_statistics(self, volumes: Dict[str, List[float]]) -> None:
        """Print detailed statistics for each region"""
        print("\nRegion Statistics:")
        print("-" * 50)
        for region, elements in self.region_elements.items():
            if elements:
                region_volumes = volumes[region]
                total_volume = sum(region_volumes)
                avg_volume = np.mean(region_volumes)
                volume_std = np.std(region_volumes)
                
                print(f"\n{region.capitalize()}:")
                print(f"Number of elements: {len(elements)}")
                print(f"Total volume: {total_volume:.2f}")
                print(f"Average element volume: {avg_volume:.2f}")
                print(f"Volume standard deviation: {volume_std:.2f}")
                print(f"Percentage of total elements: {(len(elements)/len(self.elements[0])*100):.1f}%")

    def get_element_radii(self):
        """Get sorted list of elements by radius for layered distribution"""
        element_radii = []
        for i in range(0, len(self.elements[1]), 4):
            elem_nodes = self.elements[1][i:i+4]
            coords = [self.coords[idx-1] for idx in elem_nodes]
            centroid = np.mean(coords, axis=0)
            r = self.get_radial_distance(centroid[0], centroid[1], centroid[2], self.com)
            element_radii.append((r, self.elements[0][i//4]))
        return sorted(element_radii, reverse=True)

    def assign_layers_anatomically(self):
        """Assign elements to layers based on anatomical position with guaranteed boundary coverage"""
        print("Starting enhanced anatomical layer assignment...")
        
        # Get the anatomical boundary points
        boundary_points = self.get_anatomical_boundary()
        print(f"Found {len(boundary_points)} anatomical boundary points")
        
        # Calculate the boundary center of mass
        boundary_com = np.mean(boundary_points, axis=0)
        boundary_points_array = np.array(boundary_points)
        print(f"Calculated boundary center of mass: {boundary_com}")
        
        # Store element information
        element_info = []
        total_elements = len(self.elements[1]) // 4
        
        # First pass: identify boundary elements
        boundary_elements = set()
        max_distances = {}  # Store maximum distances for each angular sector
        
        # Divide space into angular sectors for better boundary detection
        n_sectors = 36  # 10-degree sectors
        sector_elements = {i: [] for i in range(n_sectors)}
        
        print("First pass: Identifying boundary elements...")
        
        for i in range(0, len(self.elements[1]), 4):
            if i % 1000 == 0:
                print(f"Processing element {i//4}/{total_elements}...")
                
            elem_nodes = self.elements[1][i:i+4]
            elem_coords = [self.coords[idx-1] for idx in elem_nodes]
            centroid = np.mean(elem_coords, axis=0)
            
            # Calculate angle and radius from center
            dx = centroid[0] - boundary_com[0]
            dy = centroid[1] - boundary_com[1]
            angle = math.atan2(dy, dx)
            angle = (angle + 2*math.pi) % (2*math.pi)  # Normalize to [0, 2π]
            
            # Determine sector
            sector = int(angle / (2*math.pi) * n_sectors)
            radius = np.sqrt(dx*dx + dy*dy)
            
            # Store element info with its radius
            sector_elements[sector].append((self.elements[0][i//4], radius, centroid))
        
        print("Second pass: Assigning layers based on radial position...")
        
        # Process each sector to identify layer boundaries
        for sector in range(n_sectors):
            if not sector_elements[sector]:
                continue
                
            # Sort elements in this sector by radius
            sector_elements[sector].sort(key=lambda x: x[1])
            
            # Get maximum radius for this sector
            max_radius = sector_elements[sector][-1][1]
            
            # Assign elements to layers based on relative position
            n_elements = len(sector_elements[sector])
            
            for idx, (elem_tag, radius, centroid) in enumerate(sector_elements[sector]):
                # Calculate normalized radial position
                radial_pos = radius / max_radius
                
                # Determine layer based on position
                if radial_pos < self.regions.nucleus_threshold:
                    layer = "nucleus"
                else:
                    # Scale remaining position across annulus layers
                    annulus_pos = (radial_pos - self.regions.nucleus_threshold) / (1 - self.regions.nucleus_threshold)
                    layer_idx = min(7, int(annulus_pos * 7) + 1)
                    layer = f"annulus_layer{layer_idx}"
                
                element_info.append((elem_tag, layer))
        
        print("Assigning elements to regions...")
        
        # Clear existing regions
        for region in self.region_elements:
            self.region_elements[region] = []
        
        # Assign elements to regions
        for elem_tag, layer in element_info:
            self.region_elements[layer].append(elem_tag)
        
        print("Layer assignment complete.")
        
        # Verify distribution
        total_elements = len(self.elements[0])
        print("\nFinal layer distribution:")
        for region, elements in self.region_elements.items():
            percentage = len(elements) / total_elements * 100
            print(f"{region}: {len(elements)} elements ({percentage:.1f}%)")

    def revise_layer_continuity_fine(self, n_bins: int = 360):
        """
        Revision step to enforce continuous outer boundaries using a fine angular resolution.
        For each fine angular bin, if the maximum assigned layer is less than annulus_layer7 (value 7),
        all elements in that bin are promoted upward by the offset needed.
        The procedure is iterated until no bin requires promotion.
        """
        import numpy as np

        # Define layer order and its inverse mapping.
        layer_order = {
            "nucleus": 0,
            "annulus_layer1": 1,
            "annulus_layer2": 2,
            "annulus_layer3": 3,
            "annulus_layer4": 4,
            "annulus_layer5": 5,
            "annulus_layer6": 6,
            "annulus_layer7": 7
        }
        inv_layer_order = {v: k for k, v in layer_order.items()}

        # Build a mapping from element tag to its index in the self.elements arrays.
        # (Assuming self.elements[0] contains element tags in order.)
        element_tag_to_index = {}
        num_elems = len(self.elements[0])
        for i, tag in enumerate(self.elements[0]):
            element_tag_to_index[tag] = i

        # Gather element data from the current region assignment.
        # Each entry will hold the tag, centroid, angle (in radians), and current layer (as int).
        element_data = []
        for region, elem_tags in self.region_elements.items():
            for tag in elem_tags:
                index = element_tag_to_index[tag]
                nodes = self.elements[1][index * 4:(index + 1) * 4]
                pts = np.array([self.coords[n - 1] for n in nodes])
                centroid = np.mean(pts, axis=0)
                dx = centroid[0] - self.com[0]
                dy = centroid[1] - self.com[1]
                angle = np.arctan2(dy, dx) % (2 * np.pi)
                element_data.append({
                    "tag": tag,
                    "centroid": centroid,
                    "angle": angle,
                    "layer_order": layer_order[region],
                    "current_layer": region
                })

        # Iterate until no updates occur.
        updated = True
        iteration = 0
        while updated:
            updated = False
            iteration += 1
            # Initialize bins: each bin will hold a list of element data entries.
            bins = {i: [] for i in range(n_bins)}
            bin_width = 2 * np.pi / n_bins

            # Assign each element to a bin based on its angle.
            for ed in element_data:
                bin_index = int(ed["angle"] / (2 * np.pi) * n_bins)
                if bin_index >= n_bins:
                    bin_index = n_bins - 1
                bins[bin_index].append(ed)

            # Process each bin.
            for i in range(n_bins):
                if not bins[i]:
                    continue  # If no elements fall into this fine bin, skip it.
                max_layer = max(ed["layer_order"] for ed in bins[i])
                if max_layer < 7:
                    offset = 7 - max_layer
                    for ed in bins[i]:
                        new_layer = min(ed["layer_order"] + offset, 7)
                        if new_layer != ed["layer_order"]:
                            ed["layer_order"] = new_layer
                            ed["current_layer"] = inv_layer_order[new_layer]
                            updated = True
                    # Debug: report which bin was promoted.
                    # You can comment out the following print if needed.
                    print(f"Iteration {iteration}: Bin {i} (angle range {i * bin_width:.2f}-{(i+1) * bin_width:.2f} rad) promoted by offset {offset}")

        # Rebuild the region mapping from the revised element data.
        revised_regions = {key: [] for key in layer_order.keys()}
        for ed in element_data:
            revised_regions[ed["current_layer"]].append(ed["tag"])
        self.region_elements = revised_regions

        # Print final revised distribution.
        total_elements = len(self.elements[0])
        print("\nRevised Element Distribution by Region (fine revision):")
        for region, elems in self.region_elements.items():
            percent = (len(elems) / total_elements) * 100
            print(f"{region}: {len(elems)} elements ({percent:.1f}%)")

    def smooth_annulus_boundaries(self, boundaries, smoothing_factor: float = 0.1, n_points: int = 360):
        """
        Smooth the boundaries for each annulus layer using a spline-based method.
        Accepts either a dictionary mapping region names to boundary point lists,
        or a single list of boundary points.
        """
        import numpy as np
        from scipy.interpolate import UnivariateSpline

        # If boundaries is a list, wrap it in a dict with a default key.
        if isinstance(boundaries, list):
            boundaries = {'boundary': boundaries}

        smoothed = {}
        center = self.com  # Using the center of mass as the reference.

        for region, pts in boundaries.items():
            if len(pts) < 3:
                smoothed[region] = pts
                continue

            pts_arr = np.array(pts)
            # Convert to polar coordinates relative to center.
            dx = pts_arr[:, 0] - center[0]
            dy = pts_arr[:, 1] - center[1]
            angles = np.arctan2(dy, dx)
            angles = np.mod(angles, 2 * np.pi)
            radii = np.sqrt(dx**2 + dy**2)

            # Sort the points by angle.
            sort_idx = np.argsort(angles)
            angles = angles[sort_idx]
            radii = radii[sort_idx]

            # Append the first point for periodicity.
            angles_ext = np.concatenate((angles, [angles[0] + 2 * np.pi]))
            radii_ext = np.concatenate((radii, [radii[0]]))

            # Create a smoothing spline.
            spline = UnivariateSpline(angles_ext, radii_ext, s=smoothing_factor * len(angles_ext))
            angles_new = np.linspace(angles_ext[0], angles_ext[-1], n_points)
            radii_new = spline(angles_new)

            # Convert back to Cartesian coordinates.
            x_new = center[0] + radii_new * np.cos(angles_new)
            y_new = center[1] + radii_new * np.sin(angles_new)
            smoothed[region] = list(zip(x_new, y_new))

        return smoothed

    def check_element_distribution(self):
        """Validate element distribution across regions"""
        total = len(self.elements[0])
        print("\nElement Distribution Check:")
        for region, elements in self.region_elements.items():
            count = len(elements)
            percent = (count/total) * 100
            expected = 30 if region == "nucleus" else 10
            print(f"{region}: {count} elements ({percent:.1f}%) - Expected ~{expected}%")

    def process(self) -> None:
        """Main processing function"""
        # Initialize Gmsh and reset everything
        try:
            gmsh.initialize(sys.argv)
            gmsh.clear()
            gmsh.model.add("disc")
            print("Initialized Gmsh with fresh model")
        except Exception as e:
            print(f"Error initializing Gmsh: {str(e)}")
            return
        
        try:
            # Load the VTK mesh
            try:
                gmsh.open(self.input_file)
            except Exception as e:
                raise RuntimeError(f"Error opening VTK file: {str(e)}")

            # Get model and mesh data
            self.model = gmsh.model
            nodes = self.model.mesh.getNodes()
            self.coords = np.array(nodes[1]).reshape(-1, 3)
            self.com = np.mean(self.coords, axis=0)
            
            # Get tetrahedral elements
            self.elements = self.model.mesh.getElementsByType(4)  # 4 is for tetrahedra
            
            # Validate mesh
            self.validate_mesh()
            
            print(f"\nMesh Analysis:")
            print("-" * 50)
            print(f"Total nodes: {len(self.coords)}")
            print(f"Center of mass: {self.com}")
            print(f"Total elements: {len(self.elements[0])}")
            
            # Calculate maximum radius for normalization
            for coord in self.coords:
                r = self.get_radial_distance(coord[0], coord[1], coord[2], self.com)
                self.max_radius = max(self.max_radius, r)
            
            print(f"Maximum radius: {self.max_radius}")

            # Get disc boundaries 
            if not hasattr(self, 'boundaries'):
                outer_boundary = self.get_disc_boundary()
                self.boundaries = {}  # We'll use a different approach for element assignment
                # Debug boundary creation
                for region, boundary in self.boundaries.items():
                    print(f"\n{region} boundary:")
                    print(f"Number of points: {len(boundary)}")
                    print(f"Min/Max X: {min(p[0] for p in boundary):.2f}/{max(p[0] for p in boundary):.2f}")
                    print(f"Min/Max Y: {min(p[1] for p in boundary):.2f}/{max(p[1] for p in boundary):.2f}")

            # Process elements
            # Use anatomical layer assignment instead of radial-only
            self.assign_layers_anatomically()

            # Add after element processing loop and before create_physical_groups
            self.check_element_distribution()

            # Revision step to enforce continuity in outer layers.
            self.revise_layer_continuity_fine(n_bins=360)

            outer_boundary = self.get_disc_boundary()  # returns a list
            smoothed_boundaries = self.smooth_annulus_boundaries(outer_boundary, smoothing_factor=0.1, n_points=360)
        
            # Print statistics after processing
            print("\nElement distribution:")
            total_elements = sum(len(elements) for elements in self.region_elements.values())
            for region, elements in self.region_elements.items():
                percentage = (len(elements) / total_elements) * 100
                print(f"{region}: {len(elements)} elements ({percentage:.1f}%)")

            # Calculate volumes and print statistics
            volumes = self.calculate_element_volumes()
            self.print_region_statistics(volumes)
            
            # First ensure we remove any existing entities
            print("\nCleaning up existing geometry...")
            try:
                for dim, tag in self.model.getEntities():
                    self.model.removeEntities([(dim, tag)])
                print("Removed all existing entities")
            except Exception as e:
                print(f"Warning during cleanup: {str(e)}")
            
            # Now create our new geometry
            print("\nCreating geometry...")
            try:
                # Create a single volume entity
                vol_tag = 1
                self.model.addDiscreteEntity(3, vol_tag)
                print("Created volume entity")
                
                # Add all nodes at once
                node_tags = list(range(1, len(self.coords) + 1))
                node_coords = []
                for coord in self.coords:
                    node_coords.extend([float(x) for x in coord])  # Convert to float
                
                self.model.mesh.addNodes(3, vol_tag, node_tags, node_coords)
                print(f"Added {len(node_tags)} nodes")
                
                # Process elements in batches
                batch_size = 1000
                total_elements = len(self.elements[0])
                elements_added = 0
                
                # First convert all node indices to standard Python integers
                node_indices = []
                for i in range(0, len(self.elements[1]), 4):
                    tet_nodes = [int(n) for n in self.elements[1][i:i+4]]
                    node_indices.extend(tet_nodes)  # Extend rather than append to create flat list
                
                for batch_start in range(0, total_elements, batch_size):
                    batch_end = min(batch_start + batch_size, total_elements)
                    batch_size_actual = batch_end - batch_start
                    
                    # Create flat arrays for this batch
                    element_tags = list(range(batch_start + 1, batch_end + 1))
                    batch_nodes = node_indices[batch_start*4:batch_end*4]
                    
                    # Add elements for this batch
                    try:
                        self.model.mesh.addElementsByType(
                            vol_tag,           # Volume tag
                            4,                 # Element type (4 = tetrahedron)
                            element_tags,      # Element tags
                            batch_nodes        # Node tags (flat array)
                        )
                        elements_added += batch_size_actual
                        print(f"Added elements {batch_start + 1} to {batch_end} ({elements_added}/{total_elements})")
                    except Exception as e:
                        print(f"Error adding batch {batch_start}-{batch_end}: {str(e)}")
                        print(f"Sample element: Type=4, Tag={batch_start + 1}, First nodes={batch_nodes[:4]}")
                        raise
                
                print(f"Successfully added {elements_added} tetrahedral elements")
                
            except Exception as e:
                print(f"Error creating geometry: {str(e)}")
                raise
                
            # Create 3D entities for elements first
            print("\nCreating 3D entities...")
            total = len(self.elements[0])
            created_entities = 0
            failed_entities = []
            
            # Get all element data at once for efficiency
            element_types = self.elements[0]  # Already have these from getElementsByType
            element_nodes = self.elements[1]
            
            # Process in groups of 4 nodes (tetrahedra)
            # Process in groups of 4 nodes (tetrahedra)
            for i in range(len(element_types)):
                if i % 1000 == 0:
                    print(f"Progress: {i}/{total} elements processed ({(i/total*100):.1f}%)")
                try:
                    element_tag = element_types[i]
                    node_indices = element_nodes[i*4:(i+1)*4]  # Get 4 nodes for this tetrahedron
                    
                    # Use the element tag as the discrete entity tag
                    entity_tag = element_tag
                    
                    # Try to create a new discrete entity. If it already exists, skip creation.
                    try:
                        self.model.addDiscreteEntity(3, entity_tag)
                    except Exception as e:
                        if "already exists" in str(e):
                            print(f"Entity {entity_tag} already exists, skipping creation.")
                        else:
                            raise e
                    
                    # Now add the element to this entity (4 represents the tetrahedron element type)
                    try:
                        self.model.mesh.addElements(3, entity_tag, [4], [[element_tag]], [node_indices])
                        created_entities += 1
                    except Exception as e:
                        failed_entities.append((element_tag, str(e)))
                        if len(failed_entities) <= 5:  # Only print first 5 failures
                            print(f"\nError creating entity {element_tag}: {str(e)}")
                            
                except Exception as e:
                    print(f"\nError processing element {i}: {str(e)}")

            
            print(f"\nCreated {created_entities} entities out of {total} elements")
            if failed_entities:
                print(f"Failed to create {len(failed_entities)} entities:")
                for tag, error in failed_entities[:5]:
                    print(f"  Entity {tag}: {error}")
                if len(failed_entities) > 5:
                    print(f"  ... and {len(failed_entities)-5} more errors")
            
            # Now create physical groups
            print("\nPhysical Group Analysis:")
            print("-" * 50)
            
            # Get all valid entities
            entities = self.model.getEntities(3)
            valid_entities = set(tag for dim, tag in entities)
            
            self.create_physical_groups()
            self.verify_partitioning()

            # Write the partitioned mesh
            try:
                gmsh.write("partitioned_disc.msh")
                print("\nSuccessfully wrote partitioned mesh to partitioned_disc.msh")
            except Exception as e:
                print(f"\nError writing mesh file: {e}")

        finally:
            # Finalize Gmsh
            gmsh.finalize()

def main():
    # Configure regions with 7 layers for the annulus
    regions = DiscRegions(
        nucleus_threshold=0.3,    # Keep nucleus the same
        layer1_threshold=0.4,
        layer2_threshold=0.5,
        layer3_threshold=0.6,
        layer4_threshold=0.7,
        layer5_threshold=0.8,
        layer6_threshold=0.9,
        layer7_threshold=1.0
    )
    
    # Create processor and run
    try:
        processor = MeshProcessor("new_prefix_disc_1.vtk", regions)
        processor.process()
    except Exception as e:
        print(f"Error processing mesh: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()