import numpy as np

def interpolate_path(nodes, path_node_names, points_per_segment=20):
    """
    Generates a high-resolution path with interpolated points between major nodes.

    Args:
        nodes (dict): Dictionary of node names to (x, y) coordinates.
        path_node_names (list): List of node names defining the path.
        points_per_segment (int): Number of points to generate for each track segment.

    Returns:
        np.ndarray: An array of (x, y) coordinates representing the smooth path.
    """
    high_res_path = []
    
    for i in range(len(path_node_names) - 1):
        start_node_name = path_node_names[i]
        end_node_name = path_node_names[i+1]
        
        start_pos = np.array(nodes[start_node_name], dtype=float)
        end_pos = np.array(nodes[end_node_name], dtype=float)
        
        # Add interpolated points for the segment
        for j in range(points_per_segment):
            alpha = j / points_per_segment
            point = start_pos + alpha * (end_pos - start_pos)
            high_res_path.append(point)
            
    # Ensure the final node is included
    if path_node_names:
        high_res_path.append(np.array(nodes[path_node_names[-1]], dtype=float))
        
    return np.array(high_res_path)
