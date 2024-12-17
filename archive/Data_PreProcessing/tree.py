import os

def list_folders_and_file_count(root, max_depth):
    for dirpath, dirnames, filenames in os.walk(root):
        # Calculate the depth
        depth = dirpath[len(root):].count(os.sep)
        
        if depth < max_depth:
            print(' ' * 4 * depth + os.path.basename(dirpath))  # Print folder names with indentation
        
        # If it's a leaf directory (no subdirectories)
        if not dirnames and depth < max_depth:
            print(' ' * 4 * (depth + 1) + f"Number of files: {len(filenames)}")
        
        # Stop walking deeper than the specified depth
        if depth >= max_depth - 1:
            del dirnames[:]  # Prevent os.walk from going deeper

# Example usage
list_folders_and_file_count('../Dataset', 4)  # Lists folders up to depth 4 and shows number of files in leaf directories
