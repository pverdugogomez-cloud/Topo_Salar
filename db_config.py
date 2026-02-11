import os

def get_db_path(filename):
    """
    Smart path detection for database files.
    Priority:
    1. Parent Directory (Likely Cloud/Drive root if running from 'versión web' subfolder)
    2. Current Directory (Standalone deployment or GitHub repo)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Path in Parent
    path_parent = os.path.join(parent_dir, filename)
    
    # Path in Current
    path_current = os.path.join(current_dir, filename)
    
    # If file exists in parent, USE IT (Preserves Drive Data)
    if os.path.exists(path_parent):
        return path_parent
    
    # Otherwise use current (fallback or fresh install)
    return path_current
