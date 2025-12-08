import os

def find_image_case_insensitive(basename):
    """
    Find an image file with case-insensitive matching.
    
    Args:
        basename (str): The base filename to search for (e.g., "image.png")
    
    Returns:
        str or None: Full path to the file if found, None otherwise
    """
    folder = os.path.dirname(os.path.abspath(__file__))
    try:
        for fname in os.listdir(folder):
            if fname.lower() == basename.lower():
                return os.path.join(folder, fname)
    except:
        pass
    return None
