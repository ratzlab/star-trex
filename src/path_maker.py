"""
Functions to handle paths
"""

from pathlib import Path


def path_maker(path_str):
    """
    Transforms a path string into a Path object
    """
    return Path(path_str)


def path_checker(path):
    """
    Checks if a given path is a string, if it leads to a file and if the file already exists.
    Returns a Path object if no error was raised
    """
    # Check if path is a string
    if not isinstance(path, str):
        raise ValueError(f"The path '{path}' must be provided as string.")
    
    return path_maker(path)