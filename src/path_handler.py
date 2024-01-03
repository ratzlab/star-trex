"""
Functions to handle paths
"""

from pathlib import Path
import os


def path_maker(path_str):
    """
    Transforms a path string into a Path object
    """
    return Path(path_str)


def path_checker(path_str, exists=True, directory=True, json=False):
    """
    Checks path for several criteria depending on arguments. Returns
    error or, if all checks have been passed, a Path object
    """
    # Assert that path_string is a string
    assert isinstance(path_str, str), f"The provided path '{path_str}' is not a string."

    path_obj = path_maker(path_str)

    # Check if the path exists
    if exists:
        assert path_obj.exists(), f"The path '{path_str}' does not exist."

    # Check if the path is a directory
    if json:
        assert path_str.endswith(".json"), f"The path '{path_str}' does not lead to a json file."
    elif directory:
        assert path_obj.is_dir(), f"The path '{path_str}' is not a directory."
    else:
        # Check if the path is a file
        assert path_obj.is_file(), f"The path '{path_str}' is not a file."

    return path_obj


def get_files_in_path(folder_path, extension=None):
    """
    Extracts all file paths OR all file paths with a given extension
    from the input folder.
    """

    files_list = []
    for file in os.listdir(folder_path):
        if extension:
            if file.endswith(extension):
                files_list.append(os.path.join(folder_path, file))
        else:
            files_list.append(os.path.join(folder_path, file))
    return files_list


def get_folders_in_path(input_path):
    """
    Extracts all folder paths from the input folder.
    """

    folder_paths = [input_path / folder for folder in input_path.iterdir() if folder.is_dir()]
    return folder_paths
