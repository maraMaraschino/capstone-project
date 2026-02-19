# FITS files are downloaded to separate folders. Testing function to merge final folders
import os
import shutil
from pathlib import Path

def merge_folders(top_folder, merge_folder):
    """
    Select every folder starting with FITS_ and move all files to "FITS" folder.
    """
    top_path = Path(top_folder)
    merge_path = Path(merge_folder)
    dir_list = [dir for dir in os.listdir(top_path) if dir.startswith("FITS_")]
    for dir in dir_list:
        print(f"Moving {dir}")
        dir_path = Path(dir)
        num_files = len(os.listdir(dir_path))
        print(f'Moving {num_files} files...')
        for file in os.listdir(dir_path):
            old_path = dir_path / file
            new_path = merge_path / file
            shutil.move(old_path, new_path)
        print(f'Finished moving {dir} to {merge_folder}. Deleting empty folder...')
        Path.rmdir(dir)
        print('Deleted!')
    print("Done!")


merge_folders(".", "FITS")