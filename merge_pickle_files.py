import timescape_functions as tf
import os
import sys
from pathlib import Path

source_folder = sys.argv[1]
filename = sys.argv[2]
out_folder = sys.argv[3]

try:
    tf.merge_pickles(source_folder, filename, out_folder)
except Exception as e:
    print(f'Error creating {filename}:\n{e}')