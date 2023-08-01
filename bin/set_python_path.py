"""
A hacky script that is imported by all files in bin/. Ensures that the
relevant modules from this repository are in the path without needing to
actually install the repository in any sense.

Basically, since the script directory is always a part of the system
path, this module is always accessible to the scripts in bin/ no matter
where they are called from. Once imported, this script then adds the
parent directory to the system path thus making the simulation and util
modules and submodules accessible
"""

import sys
import os

script_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(script_dir, '..')))
