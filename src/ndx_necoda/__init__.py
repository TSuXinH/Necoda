# src/ndx_necoda/__init__.py

import os
from pynwb import load_namespaces

# 1. Find and load the namespace specification file
# This code block reliably finds the YAML file regardless of where the package is installed.
try:
    # __file__ is available in regular Python environments
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
except NameError:
    # __file__ may not be available in interactive environments (e.g., Jupyter)
    __location__ = os.getcwd()

# Get the absolute path to the .namespace.yaml file
spec_path = os.path.join(__location__, "spec", "ndx-necoda.namespace.yaml")

# load_namespaces is a key step. It registers the extension with the pynwb framework,
# allowing it to recognize your custom data types when reading/writing NWB files.
load_namespaces(spec_path)


# 2. Import the main class(es) from your modules for easy user access
# The '.' makes it a relative import from the 'container.py' file in the same package.
from .container import NecodaContainer, generate_pth_embedding, generate_network_embedding

# (Optional but recommended) Define the public API of the package.
# This tells Python what names to export when a user does 'from ndx_necoda import *'.
__all__ = ["NecodaContainer", "generate_pth_embedding", "generate_network_embedding"]
