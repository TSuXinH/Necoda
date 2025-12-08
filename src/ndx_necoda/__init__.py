import os
from pynwb import load_namespaces

try:
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
except NameError:
    __location__ = os.getcwd()

spec_path = os.path.join(__location__, "spec", "ndx-necoda.namespace.yaml")

load_namespaces(spec_path)


from .container import NecodaContainer, generate_pth_embedding, generate_network_embedding

__all__ = ["NecodaContainer", "generate_pth_embedding", "generate_network_embedding"]
