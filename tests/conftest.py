import sys
import os

# Ensure local package is importable when running tests from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
