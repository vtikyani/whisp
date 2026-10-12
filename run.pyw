"""Entry point for Whisp. Use pythonw.exe to run without console window."""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisp.main import main

main()
