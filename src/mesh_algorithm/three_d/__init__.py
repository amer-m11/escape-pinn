"""3D physical domain (5D phase space) mesh solver.

The 3D path is built in steps; concrete classes (``PhaseSpaceGrid3D``,
``CharacteristicResult3D``, ``local_cell_characteristic_3d``,
``interpolate_on_face_3d``) live in ``grid.py``, ``update.py``, and
``face.py``. This file is intentionally empty so importing the package
does not pull in 3D code until callers explicitly need it.
"""
