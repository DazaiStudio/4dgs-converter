"""Binary little-endian PLY reader.

Mirrors the logic of FThreeDGaussiansPlyReader from the UE plugin.
Only reads the 'vertex' element; skips other elements (faces, etc.).
Supports dynamic property mapping so any PLY layout works.
"""

import struct
import numpy as np

# Map PLY type names to (struct format char, byte size, numpy dtype)
_TYPE_MAP = {
    "char": ("b", 1, np.int8),
    "uchar": ("B", 1, np.uint8),
    "short": ("h", 2, np.int16),
    "ushort": ("H", 2, np.uint16),
    "int": ("i", 4, np.int32),
    "uint": ("I", 4, np.uint32),
    "float": ("f", 4, np.float32),
    "double": ("d", 8, np.float64),
    # Alternative names sometimes found in PLY files
    "int8": ("b", 1, np.int8),
    "uint8": ("B", 1, np.uint8),
    "int16": ("h", 2, np.int16),
    "uint16": ("H", 2, np.uint16),
    "int32": ("i", 4, np.int32),
    "uint32": ("I", 4, np.uint32),
    "float32": ("f", 4, np.float32),
    "float64": ("d", 8, np.float64),
}


class PlyProperty:
    __slots__ = ("name", "type_name", "size", "offset", "dtype")

    def __init__(self, name: str, type_name: str, offset: int):
        if type_name not in _TYPE_MAP:
            raise ValueError(f"Unsupported PLY property type: {type_name}")
        info = _TYPE_MAP[type_name]
        self.name = name
        self.type_name = type_name
        self.size = info[1]
        self.offset = offset
        self.dtype = info[2]


class PlyReader:
    """Reads binary little-endian PLY files.

    Usage::

        reader = PlyReader("file.ply")
        positions = reader.get_property_array("x")  # shape (N,)
    """

    def __init__(self, file_path: str):
        self.vertex_count: int = 0
        self._vertex_stride: int = 0
        self._properties: list[PlyProperty] = []
        self._property_map: dict[str, PlyProperty] = {}
        self._data: bytes = b""
        self._load(file_path)

    def _load(self, file_path: str):
        with open(file_path, "rb") as f:
            self._parse_header(f)
            data_size = self.vertex_count * self._vertex_stride
            self._data = f.read(data_size)
            if len(self._data) != data_size:
                raise IOError(
                    f"Expected {data_size} bytes of vertex data, got {len(self._data)}"
                )

    def _parse_header(self, f):
        in_vertex = False
        other_element_stride = 0
        other_element_count = 0

        while True:
            line = f.readline()
            if not line:
                raise IOError("Unexpected end of file in PLY header")

            line = line.decode("ascii", errors="ignore").strip()

            if line.startswith("comment"):
                continue

            if line == "end_header":
                break

            parts = line.split()
            keyword = parts[0]

            if keyword == "element":
                element_type = parts[1]
                count = int(parts[2])
                if element_type == "vertex":
                    self.vertex_count = count
                    in_vertex = True
                else:
                    in_vertex = False
                    other_element_count = count
                    other_element_stride = 0

            elif keyword == "property":
                if parts[1] == "list":
                    # List properties not supported for vertex element
                    continue
                type_name = parts[1]
                prop_name = parts[2]
                if type_name not in _TYPE_MAP:
                    raise ValueError(f"Unsupported type: {type_name}")
                type_size = _TYPE_MAP[type_name][1]

                if in_vertex:
                    prop = PlyProperty(prop_name, type_name, self._vertex_stride)
                    self._properties.append(prop)
                    self._property_map[prop_name] = prop
                    self._vertex_stride += type_size

        if self.vertex_count == 0 or not self._properties:
            raise IOError("No vertex element or properties found in PLY header")

    def has_property(self, name: str) -> bool:
        return name in self._property_map

    def get_property_array(self, name: str) -> np.ndarray:
        """Return a 1-D numpy array of shape (vertex_count,) for the named property."""
        prop = self._property_map.get(name)
        if prop is None:
            raise KeyError(f"Property '{name}' not found in PLY. "
                           f"Available: {list(self._property_map.keys())}")
        # Use numpy structured view for efficient extraction
        buf = np.frombuffer(self._data, dtype=np.uint8)
        buf = buf.reshape(self.vertex_count, self._vertex_stride)
        col = buf[:, prop.offset:prop.offset + prop.size]
        return np.frombuffer(col.tobytes(), dtype=prop.dtype)

    def get_properties_array(self, names: list[str]) -> np.ndarray:
        """Return a 2-D numpy array of shape (vertex_count, len(names)).

        All requested properties must have the same dtype.
        """
        arrays = []
        for name in names:
            arrays.append(self.get_property_array(name))
        return np.column_stack(arrays)

    @property
    def property_names(self) -> list[str]:
        return [p.name for p in self._properties]


def load_gaussian_ply(file_path: str) -> dict[str, np.ndarray]:
    """Load a 3DGS PLY file and return arrays matching the UE struct layout.

    Returns a dict with keys:
        position  : (N, 3) float32  - x, y, z (COLMAP coords)
        sh_dc     : (N, 3) float32  - f_dc_0, f_dc_1, f_dc_2
        sh_rest_r : (N, 15) float32 - f_rest_0..14
        sh_rest_g : (N, 15) float32 - f_rest_15..29
        sh_rest_b : (N, 15) float32 - f_rest_30..44
        opacity   : (N,) float32
        scale     : (N, 3) float32  - scale_0, scale_1, scale_2
        rotation  : (N, 4) float32  - rot_0, rot_1, rot_2, rot_3
    """
    reader = PlyReader(file_path)
    n = reader.vertex_count

    result = {}

    # Position
    result["position"] = reader.get_properties_array(["x", "y", "z"]).astype(np.float32)

    # SH DC
    result["sh_dc"] = reader.get_properties_array(
        ["f_dc_0", "f_dc_1", "f_dc_2"]
    ).astype(np.float32)

    # SH rest (R, G, B channels) - 15 coefficients each
    # Check how many SH rest coefficients exist
    sh_rest_r = []
    sh_rest_g = []
    sh_rest_b = []
    for i in range(15):
        r_name = f"f_rest_{i}"
        g_name = f"f_rest_{i + 15}"
        b_name = f"f_rest_{i + 30}"
        if reader.has_property(r_name):
            sh_rest_r.append(reader.get_property_array(r_name))
        else:
            sh_rest_r.append(np.zeros(n, dtype=np.float32))
        if reader.has_property(g_name):
            sh_rest_g.append(reader.get_property_array(g_name))
        else:
            sh_rest_g.append(np.zeros(n, dtype=np.float32))
        if reader.has_property(b_name):
            sh_rest_b.append(reader.get_property_array(b_name))
        else:
            sh_rest_b.append(np.zeros(n, dtype=np.float32))

    result["sh_rest_r"] = np.column_stack(sh_rest_r).astype(np.float32)
    result["sh_rest_g"] = np.column_stack(sh_rest_g).astype(np.float32)
    result["sh_rest_b"] = np.column_stack(sh_rest_b).astype(np.float32)

    # Opacity
    result["opacity"] = reader.get_property_array("opacity").astype(np.float32)

    # Scale
    result["scale"] = reader.get_properties_array(
        ["scale_0", "scale_1", "scale_2"]
    ).astype(np.float32)

    # Rotation quaternion
    result["rotation"] = reader.get_properties_array(
        ["rot_0", "rot_1", "rot_2", "rot_3"]
    ).astype(np.float32)

    return result
