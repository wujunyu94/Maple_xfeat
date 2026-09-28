"""Process-local compatibility for classic WZ convex vector lists.

The bundled reader expects an extra property tag inside Convex2D. These
extracted assets store an extended type string followed directly by x/y.
Only the lab process installs this hook; no shared source file is changed.
"""
from wzpy import properties


def install():
    original = properties._parse_extended
    if getattr(original, "_lab_convex", False):
        return

    def extended(reader, base_offset, name, ext_type, parent, wz_image, end_pos):
        if ext_type != "Shape2D#Convex2D":
            return original(reader, base_offset, name, ext_type, parent, wz_image, end_pos)
        result = properties.WzConvexProperty(name, parent)
        count = reader.read_compressed_int()
        if count < 0 or count > 100000:
            raise ValueError("Invalid Convex2D point count")
        for i in range(count):
            vector_type = reader.read_string_block(base_offset)
            if vector_type != "Shape2D#Vector2D":
                raise ValueError(f"Unsupported convex member: {vector_type}")
            point = properties.WzVectorProperty(str(i), reader.read_compressed_int(),
                                                reader.read_compressed_int(), result)
            result.points.append(point)
        return result

    extended._lab_convex = True
    properties._parse_extended = extended
