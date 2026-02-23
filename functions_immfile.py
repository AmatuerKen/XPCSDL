import os
import struct
import numpy as np
import mmap

def _read_int32(fid, offset):
    """Read little-endian int32 from absolute byte offset."""
    fid.seek(offset, os.SEEK_SET)
    return struct.unpack('<i', fid.read(4))[0]


def index_compressed_multi_imm(filename, frame_indices):
    """
    Find image start indices in a (possibly compressed) IMM multifile.

    Parameters
    ----------
    filename : str
        IMM file name
    frame_indices : array-like of int
        Frames to process (0-based indexing)

    Returns
    -------
    imagestartindex : np.ndarray
        Starting byte offset of each requested frame
    dlen : np.ndarray
        Data length associated with each requested frame
    """

    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.min() < 0:
        raise ValueError("frame_indices must be >= 0")

    max_frame = frame_indices.max()
    nframes = max_frame + 1

    with open(filename, "rb") as fid:

        # ------------------------------------------------------------
        # Check mode, compression & IMM version
        # ------------------------------------------------------------
        modeflag = _read_int32(fid, 0)
        compressionflag = _read_int32(fid, 4)
        immversionflag = _read_int32(fid, 616)

        if immversionflag >= 11 and modeflag != 2:
            raise RuntimeError(
                f"Unknown mode: seems neither compressed nor Raw {filename}"
            )

        if immversionflag < 11 and "-" not in filename:
            raise RuntimeError("Not a multifile by naming convention")

        compression = (compressionflag == 6 and immversionflag >= 11)

        # ------------------------------------------------------------
        # Common header fields
        # ------------------------------------------------------------
        bytes_per_pixel = _read_int32(fid, 116)

        imagestart_all = np.zeros(nframes, dtype=np.int64)
        dlen_all = np.zeros(nframes, dtype=np.int64)

        # ------------------------------------------------------------
        # Compressed IMM
        # ------------------------------------------------------------
        if compression:

            # Frame 0
            imagestart_all[0] = 0
            dlen_all[0] = _read_int32(fid, 152)

            for k in range(1, nframes):
                # Read dlen of previous frame
                fid.seek(imagestart_all[k - 1] + 152, os.SEEK_SET)
                dlen_prev = struct.unpack("<i", fid.read(4))[0]

                dlen_all[k - 1] = dlen_prev
                imagestart_all[k] = (
                    imagestart_all[k - 1]
                    + 1024
                    + dlen_prev * (4 + bytes_per_pixel)
                )

            # Last frame dlen
            fid.seek(imagestart_all[-1] + 152, os.SEEK_SET)
            dlen_all[-1] = struct.unpack("<i", fid.read(4))[0]

        # ------------------------------------------------------------
        # Raw (uncompressed) IMM
        # ------------------------------------------------------------
        else:
            dlen0 = _read_int32(fid, 152)
            frame_size = 1024 + dlen0 * bytes_per_pixel

            imagestart_all = np.arange(nframes, dtype=np.int64) * frame_size
            dlen_all[:] = dlen0

        # ------------------------------------------------------------
        # Select requested frames (already 0-based)
        # ------------------------------------------------------------
        imagestartindex = imagestart_all[frame_indices]
        dlen = dlen_all[frame_indices]

        return imagestartindex, dlen



# ------------------------------------------------------------
# Low-level readers (little-endian)
# ------------------------------------------------------------
def read_int(fid, offset=None):
    if offset is not None:
        fid.seek(offset, os.SEEK_SET)
    return struct.unpack('<i', fid.read(4))[0]


def read_uint32(fid, offset=None):
    if offset is not None:
        fid.seek(offset, os.SEEK_SET)
    return struct.unpack('<I', fid.read(4))[0]


def read_double(fid, offset=None):
    if offset is not None:
        fid.seek(offset, os.SEEK_SET)
    return struct.unpack('<d', fid.read(8))[0]


def read_bytes(fid, n, offset=None):
    if offset is not None:
        fid.seek(offset, os.SEEK_SET)
    return fid.read(n)


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def openmultiimm(filename, frame_indices, image_start_bytes):
    """
    Open frames from an IMM multifile (Python-native, 0-based indexing).

    Parameters
    ----------
    filename : str
        IMM file path
    frame_indices : array-like (0-based)
        Frame indices to read
    image_start_bytes : array-like
        Byte offsets for each requested frame

    Returns
    -------
    out : list of dict
        Each dict contains:
            - 'header' : dict
            - 'imm'    : 2D numpy array (float32)
    """

    frame_indices = np.asarray(frame_indices, dtype=int)
    image_start_bytes = np.asarray(image_start_bytes, dtype=np.int64)

    out = []

    with open(filename, "rb") as fid:

        # --------------------------------------------------------
        # Global IMM checks
        # --------------------------------------------------------
        modeflag = read_int(fid, 0)
        compressionflag = read_int(fid, 4)
        immversionflag = read_int(fid, 616)

        if immversionflag >= 11 and modeflag != 2:
            raise RuntimeError("Not a multifile IMM")

        compression = (compressionflag == 6 and immversionflag >= 11)

        # --------------------------------------------------------
        # Loop over frames
        # --------------------------------------------------------
        for frame_idx, image_start in zip(frame_indices, image_start_bytes):

            # ----------------------------------------------------
            # Read header
            # ----------------------------------------------------
            h = {}

            fid.seek(image_start, os.SEEK_SET)
            h["mode"] = read_int(fid)
            h["compression"] = read_int(fid)
            h["date"] = read_bytes(fid, 32)

            fid.seek(image_start + 84, os.SEEK_SET)
            h["row_beg"] = read_int(fid)
            h["row_end"] = read_int(fid)
            h["col_beg"] = read_int(fid)
            h["col_end"] = read_int(fid)
            h["row_bin"] = read_int(fid)
            h["col_bin"] = read_int(fid)
            h["rows"] = read_int(fid)
            h["cols"] = read_int(fid)
            h["bytes"] = read_int(fid)
            h["kinetics"] = read_int(fid)
            h["kinwinsize"] = read_int(fid)
            h["elapsed"] = read_double(fid)
            h["preset"] = read_double(fid)

            fid.seek(image_start + 152, os.SEEK_SET)
            h["dlen"] = read_int(fid)
            h["roi_number"] = read_int(fid)
            h["buffer_number"] = read_uint32(fid)
            h["systick"] = read_uint32(fid)

            fid.seek(image_start + 616, os.SEEK_SET)
            h["immversion"] = read_int(fid)
            h["corecotick"] = read_uint32(fid)

            rows = h["rows"]
            cols = h["cols"]
            bytes_per_pixel = h["bytes"]

            # ----------------------------------------------------
            # Read image data
            # ----------------------------------------------------
            if not compression:
                fid.seek(image_start + 1024, os.SEEK_SET)

                if bytes_per_pixel == 2:
                    img = np.fromfile(
                        fid, dtype=np.uint16, count=rows * cols
                    ).astype(np.float32)
                elif bytes_per_pixel == 4:
                    img = np.fromfile(
                        fid, dtype=np.uint32, count=rows * cols
                    ).astype(np.float32)
                else:
                    raise RuntimeError("Unsupported bytes per pixel")

                img = img.reshape((rows, cols))

            else:
                # ---------------- compressed IMM ----------------
                pixel_number = int(h["dlen"])

                fid.seek(image_start + 1024, os.SEEK_SET)

                pixel_index = np.fromfile(
                    fid, dtype=np.uint32, count=pixel_number
                )  # already 0-based

                if bytes_per_pixel == 2:
                    pixel_value = np.fromfile(
                        fid, dtype=np.uint16, count=pixel_number
                    ).astype(np.float32)
                elif bytes_per_pixel == 4:
                    pixel_value = np.fromfile(
                        fid, dtype=np.uint32, count=pixel_number
                    ).astype(np.float32)
                else:
                    raise RuntimeError("Unsupported bytes per pixel")

                img = np.zeros((rows, cols), dtype=np.float32)
                img.flat[pixel_index] = pixel_value

            out.append({
                "header": h,
                "imm": img
            })

    return out


# ------------------------------------------------------------
# Low-level unpack helpers (little-endian)
# ------------------------------------------------------------
def unpack_int(buf, offset):
    return struct.unpack_from("<i", buf, offset)[0]


def unpack_uint32(buf, offset):
    return struct.unpack_from("<I", buf, offset)[0]


def unpack_double(buf, offset):
    return struct.unpack_from("<d", buf, offset)[0]


# ------------------------------------------------------------
# IMM mmap reader
# ------------------------------------------------------------
class IMMMmapReader:
    """
    Fast IMM reader using memory mapping (0-based indexing).
    """

    def __init__(self, filename):
        self.filename = filename
        self._open_and_map()

    def _open_and_map(self):
        self._f = open(self.filename, "rb")
        self.mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        # Global header
        self.modeflag = unpack_int(self.mm, 0)
        self.compressionflag = unpack_int(self.mm, 4)
        self.immversion = unpack_int(self.mm, 616)

        if self.immversion >= 11 and self.modeflag != 2:
            raise RuntimeError("Not a multifile IMM")

        self.compressed = (self.compressionflag == 6 and self.immversion >= 11)

    def close(self):
        self.mm.close()
        self._f.close()

    # --------------------------------------------------------
    # Read a single frame
    # --------------------------------------------------------
    def read_frame(self, image_start):
        """
        Read one frame given its starting byte offset.
        """
        mm = self.mm

        # ---------------- header ----------------
        h = {}
        h["rows"] = unpack_int(mm, image_start + 108)
        h["cols"] = unpack_int(mm, image_start + 112)
        h["bytes"] = unpack_int(mm, image_start + 116)
        h["dlen"] = unpack_int(mm, image_start + 152)

        rows = h["rows"]
        cols = h["cols"]
        bpp = h["bytes"]

        # ---------------- image ----------------
        data_offset = image_start + 1024

        if not self.compressed:
            n = rows * cols
            if bpp == 2:
                img = np.frombuffer(
                    mm, dtype=np.uint16, count=n, offset=data_offset
                ).astype(np.float32)
            elif bpp == 4:
                img = np.frombuffer(
                    mm, dtype=np.uint32, count=n, offset=data_offset
                ).astype(np.float32)
            else:
                raise RuntimeError("Unsupported bytes per pixel")

            img = img.reshape((rows, cols))

        else:
            pixel_number = h["dlen"]

            pixel_index = np.frombuffer(
                mm, dtype=np.uint32, count=pixel_number, offset=data_offset
            )

            value_offset = data_offset + pixel_number * 4

            if bpp == 2:
                pixel_value = np.frombuffer(
                    mm, dtype=np.uint16, count=pixel_number, offset=value_offset
                ).astype(np.float32)
            elif bpp == 4:
                pixel_value = np.frombuffer(
                    mm, dtype=np.uint32, count=pixel_number, offset=value_offset
                ).astype(np.float32)
            else:
                raise RuntimeError("Unsupported bytes per pixel")

            img = np.zeros((rows, cols), dtype=np.float32)
            img.flat[pixel_index] = pixel_value

        return img, h

    # --------------------------------------------------------
    # Read multiple frames
    # --------------------------------------------------------
    def read_frames(self, frame_indices, image_start_bytes):
        """
        Read multiple frames using mmap.

        Parameters
        ----------
        frame_indices : array-like
            0-based frame indices (used only for ordering)
        image_start_bytes : array-like
            Byte offsets corresponding to frame_indices

        Returns
        -------
        images : list of np.ndarray
        headers : list of dict
        """
        images = []
        headers = []

        for image_start in image_start_bytes:
            img, h = self.read_frame(image_start)
            images.append(img)
            headers.append(h)

        return images, headers
