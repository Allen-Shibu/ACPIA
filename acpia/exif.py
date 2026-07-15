"""Metadata-only extraction from image evidence.

IMPORTANT: this reads EXIF headers ONLY — the capture timestamp, GPS location and
camera info. It never decodes, displays, analyzes or interprets the image content
itself. That keeps image-evidence handling forensically and ethically safe: we get
the useful metadata without the tool ever looking at the picture.
"""

from datetime import datetime

from PIL import Image, ExifTags

_GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}


def _to_degrees(dms, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)
        val = d + m / 60 + s / 3600
        if ref in ("S", "W"):
            val = -val
        return val
    except Exception:
        return None


def extract_image_metadata(path) -> dict:
    """Return {timestamp, gps, camera} from an image's EXIF headers (no pixels read)."""
    result = {"timestamp": None, "timestamp_raw": None, "gps": None, "camera": None}
    with Image.open(path) as img:
        exif = img.getexif()
        if not exif:
            return result

        tag_by_name = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        # DateTimeOriginal lives in the Exif sub-IFD, not the top-level 0th IFD.
        try:
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
            for k, v in exif_ifd.items():
                tag_by_name.setdefault(ExifTags.TAGS.get(k, k), v)
        except Exception:
            pass

        raw_dt = tag_by_name.get("DateTimeOriginal") or tag_by_name.get("DateTime")
        if raw_dt:
            result["timestamp_raw"] = str(raw_dt)
            try:
                result["timestamp"] = datetime.strptime(
                    str(raw_dt), "%Y:%m:%d %H:%M:%S"
                ).isoformat()
            except ValueError:
                pass

        make = tag_by_name.get("Make")
        model = tag_by_name.get("Model")
        if make or model:
            result["camera"] = " ".join(str(x).strip() for x in (make, model) if x)

        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(ExifTags, "IFD") else {}
        if gps_ifd:
            lat = _to_degrees(
                gps_ifd.get(_GPS_TAGS.get("GPSLatitude")),
                gps_ifd.get(_GPS_TAGS.get("GPSLatitudeRef")),
            )
            lon = _to_degrees(
                gps_ifd.get(_GPS_TAGS.get("GPSLongitude")),
                gps_ifd.get(_GPS_TAGS.get("GPSLongitudeRef")),
            )
            if lat is not None and lon is not None:
                result["gps"] = {"lat": lat, "lon": lon}

    return result
