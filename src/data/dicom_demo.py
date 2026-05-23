"""
DICOM loading demonstration using pydicom.

This module shows how to:
  1. Load a folder of DICOM slices (.dcm) as a 3D volume
  2. Convert raw pixel values to Hounsfield Units (HU)
  3. Extract clinically relevant metadata

The main pipeline in this project uses NIfTI files (lighter format for
training), but DICOM is the standard format used by CT scanners and PACS
systems — this module demonstrates that pydicom workflow.

Sample DICOM files for testing:
  - pydicom ships test files: import pydicom; pydicom.data.get_testfiles_name('CT_small.dcm')
  - TCIA (The Cancer Imaging Archive): https://www.cancerimagingarchive.net
"""

import numpy as np
import pydicom
from pathlib import Path


def load_dicom_slice(dcm_path: str) -> tuple[np.ndarray, dict]:
    """
    Load a single DICOM slice and convert to Hounsfield Units.

    Returns:
        hu_slice: float32 2-D array (H, W) in HU
        metadata: dict with key DICOM tags
    """
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Rescale to HU: HU = pixel * slope + intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu_slice = pixel_array * slope + intercept

    metadata = {
        "patient_id": str(getattr(ds, "PatientID", "UNKNOWN")),
        "study_date": str(getattr(ds, "StudyDate", "")),
        "modality": str(getattr(ds, "Modality", "CT")),
        "rows": int(ds.Rows),
        "cols": int(ds.Columns),
        "pixel_spacing_mm": [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])],
        "slice_thickness_mm": float(getattr(ds, "SliceThickness", 0.0)),
        "instance_number": int(getattr(ds, "InstanceNumber", 0)),
        "hu_min": float(hu_slice.min()),
        "hu_max": float(hu_slice.max()),
    }
    return hu_slice, metadata


def load_dicom_series(series_dir: str) -> tuple[np.ndarray, dict]:
    """
    Load all DICOM slices in a directory and stack into a 3-D volume.

    Slices are ordered by InstanceNumber (acquisition order).

    Returns:
        volume:   float32 array (D, H, W) in HU
        metadata: dict with series-level metadata + per-slice count
    """
    dcm_files = sorted(
        Path(series_dir).glob("*.dcm"),
        key=lambda f: float(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber),
    )

    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {series_dir}")

    slices = []
    for f in dcm_files:
        hu_slice, _ = load_dicom_slice(str(f))
        slices.append(hu_slice)

    volume = np.stack(slices, axis=0)  # (D, H, W)

    # Read metadata from the last slice (headers are consistent within a series)
    _, slice_meta = load_dicom_slice(str(dcm_files[-1]))
    series_meta = {
        **slice_meta,
        "n_slices": len(slices),
        "volume_shape": list(volume.shape),
        "spacing_mm": [
            slice_meta["slice_thickness_mm"],
            *slice_meta["pixel_spacing_mm"],
        ],
    }
    return volume, series_meta


def demo_with_pydicom_testfile() -> tuple[np.ndarray, dict]:
    """
    Quick demo using pydicom's built-in test CT file.
    No external data required — useful for CI and notebook demos.
    """
    test_file = pydicom.data.get_testdata_file("CT_small.dcm")
    return load_dicom_slice(str(test_file))
