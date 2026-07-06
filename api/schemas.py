from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_sw_loaded: bool = False
    model_sw_v5_loaded: bool = False
    model_sw_v6_loaded: bool = False
    model_sw_v7_loaded: bool = False
    model_sw_v8_loaded: bool = False
    model_sw_v9_loaded: bool = False
    model_sw_v10_loaded: bool = False
    model_sw_v11_loaded: bool = False


class SegmentationResponse(BaseModel):
    lesion_volume_ml: float = Field(
        description="Estimated lesion volume in millilitres, derived from voxel count × voxel size"
    )
    hemisphere: str = Field(
        description="Dominant hemisphere of the lesion: 'left', 'right', 'bilateral', or 'none'"
    )
    centroid_voxel: list[int] = Field(
        description="[D, H, W] index of the lesion centroid in the resized (64×128×128) volume"
    )
    lesion_voxel_count: int = Field(
        description="Number of lesion voxels in the resized prediction volume"
    )
    mask_shape: list[int] = Field(
        description="[D, H, W] shape of the returned prediction mask"
    )
    mask_base64: str = Field(
        description="Base64-encoded uint8 numpy array of the binary segmentation mask (row-major)"
    )
    model_version: str = "v1.0"


class CascadeSegmentationResponse(SegmentationResponse):
    """v15 detect-then-segment response: the standard segmentation fields plus
    the Stage-1 detector verdict. When hemorrhage_detected is False the mask is
    empty by design (the gate suppressed segmentation on a scan the detector
    judged lesion-free)."""

    hemorrhage_detected: bool = Field(
        description="Stage-1 verdict: did the detector flag hemorrhage in this scan?"
    )
    case_probability: float = Field(
        description="Case-level P(hemorrhage) = max detector probability over all slices"
    )
    case_threshold: float = Field(
        description="Gate threshold; case_probability >= this triggers segmentation"
    )
    slice_probabilities: list[float] = Field(
        description="Per-axial-slice detector probability (ensemble mean over folds), index = D"
    )
