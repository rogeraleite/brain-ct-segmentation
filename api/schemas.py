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
