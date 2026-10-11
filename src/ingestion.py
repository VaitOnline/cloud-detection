import numpy as np

from typing import List, Tuple
from satpy import Scene
from pyresample.geometry import create_area_def




class SpatialResampler:
    """
    Deals with the reprojection and resampling of a Satpy Scene and store the result in a zarr file.
    """

    def __init__(self, bbox: List[float], shape: Tuple[int,int]=(512,512), crs: str="EPSG:4326", area_id: str="target_roi"):

        self.bbox = bbox
        self.shape = shape
        self.crs = crs
        self.area_def = create_area_def(    #Create a blank box of a given shape centered on a geographic area
            area_id = area_id,
            projection = self.crs,
            area_extent = self.bbox,
            shape = self.shape,
            units = "degrees"
        )

    def resample(self, scene: Scene, resampler: str="bilinear") -> Scene:
        """
        Project the satpy Scene on the geographical grid previously initialized with create area_dif.
        """
        return scene.resample(self.area_def, resampler=resampler)