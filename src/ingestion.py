import numpy as np
import os
import zipfile
import glob

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


class EumetsatZarrPipeline:
    """
    Pipeline to extract, filter, resample and archive a given ROI from and image into local Zarr files.
    """

    def __init__(self, zarr_path: str, bbox: List[float], shape: Tuple[int,int]=(512,512), min_cloud_cover: float=0.20, temp_dir: str="temp", area_id: str="roi"):
        self.min_cloud_cover = min_cloud_cover
        self.shape = shape
        self.resampler = SpatialResampler(bbox=bbox, shape=shape, area_id=area_id)

        # Creation of files on local storage
        base_path = os.getenv("DATA_DIR", "./default_data_temp")

        self.temp_path = os.path.join(base_path,temp_dir)
        self.zarr_path = os.path.join(base_path,zarr_path)

        os.makedirs(self.temp_path, exist_ok=True)
        os.makedirs(self.zarr_path, exist_ok=True)

def extract_zip(self, zip_path: str, subfolder_name: str) -> str:
    """
    Decompress a EUMETSAT archive zip file in the local temporary folder.
    """
    target_path = os.path.join(self.temp_path, subfolder_name)
    os.makedirs(target_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    return target_path

def cleanup_dir(self, dir_path: str):
    """
    Clean the target folder to preserve disk storage.
    """
    if os.path.exists(dir_path):
        for file_path in glob.glob(os.path.join(dir_path, "*")):
            if os.path.isfile(file_path):   #Verify that the target is a file
                os.remove(file_path)








