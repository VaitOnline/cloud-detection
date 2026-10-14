import numpy as np
import os
import zipfile
import glob
import xarray as xr

from typing import List, Tuple, Optional
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

def proces_clm(self, clm_zip_path: str) -> Tuple[bool, float, Optional[np.array]]:
    """
    Decompress the GRIB format Cloud Mask, resample it and evaluate the cloud coverage.
    It makes use of the Satpy 'eum_l2_grib'.
    """

    work_dir = self.extract_zip(clm_zip_path, "clm_work")
    grb_files = glob.glob(os.path.join(work_dir, "*.grb"))

    if not grb_files:
        self.cleanup_dir(work_dir)
        raise FileNotFoundError(f"No GRIB (.grb) files found in: {clm_zip_path}")

    scene_clm = Scene(reader="eum_l2_grib", filenames=[grb_files[0]])
    var_names = scene_clm.available_dataset_names()[0]
    scene_clm.load([var_names])

    # Resample to the nearest neighbour to preserve the classes
    # If it were to be done by interpolation, we would lose the boolean property of the mask

    aligned_clm = self.resampler.resample(scene_clm, resampler="nearest")
    clm_array = aligned_clm[var_names].values
    self.cleanup_dir(work_dir)

    # 0, 1 are clear surface over land and see || 2, 3 are cloudy or un processed
    # Binarisation
    binary_mask = (clm_array >= 2).astype(np.utin8)
    cloud_coverage = float(np.nanmean(binary_mask))

    if cloud_coverage >= self.min_cloud_coverage:
        return True, cloud_coverage, binary_mask

    return False, cloud_coverage, None


def process_seviri(self, seviri_path: str) -> np.ndarray:
    """
    Decompress, load and resample the spectral channels of SEVIRI.
    It makes use of the Satpy 'seviri_l1b_native'.
    """

    channels = ["VIS006", "VIS008", "IR_016"]
    scene_sev = Scene(reader="seviri_l1b_native", filenames=[seviri_path])
    scene_sev.load(channels)

    # Bilinear resampling for continuous data
    aligned_sev = self.resampler.resample(scene_sev, resampler="bilinear")

    vis006 = aligned_sev["VIS006"].values.astype(np.float32)
    vis008 = aligned_sev["VIS008"].values.astype(np.float32)
    ir016 = aligned_sev["IR_016"].values.astype(np.float32)

    # Final shape: (3, 15, 15)
    return np.stack([vis006, vis008, ir016])


def append_sample(self, timestamp: str, image_data: np.ndarray, mask_data: np.ndarray):
    """
    Add the resampled and aligned data in the Zarr file.
    """

    ds = xr.Dataset(
        data_vars={
            "image": (["time", "channel", "y", "x"], image_data[None, ...]),    # Add 1D in front of the data to fit the dictionnary format (time)
            "mask": (["time", "y", "x"], mask_data[None, ...])
        },
        coords={
            "time":[np.datetime64(timestamp)],
            "channel": ["VIS006", "VIS008", "IR_016"],
            "y": np.arrange(self.shape[0]),
            "x": np.arrange(self.shape[1])
        }
    )

    mode = "a" if os.path.exists(self.zarr_path) else "w"   # Append if file already exists and write otherwise (create)
    kwargs = {"append_dim": "time"} if mode == "a" else {}  # Add data by enlarging the time dimension

    ds.to_zarr(self.zarr_path, mode=mode, **kwargs)
















