import glob
import os
import shutil
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
 
import numpy as np
import xarray as xr
from pyresample.geometry import create_area_def
from satpy import Scene


SEVIRI_CHANNELS = ["VIS006", "VIS008", "IR_016"]

ClmResult = Tuple[bool, float, Optional[np.ndarray]]


class SpatialResampler:
    """
    Reprojects a Satpy Scene onto a regular lon/lat grid covering one ROI.
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
    Extracts, resample and archives ROIs from EUMETSAT products into Zarr stores.

    One spatial resampler is build for each ROI. Each product (CLM or SEVIRI) is extracted and read once, then resampled for every requested ROI.
    """

    def __init__(self, rois:List[dict], shape: Tuple[int,int]=(512,512), min_cloud_cover: float=0.20, temp_dir: str="temp"):
        self.min_cloud_cover = min_cloud_cover
        self.shape = tuple(shape)
        self.temp_path = temp_dir
        os.makedirs(self.temp_path, exist_ok=True)

        # We initialize one Resampler per ROI keyed by the roi name
        self.resamplers: Dict[str, SpatialResampler] = {
            roi["name"]: SpatialResampler(
                bbox=roi["bbox"], shape=self.shape, area_id=roi["name"]
            )
            for roi in rois
        }

    #============================
    #          FILE            ||
    #============================

    def extract_zip(self, zip_path: str, subfolder_name: str) -> str:
        """
        Extract a zip into a clean subfolder of the temporary directory.
        """
        target_path = os.path.join(self.temp_path, subfolder_name)
        self.cleanup_dir(target_path)
        
        os.makedirs(target_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        return target_path

    @staticmethod
    def cleanup_dir(dir_path: str):
        """
        Remove a folder and everything inside it (files and subfolders).
        """
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)

    @staticmethod
    def find_files(root: str, patterns: Tuple[str, ...]) -> List[str]:
        found = []
        for pattern in patterns:
            found += glob.glob(os.path.join(root, "**", pattern), recursive=True)
        return sorted(found)

    def process_clm(self, clm_zip_path: str, roi_names: List[str]) -> Dict[str, ClmResult]:
        """
        Extract the GRIB cloud mask once, then for each ROI resample (nearest neighbour to keep the classes), binarise and compute the cloud cover.

        Binary mask: 1 = cloudy / unprocessed / no data, 0 = clear.
        ROI is valide when its cloud cover is >= min_cloud_cover.
        """

        work_dir = self.extract_zip(clm_zip_path, "clm_work")

        try:
            grb_files = self.find_files(work_dir, ("*.grb", "*.grib", "*.grb2"))
            if not grb_files:
                raise FileNotFoundError(f"No GRIB file found in: {clm_zip_path}")

            scene_clm = Scene(reader="eum_l2_grib", filenames=[grb_files[0]])
            var_names = scene_clm.available_dataset_names()[0]
            scene_clm.load([var_names])

            results = {}
            for roi_name in roi_names:
                try:
                    aligned = self.resamplers[roi_name].resample(scene_clm, resampler="nearest")
                    clm_array = np.asarray(aligned[var_names].values, dtype=np.float64)

                    binary_mask = (np.isnan(clm_array) | (clm_array >= 2)).astype(np.uint8)
                    cloud_cover = float(binary_mask.mean())

                    is_valid = cloud_cover>=self.min_cloud_cover
                    results[roi_name] = (is_valid, cloud_cover, binary_mask if is_valid else None)

                except Exception as e:
                    print(f"       [ERROR CLM] ROI {roi_name}: {e}")
                    results[roi_name] = (False, float("nan"), None)

            return results
    
        finally:
            self.cleanup_dir(work_dir)



    def process_seviri(self, seviri_zip_path: str, roi_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract the SEVIRI native data once, load the channels once, then resample (bilinear) for each ROI. 
        Returns {roi_name: array (3, H, W)}.
        """
        work_dir = self.extract_zip(seviri_zip_path, "seviri_work")
        try:
            nat_files = self.find_files(work_dir, ("*.nat",))       # The additional , is to be sure that python takes it as a singleton
            if not nat_files:
                raise FileNotFoundError(f"No .nat files found in: {seviri_zip_path}")

            scene = Scene(reader="seviri_l1b_native", filenames=[nat_files[0]])
            scene.load(SEVIRI_CHANNELS)

            results = {}
            for roi_name in roi_names:
                try:
                    aligned = self.resamplers[roi_name].resample(scene, resampler="bilinear")
                    bands = [aligned[ch].values.astype(np.float32) for ch in SEVIRI_CHANNELS]
                    results[roi_name] = np.stack(bands)     # (3, H, W)
                except Exception as e:
                    print(f"    [ERROR SEVIRI] ROI {roi_name}: {e}")

            return results
        finally:
            self.cleanup_dir(work_dir)


    #======================================
    #               ZARR                 ||   
    #======================================
    @staticmethod
    def make_sample_id(timestamp:datetime, roi_name:str) -> str:
        return f"{timestamp.strftime('%Y%m%dT%H%M%S')}_{roi_name}"

    @staticmethod
    def zarr_exists(zarr_path:str) -> bool:
        return os.path.isdir(zarr_path) and len(os.listdir(zarr_path)) > 0

    def sample_exists(self, zarr_path: str, sample_id: str) -> bool:
        """
        True if the sample is already stored.
        """
        if not self.zarr_exists(zarr_path):
            return False
        with xr.open_zarr(zarr_path, consolidated=False) as ds:
            return sample_id in ds["sample_id"].values

    def append_sample(self, zarr_path: str, timestamp: datetime, roi_name: str, image_data: np.ndarray, mask_data: np.ndarray):
        """
        Append one (image, mask) sample to the Zarr file, creating it if necessary.
        """
        sample_id = self.make_sample_id(timestamp, roi_name)
        ts = np.datetime64(timestamp.replace(tzinfo=None), "ns")        # Convert datetime into numpy without timezone
        height, width = mask_data.shape

        ds = xr.Dataset(
            data_vars={
                "image": (
                    ("sample", "channel", "y", "x"),
                    image_data[None, ...].astype(np.float32)
                ),
                "mask": (
                    ("sample", "y", "x"),
                         mask_data[None, ...].astype(np.uint8))
            },
            coords={
                "sample_id": ("sample", np.array([sample_id], dtype=object)),
                "time": ("sample", np.array([ts], dtype="dataetime64[ns]")),
                "roi_name": ("sample", np.array([roi_name], dtype=object)),
                "channel": SEVIRI_CHANNELS,
                "y": np.arange(height),
                "x": np.arange(width)
            }
        )

        if self.zarr_exists(zarr_path):
            # No encoding when appending 
            ds.to_zarr(zarr_path, mode="a", append_dim="sample", consolidated=False)
        else:
            os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
            encoding = {
                "image": {"chunks": (1, len(SEVIRI_CHANNELS), height, width)},
                "mask": {"chunks": (1, height, width)}
            }
            ds.to_zarr(zarr_path, mode="w-", encoding=encoding, consolidated=False)