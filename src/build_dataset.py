import os
import yaml
import eumdac
import shutil
import argparse

from datetime import datetime, timedelta
from typing import Dict, List, Any
from dotenv import load_dotenv

from src.ingestion import EumetsatZarrPipeline


# Load values in the .env
load_dotenv()

BASE_DATA_DIR = os.getenv("DATA_DIR", "data")

def get_datastore() -> eumdac.DataStore:
    """
    Retrieve the key and secret from the .env file and set up the eumdac DataStore.
    """
    key = os.getenv("EUMETSAT_CONSUMER_KEY")
    secret = os.getenv("EUMETSAT_CONSUMER_SECRET")

    if not key or not secret:
        raise ValueError("EUMETSAT_CONSUMER_KEY or EUMETSAT_CONSUMER_SECRET missing from the .env file.")
    token = eumdac.AccessToken((key, secret))
    return eumdac.DataStore(token)


def load_config(config_path:str) -> Dict[str, Any]:
    """
    Load the configuration YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_within_target_hours(dt: datetime, start_hour:int, end_hour: int) -> bool:
    """
    Verify the validity of the hours setting in the config file.
    """
    return start_hour <= dt.hour <= end_hour


def download_product(product, dest_path: str) -> None:
    with product.open() as frsc, open(dest_path, "wb") as fdst:
        shutil.copyfileobj(frsc, fdst)


def process_timestamp(dt: datetime, clm_prod, seviri_collection, pipeline: EumetsatZarrPipeline, rois: List[Dict], zarr_paths: Dict[str, str], temp_dir: str):
    """
    Handle one single timestamp:
        - Download the CLM once
        - Keep the rois that are valid (clouded enough)
        - Download SEVIRI once
        - Store one sample per valid roi
    The temporary zip files are always removed, even on error.
    """
    stamp = dt.strftime("%Y%m%d_%H%M%S")
    clm_zip = os.path.join(temp_dir, f"clm_{stamp}.zip")
    sev_zip = os.path.join(temp_dir, f"sev_{stamp}.zip")

    try:
        # Download 1 CLM -> 1 read and N ROIs
        download_product(clm_prod, clm_zip)
        clm_results = pipeline.process_clm(clm_zip, [r["name"] for r  in rois]) # Comprehension create a List containing the keys due to the []

        valid_rois = []
        for roi in rois:
            is_valid, cloud_cover, _ = clm_results[roi["name"]]
            if is_valid:
                valid_rois.append(roi)
                print(f"    [ROI {roi['name']} ({roi['group'].upper()})] valid ({cloud_cover *100:.1f}% clouds)")

        if not valid_rois:
            return

        # Seviri data downloaded only if at least one roi is valid
        results = seviri_collection.search(dtstart=dt-timedelta(minutes=1), dtend=dt+timedelta(minutes=1))

        sev_prod = next(iter(results), None)    # The search outputs a itterator whose element has to be accessed via itter and next (None is used to avoid a crash when itter faces nothing)\
        if sev_prod is None:
            print(f"    [NO SEVIRI] nothing found for {dt.isoformat()}")
            return

        download_product(sev_prod, sev_zip)
        images = pipeline.process_seviri(sev_zip, [r["name"] for r in valid_rois])

        # Storage of the timestamp
        for roi in valid_rois:
            name, group = roi["name"], roi["group"]
            if name not in images:
                continue
            _, _, mask = clm_results[name]
            try:
                pipeline.append_sample(
                    zarr_path=zarr_paths[group],
                    timestamp=dt,
                    roi_name=name,
                    image_data=images[name],
                    mask_data=mask                    
                )
                print(f"    [SAMPLE ADDED TO {group.upper()}] {dt.isoformat()} ({name})")

            except Exception as e:
                print(f"    [ERROR ZARR] ROI {name} ({group}) at {dt.isoformat()}: {e}")

    finally:
        for path in (clm_zip, sev_zip):
            if os.path.exists(path):
                os.remove(path)


def run_pipeline(datastore: eumdac.DataStore, config_path: str = "config/config.yaml") -> None:
    """
    - List the over each period and keep the targeted hours.
    - Download each CLM once and check the cloud cover on every ROIs.
    - Download SEVIRI only if at least one ROI is clear enough. 
    - Crop/resample on each valid ROI and append to the train or test zarr. 
    """

    config = load_config(config_path)
    pipe_config = config["pipeline"]
    outputs = config["outputs"]
    target_hours = config["target_hours"]
    collections = config["collections"]
    years = pipe_config["years"]
    seasons = config["season_ranges"]
    shape = tuple(pipe_config["shape"])
    min_cloud_cover = pipe_config["min_cloud_cover"]
    all_rois = config["rois"]

    temp_dir = os.path.join(BASE_DATA_DIR, pipe_config.get("temp_dir", "temp"))

    zarr_paths = {
        "train": os.path.join(BASE_DATA_DIR, outputs["train_zarr"]),
        "test": os.path.join(BASE_DATA_DIR, outputs["test_zarr"])
    }
    for roi in all_rois:
        if roi["group"] not in zarr_paths:
            raise ValueError(f"ROI {roi['name']}: unknown group '{roi['group']}' (expected train/test)")

    print("\n" + "="*50)
    print(f"Zarr train path: {zarr_paths['train']}")
    print(f"Zarr test path: {zarr_paths['test']}")
    print("\n" + "="*50)

    clm_collection = datastore.get_collection(collections["cloud"])
    seviri_collection = datastore.get_collection(collections["data"])

    pipeline = EumetsatZarrPipeline(
        rois=all_rois, shape=shape, min_cloud_cover=min_cloud_cover, temp_dir=temp_dir
    )

    for year in years:
        for season in seasons:
            start_s = f"{year}-{season['start_md']}"
            end_s = f"{year}-{season['end_md']}"
            print(f"\n Period: {season['name']} ({start_s} to {end_s})")

            dt_start = datetime.fromisoformat(f"{start_s}T00:00:00")
            dt_end = datetime.fromisoformat(f"{end_s}T23:59:59")

            # Full-disk product so all rois are contained in one product
            clm_products = {}
            try:
                for prod in clm_collection.search(dtstart=dt_start, dtend=dt_end):
                    dt = prod.sensing_start
                    if is_within_target_hours(dt, target_hours["start_hour"], target_hours["end_hour"]):
                        clm_products[dt] = prod
            except Exception as e:
                print(f"    [ERROR API CLM] search failed: {e}")

            print(f"    -> {len(clm_products)} timestamps to analyze.")

            for dt, clm_prod in sorted(clm_products.items()):
                # Skip the ROIs which are already stored so it is safe to re-run the script
                pending = [
                    roi for roi in all_rois
                    if not pipeline.sample_exists(zarr_paths[roi["group"]], pipeline.make_sample_id(dt, roi["name"]))
                ]
                if not pending:
                    continue

                try:
                    process_timestamp(
                        dt, clm_prod, seviri_collection, pipeline,
                        pending, zarr_paths, temp_dir
                    )
                except Exception as e:
                    print(f"    [ERROR] {dt.isoformat()}: {e}")

    print("\n Pipeline finished. Train and test Zarr are ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline EUMETSAT to Zarr (train/test)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the YAML config")
    args = parser.parse_args()

    run_pipeline(get_datastore(), config_path=args.config)
