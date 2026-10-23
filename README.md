# Cloud Detection over Desertic Surfaces using U-Net

> A deep learning project leveraging the U-Net architecture to perform semantic segmentation of clouds over arid and desert regions using EUMETSAT SEVIRI satellite data.

---

## Project Overview

The main objective of this project is to detect clouds over bright desert surfaces, a notoriously challenging task in satellite remote sensing, using a **U-Net** deep learning network. It is divided into two main parts:

1) **Data Gathering:** A pipeline that downloads EUMETSAT's data and automatically resamples and crops it over the desired ROIs. It then assembles the corresponding ROIs across different channels alongside the cloud mask, storing them into Zarr files for later use.
2) **U-Net Training:** Creation of the model architecture and training over the previously gathered data. The final goal of this part is to visualize the cloud segmentation over an RGB satellite image.

### Key Components
- **Data Source:** EUMETSAT Data Store (accessed via the `eumdac` API).
- **Input Data:** SEVIRI full-disk imagery across multiple spectral channels focused on various desert Regions of Interest (ROIs).
- **Target / Ground Truth:** MSG Cloud Mask (CLM) product used for training, evaluation, and validation of the segmentation models.

---

## Project Roadmap and Progress

### Completed
- [x] Project scope definition and architecture selection (U-Net)
- [x] Repository setup and file-system structure
- [x] EUMETSAT Data Store API connection (`eumdac`)
- [x] Desert Regions of Interest (ROIs) bounding boxes definition -> 3 ROIs over the Sahara desert
- [x] Data pipeline fully implemented (download, resample, crop, train/test split, and Zarr storage)

### In Progress (Current Focus)
- [ ] Running a scouting script to analyze average cloud coverage/days over the ROIs
- [ ] Running the data download and generation pipeline

### Upcoming Steps (Modeling & Evaluation)
- [ ] Building and training the U-Net model on the ready-to-use Zarr datasets
- [ ] Model evaluation and metrics computation
- [ ] Visualizing cloud segmentation results on desert test cases
- [ ] Writing final documentation and conclusions