import concurrent.futures
import os

import ee
import geopandas as gpd
from google.api_core import retry
import numpy as np
import pandas as pd
import timm
from torchgeo.models import ViTSmall16_Weights
import torch
import torchvision
import datetime

import utils


class S2_Data_Extractor:
    """
    Pull Sentinel-2 data for a set of tiles.
    Inputs:
        - tiles: a list of DLTile objects
        - start_date: the start date of the data
        - end_date: the end date of the data
        - clear_threshold: the threshold for cloud cover
        - batch_size: the number of tiles to process in each batch (default: 500)
    Methods: Functions are run in parallel.
        - get_data: pull the data for the tiles. Returns numpy arrays of chips
        - predict: predict on the data for the tiles. Returns a gdf of predictions and geoms
        - process_tile: Function h

    """

    def __init__(self,
                 tiles,
                 start_date,
                 end_date,
                 clear_threshold,
                 batch_size=500):
        self.tiles = tiles
        self.start_date = start_date
        self.end_date = end_date
        self.clear_threshold = clear_threshold
        self.batch_size = batch_size

        ee.Initialize(
            opt_url="https://earthengine-highvolume.googleapis.com",
            project="earth-engine-ck",
        )

        # Harmonized Sentinel-2 Level 2A collection.
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")

        # Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
        # Level 1C data and can be applied to either L1C or L2A collections.
        csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        QA_BAND = "cs_cdf"

        # Make a clear median composite.
        self.composite = (s2.filterDate(start_date, end_date).linkCollection(
            csPlus, [QA_BAND]).map(lambda img: img.updateMask(
                img.select(QA_BAND).gte(clear_threshold))).median())

        landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1").filterDate(
            start_date, end_date)

        self.composite_landsat = ee.Algorithms.Landsat.simpleComposite(
            **{
                'collection': landsat,
                'percentile': 30,
                'cloudScoreRange': 5,
                'asFloat': True,
            })

        # Prioritize GPU, then MPS, then CPU
        self.device = torch.device("cuda" if torch.cuda.is_available(
        ) else "mps" if torch.backends.mps.is_available() else "cpu")

        self.model = self.create_model()

    @retry.Retry(timeout=240)
    def get_tile_data(self, tile):
        """
        Download Sentinel-2 data for a tile.
        Inputs:
            - tile: a DLTile object
            - composite: a Sentinel-2 image collection
        Outputs:
            - pixels: a numpy array containing the Sentinel-2 data
        """
        tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
        composite_tile = self.composite.clipToBoundsAndScale(
            geometry=tile_geom, width=tile.tilesize, height=tile.tilesize)
        pixels = ee.data.computePixels({
            "bandIds": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8A",
                "B8",
                "B9",
                "B10",
                "B11",
                "B12",
            ],
            "expression":
            composite_tile,
            "fileFormat":
            "NUMPY_NDARRAY",
            #'grid': {'crsCode': tile.crs} this was causing weird issues that I believe caused problems.
        })

        # convert from a structured array to a numpy array
        pixels = np.array(pixels.tolist())

        return pixels, tile

    @retry.Retry(timeout=240)
    def get_chips(self, tile):
        """
        Takes in a tile of data and a model
        Outputs a gdf of predictions and geometries
        """
        pixels, tile_info = self.get_tile_data(tile)
        pixels = np.array(utils.pad_patch(pixels, tile_info.tilesize))

        chip_size = 32
        stride = chip_size // 2
        chips, chip_geoms = utils.chips_from_tile(pixels, tile_info, chip_size,
                                                  stride)
        chips = np.array(chips)
        chip_geoms.to_crs("EPSG:4326", inplace=True)

        return chips, chip_geoms, tile_info.key

    def create_model(self):
        weights = ViTSmall16_Weights.SENTINEL2_ALL_DINO

        in_chans = weights.meta["in_chans"]
        model = timm.create_model("vit_small_patch16_224", in_chans=in_chans)
        model.load_state_dict(weights.get_state_dict(progress=True),
                              strict=False)
        # remove the last layer of the model
        model.head = torch.nn.Identity()
        model = model.to(self.device)
        model.eval()
        return model

    def get_features(self, model, chips):
        chips_tensor = torch.from_numpy(np.array(chips) / 10_000).permute(
            0, 3, 1, 2).float()
        # resize chips to 224x224
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(224, antialias=True)])
        chips_tensor = transform(chips_tensor)
        chips_tensor = chips_tensor.to(self.device)
        with torch.no_grad():
            features = model(chips_tensor)
        return features.cpu().numpy()

    def get_patches(self):
        chips = []
        tile_data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, len(self.tiles), self.batch_size):
                batch_tiles = self.tiles[i:i + self.batch_size]

                # Process each tile in parallel
                futures = [
                    executor.submit(self.get_tile_data, tile)
                    for tile in batch_tiles
                ]

                # Collect the results as they become available
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    pixels, tile = result
                    chips.append(pixels)
                    tile_data.append(tile)
        return chips, tile_data

    def create_embeddings(self):
        """
        Predict on the data for the tiles.
        Inputs:
            - model: a keras model
            - batch_size: the number of tiles to process in each batch (default: 500)
        Outputs:
            - predictions: a gdf of predictions and geoms
        """

        completed_tasks = 0
        start_time = datetime.datetime.now()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, len(self.tiles), self.batch_size):
                batch_tiles = self.tiles[i:i + self.batch_size]
                # check if the embeddings have already been created
                batch_tiles = [
                    tile for tile in batch_tiles if
                    not os.path.exists(f"../outputs/embeddings/{tile.key}.npy")
                ]
                futures = [
                    executor.submit(self.get_chips, tile)
                    for tile in batch_tiles
                ]

                for future in concurrent.futures.as_completed(futures):
                    # make predictions on the chips
                    result = future.result()
                    chips, chip_geoms, tile_key = result
                    # project chip_geoms to a crs that is compatible with the centroids
                    chip_geoms = chip_geoms.to_crs("EPSG:3857")
                    centroids = chip_geoms.centroid
                    # convert the centroids to epsg:4326
                    centroids = centroids.to_crs("EPSG:4326")
                    # convert the centroids to a numpy array
                    centroids = np.array(
                        [(point.x, point.y) for point in centroids])
                    start = datetime.datetime.now()
                    features = self.get_features(self.model, chips)
                    end = datetime.datetime.now()
                    # write the geometries to a file
                    np.save(f"../outputs/centroids/{tile_key}.npy", centroids)
                    np.save(f"../outputs/embeddings/{tile_key}.npy", features)
                    completed_tasks += 1
                    # printing stats is a vanity metric. Maybe should delete, but I like to see the progress accumulate.
                    tiles_remaining = len(self.tiles) - completed_tasks
                    average_tile_speed = (
                        end - start_time).total_seconds() / completed_tasks
                    last_tile_speed = (end - start).total_seconds()
                    estimated_time_remaining = tiles_remaining * average_tile_speed / 60
                    print(
                        f"Completed {completed_tasks:,}/{len(self.tiles):,} tiles. {estimated_time_remaining:.1f} minutes remaining. Last tile took {last_tile_speed:.1f} seconds.",
                        end="\r",
                    )
