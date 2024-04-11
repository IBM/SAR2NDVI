import datetime
import glob
import json
import os
import random
import shutil
import time
import zipfile
from decimal import Decimal

import cv2
import ee
import matplotlib.pyplot as plt
import numpy as np
import PIL
import rasterio
import rasterio.mask
import requests
import richdem as rd
import torch
from pyproj import Transformer
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset
from tqdm import tqdm


def do_nothing(x, dems=None):
    return x


def do_nothing_xy(x, y):
    return x, y


def img_reshape(_img, axis=0):
    if axis == 0:
        return _img.reshape((1, _img.shape[0], _img.shape[1]))
    elif axis == 2:
        return _img.reshape((_img.shape[0], _img.shape[1], 1))
    else:
        raise NotImplementedError


def transform_vvvh(vvvh, dems=None,
                   img_size=0, input_types="vvmax_vhmax_dvvvhmax"):
    vv = cv2.resize(vvvh[0], dsize=(img_size, img_size)) \
        if img_size > 0 else vvvh[0]
    vh = cv2.resize(vvvh[1], dsize=(img_size, img_size)) \
        if img_size > 0 else vvvh[1]
    inputs = list()
    for input_type in input_types.split("_"):
        if input_type.startswith("vv"):
            d = vv
        elif input_type.startswith("vh"):
            d = vh
        elif input_type.startswith("dvvvh"):
            d = np.divide(vv, vh,
                          out=np.zeros_like(vv, dtype=np.float64),
                          where=vh != 0)
        elif input_type.startswith("dvhvv"):
            d = np.divide(vh, vv,
                          out=np.zeros_like(vh, dtype=np.float64),
                          where=vv != 0)
        elif input_type.startswith("prvi"):
            d = (1 - vv / (vh + vv)) * vh
        elif input_type.startswith("rfdi"):
            d = (vv - vh) / (vh + vv)
        elif input_type.startswith("rvi4s1"):
            d = np.sqrt(vv / (vh + vv)) * ((4 * vh) / (vh + vv))
        elif input_type.startswith("rvi"):
            d = (4 * vh) / (vh + vv)
        elif input_type.startswith("sni"):
            d = (vh - vv) / (vh + vv)
        elif input_type.startswith("pvhvv"):
            d = vh + vv
        elif input_type.startswith("mvhvv"):
            d = vh - vv
        elif input_type.startswith("dem"):
            d = cv2.resize(dems["dem"], dsize=(img_size, img_size)) \
                if img_size > 0 else dems["dem"]
        elif input_type.startswith("slope"):
            d = cv2.resize(dems["slope"], dsize=(img_size, img_size)) \
                if img_size > 0 else dems["slope"]
        elif input_type.startswith("aspect"):
            d = cv2.resize(dems["aspect"], dsize=(img_size, img_size)) \
                if img_size > 0 else dems["aspect"]
        elif input_type.startswith("gradx") or input_type.startswith("grady"):
            slope = cv2.resize(dems["slope"], dsize=(img_size, img_size)) \
                if img_size > 0 else dems["slope"]
            xylen = np.sin(np.deg2rad(slope))
            aspect = cv2.resize(dems["aspect"], dsize=(img_size, img_size)) \
                if img_size > 0 else dems["aspect"]
            aspect_rad = np.deg2rad(aspect)
            if input_type.startswith("gradx"):
                d = np.multiply(xylen, np.cos(aspect_rad))
            else:
                d = np.multiply(xylen, np.sin(aspect_rad))
        else:
            raise NotImplementedError("Not Found input_type: " + input_type)

        if input_type.endswith("max"):
            d /= d.max()
        elif input_type.endswith("01"):
            d = np.clip(d, 0.0, 1.0)
        elif input_type.endswith("995"):
            d = np.clip(d, 0, np.percentile(d, 99.5)) / np.percentile(d, 99.5)
        inputs.append(img_reshape(d))
    return np.concatenate(inputs, axis=0, dtype=np.float32)


def transform_ndvi(ndvi, dems=None,
                   img_size=0, n_channel=3, convert=True):
    ndvi = cv2.resize(ndvi[0], dsize=(img_size, img_size)) \
        if img_size > 0 else ndvi[0]
    if convert:
        ndvi = (ndvi + 1) / 2
    if n_channel == 1:
        return img_reshape(ndvi)
    elif n_channel == 3:
        return np.concatenate(
            (img_reshape(ndvi), img_reshape(ndvi), img_reshape(ndvi)),
            axis=0, dtype=np.float32)
    else:
        raise NotImplementedError()


def rotate_img(img, degree):
    height, width, _ = img.shape
    center = (width / 2, height / 2)
    rotate_mat = \
        cv2.getRotationMatrix2D(center=center, angle=degree, scale=1)
    rotated_img = cv2.warpAffine(src=img, M=rotate_mat, dsize=(width, height))
    return rotated_img


def augment_vvvh_ndvi(x, y, img_size, aug_types):
    x = np.moveaxis(x, 0, 2)
    y = np.moveaxis(y, 0, 2)

    _, _, x_channel = x.shape
    _, _, y_channel = y.shape

    for aug_type in aug_types.split("_"):
        if aug_type.startswith("aspect"):
            if x.shape[0] < x.shape[1]:  # 003
                w = random.randint(x.shape[0], x.shape[1])
                x0 = random.randint(0, x.shape[1] - w)
                x = x[:, x0:x0 + w, :]
                y = y[:, x0:x0 + w, :]

            elif x.shape[0] == x.shape[1]:  # 002
                h = random.randint(int(x.shape[0] * (76 / 102)), x.shape[0])
                y0 = random.randint(0, x.shape[0] - h)
                x = x[y0:y0 + h, :, :]
                y = y[y0:y0 + h, :, :]

        elif aug_type.startswith("rot"):
            rotation_max_degree = int(aug_type[3:])
            rotation_degree = \
                random.randint(-1 * rotation_max_degree, rotation_max_degree)
            x = rotate_img(x, rotation_degree)
            y = rotate_img(y, rotation_degree)

        elif aug_type.startswith("90rot"):
            rotation_value = random.randint(0, 3)
            if rotation_value > 0:
                x = np.rot90(x, k=rotation_value)
                y = np.rot90(y, k=rotation_value)

        elif aug_type.startswith("flip"):
            flip_value = random.randint(-1, 2)
            if flip_value < 2:
                x = cv2.flip(x, flip_value)
                y = cv2.flip(y, flip_value)

    x = cv2.resize(x, dsize=(img_size, img_size))
    y = cv2.resize(y, dsize=(img_size, img_size))

    if x_channel == 1:
        x = x.reshape((1, img_size, img_size))
    else:
        x = np.moveaxis(x, 2, 0)

    if y_channel == 1:
        y = y.reshape((1, img_size, img_size))
    else:
        y = np.moveaxis(x, 2, 0)

    return x, y


class SpaceShiftNdviDataset(Dataset):
    """
    SpaceShift NDVI inference dataset.
    """

    metadata = {
        "train": {
            "directory": "train",
            "start_ratio": 0,
            "end_ratio": 1,
            "shuffle": False,
            "y_data": True,
        },
        "train_train": {
            "directory": "train",
            "start_ratio": 0,
            "end_ratio": 0.6,
            "shuffle": True,
            "y_data": True,
        },
        "train_test": {
            "directory": "train",
            "start_ratio": 0.6,
            "end_ratio": 1,
            "shuffle": True,
            "y_data": True,
        },
        "test": {
            "directory": "test",
            "start_ratio": 0,
            "end_ratio": 1,
            "shuffle": False,
            "y_data": False,
        },
        "002": {
            "directory": "train",
            "start_ratio": 0,
            "end_ratio": 1,
            "shuffle": True,
            "y_data": True,
        },
        "003": {
            "directory": "train",
            "start_ratio": 0,
            "end_ratio": 1,
            "shuffle": True,
            "y_data": True,
        },
        "sentinel2_adjusted": {
            "directory": "",
            "start_ratio": 0,
            "end_ratio": 1,
            "shuffle": True,
            "y_data": True,
        },
        "sentinel2_adjusted_001": {
            "directory": "test",
            "start_ratio": 0,
            "end_ratio": 1,
            "shuffle": False,
            "y_data": True,
        },
        "sentinel2": {
            "directory": "",
            "start_ratio": 0,
            "end_ratio": 1,
            "shuffle": True,
            "y_data": True,
        },
    }

    def __init__(
        self,
        root: str = "dataset",
        split: str = "train",
        vvvh_original: bool = True,
        vvvh_crop: bool = False,
        ndvi_original: bool = True,
        ndvi_crop: bool = False,
        vv: bool = True,
        vh: bool = True,
        non_target: str = "skip",
        seed: int = 0,
        transform_x=do_nothing,
        transform_y=do_nothing,
        augmentation=do_nothing_xy,
    ) -> None:

        assert split in self.metadata.keys()
        assert vvvh_original or vvvh_crop

        self.root = root
        self.split = split
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.vvvh_original = vvvh_original
        self.vvvh_crop = vvvh_crop
        self.ndvi_original = ndvi_original
        self.ndvi_crop = ndvi_crop
        self.non_target = non_target
        self.vv = vv
        self.vh = vh
        self.augmentation = augmentation

        self.set_seed(seed)

        self.dems = self.load_dems()

        vv_filepaths = list()
        directory = os.path.join(self.root, self.metadata[split]["directory"])
        if split == "002" or split == "003":
            directory = os.path.join(directory, split)
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath.endswith("_VV.tif"):
                    vv_filepaths.append(filepath)

        vv_filepaths = sorted(vv_filepaths)
        self.paths_dict = self.create_path_dict(vv_filepaths)
        self.selected_paths_list_dict = dict()

        self.xs_list = self.load_images(self.paths_dict, "x", self.transform_x)
        if self.metadata[self.split]["y_data"]:
            self.ys_list = self.load_images(self.paths_dict, "y",
                                            self.transform_y)

        self.all_indexes = self.get_indexes(self.xs_list)

        ids = random.sample(self.all_indexes, len(self.all_indexes)) \
            if self.metadata[split]["shuffle"] else self.all_indexes
        self.indexes = \
            ids[int(self.metadata[split]["start_ratio"] * len(ids)):
                int(self.metadata[split]["end_ratio"] * len(ids))]

    def __getitem__(self, index: int):
        x, y = self.augmentation(
            self.xs_list[self.indexes[index][0]][self.indexes[index][1]],
            self.ys_list[self.indexes[index][0]][self.indexes[index][1]]
            if self.metadata[self.split]["y_data"] else 0)
        return x, y

    def __len__(self) -> int:
        return len(self.indexes)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    def get_indexes(self, xs_list: list):
        idx = list()
        dataset_idx = list()

        for i, xs in enumerate(xs_list):
            len_xs = len(xs)
            idx.extend(range(len_xs))
            dataset_idx.extend([i for _ in range(len_xs)])

        return [[di, i] for i, di in zip(idx, dataset_idx)]

    def fill_ndvi_path(self):
        def get_all_ndvi_tifs(folder):
            ndvi_tifs = list()
            for root, _, files in os.walk(folder):
                for file in files:
                    filepath = os.path.join(root, file)
                    if filepath.endswith("_ndvi.tif"):
                        ndvi_tifs.append(filepath)
            return ndvi_tifs
        if self.split == "sentinel2_adjusted_001":
            self.sentinel2_ndvi_paths = {
                "001": get_all_ndvi_tifs(self.root +
                                         "/../sentinel2/adjusted_001"),
            }
        else:
            self.sentinel2_ndvi_paths = {
                d: get_all_ndvi_tifs(self.root +
                                     (("/../sentinel2/adjusted_%s" % d)
                                      if self.split == "sentinel2_adjusted"
                                      else ("/../sentinel2/raw_%s" % d)))
                for d in ["001", "002", "003"]
            }

    def find_nearest_ndvi_path(self, vv_path, data_name):
        planet_date_str = \
            vv_path.replace(self.root, "").split("/")[3].split("_")[0]
        planet_datetime = datetime.datetime.strptime(planet_date_str, "%Y%m%d")
        closest_date_index = -1
        closest_date_dif_day = 60
        for i in range(len(self.sentinel2_ndvi_paths[data_name])):
            s2_date_str = self.sentinel2_ndvi_paths[data_name][i] \
                .split("/")[-1].replace("_ndvi.tif", "")
            s2_datetime = datetime.datetime.strptime(s2_date_str, "%Y%m%d")
            if abs(planet_datetime - s2_datetime).days < closest_date_dif_day:
                closest_date_dif_day = abs(planet_datetime - s2_datetime).days
                closest_date_index = i
        closest_ndvi_path = \
            self.sentinel2_ndvi_paths[data_name][closest_date_index]
        if closest_date_dif_day < 7:
            ndvi_img = self.load_tif_image(closest_ndvi_path)
            if not np.all(ndvi_img == -100):
                return closest_ndvi_path, \
                    closest_ndvi_path.replace("_ndvi.tif", "_ndvi_crop.tif")
            else:
                print("cloud detected: %s, %s" % (vv_path, closest_ndvi_path))
        else:
            print("cannot find near date: %s" % vv_path)
        return None, None

    def create_path_dict(self, paths: list):
        if self.split.startswith("sentinel2"):
            self.fill_ndvi_path()
        paths_dict = dict()
        for vv_path in paths:
            data_name = vv_path.replace(self.root, "").split("/")[2]
            if data_name not in paths_dict:
                paths_dict[data_name] = list()

            if self.split.startswith("sentinel2"):
                ndvi_path, ndvi_crop_path = \
                    self.find_nearest_ndvi_path(vv_path, data_name)
                if ndvi_path is None:
                    continue
            else:
                ndvi_folder = "/".join(vv_path.split("/")[:-2]) + "/PLANET/"
                ndvi_path, ndvi_crop_path = None, None
                for root, _, files in os.walk(ndvi_folder):
                    for file in files:
                        filepath = os.path.join(root, file)
                        if filepath.endswith("_ndvi.tif"):
                            ndvi_path = filepath
                        elif filepath.endswith("_ndvi_crop.tif"):
                            ndvi_crop_path = filepath

            x_paths, y_paths = list(), list()
            if self.vvvh_original:
                if self.vv:
                    x_paths += [vv_path]
                if self.vh:
                    x_paths += [vv_path.replace("VV.tif", "VH.tif")]
            if self.vvvh_crop:
                if self.vv:
                    x_paths += [vv_path.replace(".tif", "_crop.tif")]
                if self.vh:
                    x_paths += [vv_path.replace("VV.tif", "VH_crop.tif")]
            if self.ndvi_original:
                y_paths += [ndvi_path]
            if self.ndvi_crop:
                y_paths += [ndvi_crop_path]
            paths_dict[data_name].append(
                {"x": x_paths, "y": y_paths}
            )

        return paths_dict

    def load_tif_image(self, path: str):
        return rasterio.open(path).read(1)

    def load_dems(self):
        dem_folder = self.root + "/../dem/"
        place_names = ["001", "002", "003"]
        feature_names = ["dem", "slope", "aspect"]
        max_max_minus_min = {f: 0 for f in feature_names}
        dems = dict()
        for place_name in place_names:
            dems[place_name] = dict()
            for feature_name in feature_names:
                dems[place_name][feature_name] = \
                    self.load_tif_image(
                        dem_folder + place_name +
                        ("" if feature_name == "dem"
                         else ("_" + feature_name)) + "_filled.tif")
                if (dems[place_name][feature_name].max() -
                    dems[place_name][feature_name].min()) > \
                        max_max_minus_min[feature_name]:
                    max_max_minus_min[feature_name] = \
                        dems[place_name][feature_name].max() - \
                        dems[place_name][feature_name].min()
        for feature_name in ["dem", "slope"]:
            for place_name in place_names:
                dems[place_name][feature_name] = \
                    (dems[place_name][feature_name] -
                     dems[place_name][feature_name].min()) / \
                    max_max_minus_min[feature_name]
        for place_name in place_names:
            dems[place_name]["aspect"] /= 360

        return dems

    def load_images(self, paths_dict: dict, x_or_y: str, transform):
        selected_paths_list = list()
        xss_list = list()
        for key in sorted(paths_dict.keys()):
            xs_list = list()
            selected_pathss = list()
            for paths in paths_dict[key]:
                if self.metadata[self.split]["y_data"] \
                        and paths["y"][0] is None:
                    if self.non_target == "skip":
                        continue
                    else:
                        raise NotImplementedError()
                else:
                    xs = None
                    selected_paths = list()
                    for i, path in enumerate(paths[x_or_y]):
                        selected_paths.append(path)
                        x = np.array([self.load_tif_image(path)])
                        xs = x if i == 0 else np.concatenate((xs, x), axis=0)
                    if x_or_y == "x":
                        xs_list.append(
                            transform(xs,
                                      dems=self.dems[
                                          path.replace(self.root,
                                                       "").split("/")[2]]))
                    else:
                        xs_list.append(transform(xs))
                    selected_pathss.append(selected_paths)
            selected_paths_list.append(selected_pathss)
            xss_list.append(np.stack(xs_list, 0))
        self.selected_paths_list_dict[x_or_y] = selected_paths_list
        return xss_list


# Excluded date list for each area
EXCLUDE_LIST_002 = \
    ["20170501", "20170528", "20170630", "20170715", "20170722", "20170727",
     "20170804", "20170811", "20170816", "20170824", "20170829", "20170831",
     "20170913", "20170920", "20171003", "20171010", "20171015", "20171023",
     "20171028", "20171030", "20171104", "20171109", "20171112", "20171119",
     "20171129"]
EXCLUDE_LIST_003 = \
    ["20170501", "20170528", "20170610", "20170630", "20170715", "20170722",
     "20170727", "20170804", "20170811", "20170816", "20170824", "20170829",
     "20170831", "20170918", "20170920", "20171003", "20171010", "20171015",
     "20171023", "20171028", "20171030", "20171104", "20171109", "20171117",
     "20171119", "20171124"]
EXCLUDE_LIST_001 = \
    ["20170501", "20170630", "20170715", "20170722", "20170727", "20170811",
     "20170816", "20170829", "20170831", "20170913", "20170920", "20171003",
     "20171010", "20171015", "20171028", "20171030", "20171104", "20171109",
     "20171117", "20171119", "20171124"]


def get_sar_vh_list(area_id, crop=False):
    if area_id == "001":
        tif_folder_base = "dataset/test/%s/*/"
    else:
        tif_folder_base = \
            "dataset/train/%s/*_0/resolution_10x10/Sentinel-1_dsc/"

    if crop:
        sar_list = \
            sorted(glob.glob(tif_folder_base % area_id + "*_VH_crop.tif"))
    else:
        sar_list = \
            sorted(glob.glob(tif_folder_base % area_id + "*_VH.tif"))

    return sar_list


def get_planet_ndvi_list(area_id, crop=False):
    tif_folder_base = \
        "dataset/train/%s/*_0/resolution_10x10/PLANET/"
    if crop:
        ndvi_list = \
            sorted(glob.glob(tif_folder_base % area_id + "*_ndvi_crop.tif"))
    else:
        ndvi_list = \
            sorted(glob.glob(tif_folder_base % area_id + "*_ndvi.tif"))
    return ndvi_list


def get_region(area_id, epsg_code):
    from osgeo import gdal

    sar_vh = get_sar_vh_list(area_id)[0]
    ds_sar_vh = gdal.Open(sar_vh)
    if epsg_code == "32654":
        xmin, xres, xskew, ymin, yskew, yres = ds_sar_vh.GetGeoTransform()
        xmax = xmin + (ds_sar_vh.RasterXSize * xres)
        ymax = ymin + (ds_sar_vh.RasterYSize * yres)
        return [xmin, ymax, xmax, ymin]
    elif epsg_code == "4326":
        ds_trans = gdal.Warp("", ds_sar_vh,
                             format="MEM", dstSRS="EPSG:" + epsg_code)
        xmin, xres, xskew, ymin, yskew, yres = ds_trans.GetGeoTransform()
        xmax = xmin + (ds_trans.RasterXSize * xres)
        ymax = ymin + (ds_trans.RasterYSize * yres)
        xmin = float(Decimal(xmin).quantize(Decimal("1e-7")))
        xmax = float(Decimal(xmax).quantize(Decimal("1e-7")))
        ymin = float(Decimal(ymin).quantize(Decimal("1e-7")))
        ymax = float(Decimal(ymax).quantize(Decimal("1e-7")))
        ds_trans = None
        return [xmin, ymax, xmax, ymin]
    else:
        print("invalid epsg_code")
        return []


def create_shape_for_geotiff(region):
    shape = [{
        "type": "Polygon",
        "coordinates": [[[region[2], region[1]],
                         [region[0], region[1]],
                         [region[0], region[3]],
                         [region[2], region[3]],
                         [region[2], region[1]]]]
    }]
    return shape


def download_raw_sentinel2_images(area_id, output_dir,
                                  start_date="2017-05-01",
                                  end_date="2017-11-30"):
    os.makedirs(output_dir, exist_ok=True)

    # Get Image Collection by geometry and date
    geometry = ee.Geometry.Rectangle(get_region(area_id, "4326"))
    ic = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").select("B2", "B3",
                                                               "B4", "B8")
    ic = ic.filterDate(start_date, end_date).filterBounds(geometry)
    n = ic.size().getInfo()
    print(n, "image collections available and will be downloaded.")

    # Get URL lists from Image Collection
    ic_l = ic.toList(ic.size())
    kwargs = {"scale": 10, "crs": "EPSG:32654",
              "fileFormat": "GeoTIFF", "region": geometry}
    urls = [ee.Image(ic_l.get(i)).getDownloadURL(kwargs) for i in range(n)]
    # print(urls)

    # Create Cloudy Pixel Percentage map and save to json
    cpp_map = {}
    for i in range(n):
        image = ee.Image(ic_l.get(i))
        date = ee.Date(image.get("system:time_start")).format("YYYYMMdd")
        cpp = image.get("CLOUDY_PIXEL_PERCENTAGE")
        print("Timestamp:", date.getInfo(),
              "Cloud Pixel Percentage:", cpp.getInfo())
        cpp_map[date.getInfo()] = float(cpp.getInfo())
    with open(os.path.join(output_dir, "cpp.json"), "w") as cpp_json:
        json.dump(cpp_map, cpp_json)

    # Download image zip files and extract them
    for i, url in enumerate(urls):
        filename = os.path.join(output_dir, "temp" + str(i) + ".zip")
        with open(filename, "wb") as f:
            f.write(requests.get(url, stream=True).content)
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall(output_dir)
        os.remove(filename)

    # Merge RGB and create RGB geotif
    prefixs = [tif[:-6]
               for tif in sorted(glob.glob(os.path.join(output_dir,
                                                        "*B4.tif")))]
    for prefix in prefixs:
        date = os.path.basename(prefix)[:8]
        R = rasterio.open(prefix + "B4.tif")
        G = rasterio.open(prefix + "B3.tif")
        B = rasterio.open(prefix + "B2.tif")
        RGB = rasterio.open(os.path.join(output_dir,
                                         date + "_pre_rgb.tif"), "w",
                            driver="GTiff",
                            width=R.width, height=R.height,
                            count=3, crs=R.crs,
                            transform=R.transform, dtype=R.dtypes[0])
        RGB.write(R.read(1), 1)
        RGB.write(G.read(1), 2)
        RGB.write(B.read(1), 3)
        RGB.close()
        for b in ["2", "3"]:
            os.remove(prefix + "B" + b + ".tif")

    # Generate NDVI geotif
    prefixs = [tif[:-6]
               for tif in sorted(glob.glob(
                   os.path.join(output_dir, "*B4.tif")))]
    for prefix in prefixs:
        date = os.path.basename(prefix)[:8]
        R = rasterio.open(prefix + "B4.tif")
        NIR = rasterio.open(prefix + "B8.tif")
        NDVI = \
            rasterio.open(os.path.join(output_dir,
                                       date + "_pre_ndvi.tif"), "w",
                          driver="GTiff",
                          width=R.width, height=R.height,
                          count=1, crs=R.crs, transform=R.transform,
                          dtype=rasterio.float32)
        Rdata = R.read(1)
        NIRdata = NIR.read(1)
        NDVIdata = \
            np.where(((Rdata != 0) & (NIRdata != 0)),
                     (NIRdata.astype(float) - Rdata.astype(float)) /
                     (NIRdata.astype(float) + Rdata.astype(float)), 0)
        NDVI.write(NDVIdata, 1)
        NDVI.close()
        for b in ["4", "8"]:
            os.remove(prefix + "B" + b + ".tif")

    # Create shape for area cropping
    shape = create_shape_for_geotiff(get_region(area_id, "32654"))

    # Create ROI rgb tif
    tif_list = sorted(glob.glob(os.path.join(output_dir, "*_pre_rgb.tif")))
    for tif in tif_list:
        prefix = tif[:-12]
        with rasterio.open(tif) as src:
            out_image, out_transform = \
                rasterio.mask.mask(src, shape, crop=True)
            out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rasterio.open(prefix + "_rgb.tif", "w", **out_meta) as dest:
            dest.write(out_image)
        os.remove(tif)

    # Convert rgb tif to png
    tif_list = sorted(glob.glob(os.path.join(output_dir, "*rgb.tif")))
    for tif in tif_list:
        prefix = tif[:-4]
        with rasterio.open(tif) as src:
            # print(src.profile)
            data = src.read([1, 2, 3])
        img = (data * (255 / np.max(data))).astype(np.uint8)
        img = img.transpose((1, 2, 0))  # CHW > HWC
        img = PIL.Image.fromarray(img)
        img.save(prefix + ".png")
        os.remove(tif)

    # Create ROI ndvi tif
    tif_list = sorted(glob.glob(os.path.join(output_dir, "*_pre_ndvi.tif")))
    for tif in tif_list:
        prefix = tif[:-13]
        with rasterio.open(tif) as src:
            out_image, out_transform = \
                rasterio.mask.mask(src, shape, crop=True)
            out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rasterio.open(prefix + "_ndvi.tif", "w", **out_meta) as dest:
            dest.write(out_image)
        os.remove(tif)


def remove_by_exclude_list(area_id, output_dir):
    for file in sorted(glob.glob(os.path.join(output_dir, "*"))):
        for ex in eval("EXCLUDE_LIST_" + area_id):
            if os.path.basename(file).startswith(ex):
                # print(file)
                os.remove(file)


def generate_crop_image(data_list, crop_vh_file, viz=True):
    crop = rasterio.open(crop_vh_file)
    cropdata = crop.read(1)

    for i in data_list:
        src = rasterio.open(i)
        srcdata = src.read(1)

        if (src.profile["width"] != crop.profile["width"] or
                src.profile["height"] != src.profile["height"]):
            print("data inconsistency")

        crop_viz_img = np.where(cropdata != 0, srcdata, np.nan)
        crop_img = np.where(cropdata != 0, srcdata, -100)

        output_file = i[0:-4] + "_crop.tif"

        if viz:
            print(output_file)
            fig, axes = plt.subplots(1, 2, figsize=(8, 8))
            axes[0].imshow(srcdata)
            axes[1].imshow(crop_viz_img)
            plt.show()

        with rasterio.open(output_file, "w", **src.profile) as dst:
            dst.write(crop_img.astype(rasterio.float32), 1)


def load_lr_train_images(s2_list, planet_list):
    if len(s2_list) != len(planet_list):
        print("Found data inconsistency")

    ndvi_s2, ndvi_planet = [], []

    for i in range(len(s2_list)):
        s2 = rasterio.open(s2_list[i]).read(1)
        planet = rasterio.open(planet_list[i]).read(1)

        ndvi_s2.append(s2.ravel())
        ndvi_planet.append(planet.ravel())

    ndvi_s2 = np.concatenate(ndvi_s2, 0)
    ndvi_planet = np.concatenate(ndvi_planet, 0)

    return ndvi_s2, ndvi_planet


def train_lr_model(s2_list, planet_list):
    ndvi_s2, ndvi_planet = load_lr_train_images(s2_list, planet_list)
    reg_s2 = LinearRegression().fit(ndvi_s2.reshape(-1, 1), ndvi_planet)

    return reg_s2


def generate_adjusted_ndvi(reg, s2_list, output_dir, viz=True):
    os.makedirs(output_dir, exist_ok=True)

    for i in s2_list:
        s2 = rasterio.open(i)
        ndvi = s2.read(1)
        ndvi_shape = ndvi.shape
        adjusted_ndvi = \
            reg.predict(ndvi.ravel().reshape(-1, 1)).reshape(ndvi_shape)

        output_file = os.path.join(output_dir, os.path.basename(i))

        print(output_file)
        print("max:", ndvi.max(), adjusted_ndvi.max())
        print("min:", ndvi.min(), adjusted_ndvi.min())
        if viz:
            fig, axes = plt.subplots(1, 2, figsize=(8, 8))
            axes[0].imshow(ndvi)
            axes[1].imshow(adjusted_ndvi)
            plt.show()

        with rasterio.open(output_file, "w", **s2.profile) as dst:
            dst.write(adjusted_ndvi.astype(rasterio.float32), 1)


def exclude_s2_adj_by_cpp(area_id, s2_dir, s2_adj_dir):
    with open(os.path.join(s2_dir, "cpp.json")) as cpp_json:
        cpp_map = json.load(cpp_json)

    for i in sorted(glob.glob(os.path.join(s2_adj_dir, "*_ndvi.tif"))):
        if cpp_map[os.path.basename(i)[:8]] >= 50:
            crop_file = i[0:-4] + "_crop.tif"

            with rasterio.open(i) as src:
                profile = src.profile
                srcdata = src.read(1)
                dummy = np.full_like(srcdata, -100)

            print("Excluding", i)
            with rasterio.open(i, "w", **profile) as dst:
                dst.write(dummy.astype(rasterio.float32), 1)

            print("Excluding", crop_file)
            with rasterio.open(crop_file, "w", **profile) as dst:
                dst.write(dummy.astype(rasterio.float32), 1)


def generate_sentinel2_adjusted():
    # Initialize earthengine-api
    ee.Initialize()

    root_dir = "sentinel2/"
    s2_dir_prefix = root_dir + "raw_"
    s2_adj_dir_prefix = root_dir + "adjusted_"

    for data_str in ["001", "002", "003"]:
        shutil.rmtree(s2_dir_prefix + data_str, ignore_errors=True)
        download_raw_sentinel2_images(data_str, s2_dir_prefix + data_str)
        remove_by_exclude_list(data_str, s2_dir_prefix + data_str)
        generate_crop_image(sorted(glob.glob(
            os.path.join(s2_dir_prefix + data_str, "*_ndvi.tif"))),
            get_sar_vh_list(data_str, crop=True)[0], viz=False)

    # Create LR model to fit sentinel-2 data to planet data
    lr = train_lr_model(
        sorted(glob.glob(os.path.join(s2_dir_prefix + "002", "*_ndvi.tif"))) +
        sorted(glob.glob(os.path.join(s2_dir_prefix + "003", "*_ndvi.tif"))),
        get_planet_ndvi_list("002") + get_planet_ndvi_list("003"))

    # Generate 002 sentinel-2 adjusted data
    for data_str in ["001", "002", "003"]:
        shutil.rmtree(s2_adj_dir_prefix + data_str, ignore_errors=True)
        generate_adjusted_ndvi(lr, sorted(glob.glob(os.path.join(
            s2_dir_prefix + data_str, "*_ndvi.tif"))),
            s2_adj_dir_prefix + data_str, viz=False)
        generate_crop_image(sorted(glob.glob(os.path.join(
            s2_adj_dir_prefix + data_str, "*_ndvi.tif"))),
            get_sar_vh_list(data_str, crop=True)[0], viz=False)
        exclude_s2_adj_by_cpp(data_str,
                              s2_dir_prefix + data_str,
                              s2_adj_dir_prefix + data_str)


def get_elevation(location, no_data):
    url = "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/" + \
        "getelevation.php?lon={0[1]}&lat={0[0]}&outtype=JSON"
    time.sleep(0.01)
    r = json.loads(requests.get(url.format(location)).text)
    return r["elevation"] if r["hsrc"] == "5m（レーザ）" else no_data


def generate_dem(dem_dir, references, locations, no_data):
    dem_tif = os.path.join(dem_dir, "{}.tif")
    for reference, location in zip(references, locations):
        print("Getting DEM for %s" % reference)
        raster = rasterio.open(reference)
        transformer = Transformer.from_crs(raster.crs, "EPSG:6668")
        dataset = raster.read(1)
        height, width = dataset.shape[0], dataset.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
        elv = np.array(
            [[get_elevation(transformer.transform(xs[i][j], ys[i][j]), no_data)
              for j in tqdm(range(width), leave=False)]
             for i in tqdm(range(height), leave=True)], dtype=dataset.dtype)
        dem = rasterio.open(dem_tif.format(location), "w",
                            driver="GTiff", height=height, width=width,
                            count=1, dtype=dataset.dtype,
                            crs=raster.crs, transform=raster.transform)
        dem.write(elv, 1)
        dem.close()


def fill_elevation(dem_dir, locations, no_data):
    dem_tif = os.path.join(dem_dir, "{}.tif")
    dem_filled_tif = os.path.join(dem_dir, "{}_filled.tif")
    for location in locations:
        dem = rasterio.open(dem_tif.format(location))
        elv = dem.read(1)
        height, width = elv.shape[0], elv.shape[1]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        points, values = [], []
        for j in range(height):
            for i in range(width):
                if elv[j, i] > no_data + 1.0:
                    points.append([x[j, i], y[j, i]])
                    values.append(elv[j, i])
        filled = rasterio.open(dem_filled_tif.format(location), "w",
                               driver="GTiff", height=height, width=width,
                               count=1, dtype=elv.dtype,
                               crs=dem.crs, transform=dem.transform)
        filled.write(griddata(points=points, values=values,
                              xi=(x, y), method="nearest"), 1)
        filled.close()


def calculate_slope(dem_dir, locations, no_data):
    dem_filled_tif = os.path.join(dem_dir, "{}_filled.tif")
    slope_filled_tif = os.path.join(dem_dir, "{}_slope_filled.tif")
    aspect_filled_tif = os.path.join(dem_dir, "{}_aspect_filled.tif")

    for location in locations:
        dem = rd.LoadGDAL(dem_filled_tif.format(location), no_data=no_data)
        slope = rd.TerrainAttribute(dem, attrib="slope_degrees")
        aspect = rd.TerrainAttribute(dem, attrib="aspect")
        rd.SaveGDAL(slope_filled_tif.format(location), slope)
        rd.SaveGDAL(aspect_filled_tif.format(location), aspect)


def get_reference(root_dir):
    for _dir, _, _files in os.walk(root_dir):
        for _file in _files:
            if _file.endswith("_VH.tif"):
                return os.path.join(_dir, _file)


def generate_dems():

    dem_dir = "dem"
    dataset_dir = "dataset"
    os.makedirs(os.path.join(dem_dir), exist_ok=True)

    references = [
        get_reference(os.path.join(dataset_dir, "test", "001")),
        get_reference(os.path.join(dataset_dir, "train", "002")),
        get_reference(os.path.join(dataset_dir, "train", "003"))
    ]
    locations = ["001", "002", "003"]
    no_data = -100.0

    generate_dem(dem_dir, references, locations, no_data)
    fill_elevation(dem_dir, locations, no_data)
    calculate_slope(dem_dir, locations, no_data)


if __name__ == "__main__":
    generate_sentinel2_adjusted()
    generate_dems()
