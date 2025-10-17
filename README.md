# Rain over Africa

**Publication**: Amell, A., Hee, L., Pfreundschuh, S., & Eriksson, P. (2025). Probabilistic near‐real‐time retrievals of Rain over Africa using deep learning. *Journal of Geophysical Research: Atmospheres, 30*, e2025JD044595. https://doi.org/10.1029/2025JD044595

## Contents of this page
1. [Background](#1-background)
2. [The dataset](#2-the-dataset)
    1. [Data access](#21-data-access)
    2. [Reading RoA data](#22-reading-roa-data)
    3. [Dataset content](#23-dataset-content)
3. [How to use the data](#3-how-to-use-the-data)
4. [The code](#4-the-code)
    1. [How the public dataset is produced](#41-how-the-public-dataset-is-produced)
    2. [How to produce your own precipitation estimates](#42-how-to-produce-your-own-precipitation-estimates)
5. [How to cite](#5-how-to-cite)
6. [Acknowledgmenets](#6-acknowledgements)

## 1. Background

Satellites are the only way to continuously monitor rainfall across all of Africa. However, current methods for estimating rain from space can take a long time because they combine data from multiple sources.

We introduce Rain over Africa (RoA) [in this publication](https://doi.org/10.1029/2025JD044595), a new public method that can provide near-real-time precipitation estimates for Africa. The approach works by downloading a [Meteosat image](https://data.eumetsat.int/data/map/EO:EUM:DAT:MSG:HRSEVIRI) and processing it with an artificial neural network trained on precipitation estimates from the calibration satellite in the Global Precipitation Measurement (GPM) mission. Note that GPM, despite being a constellation of satellites, has less continuous coverage of Africa than Meteosat.

We found that RoA estimates show good agreement with estimates from dedicated precipitation sensors. Moreover, while the latter are available every few hours at best, RoA estimates can be updated every 15 minutes (or even faster\*). This makes RoA valuable for disaster preparedness and water management. Additionally, RoA provides practical probabilities of rain to help predict different scenarios, delivered as precipitation quantiles.

\*The code supports the latest Meteosat generation which offers a better revisit time, but we have not evaluated its accuracy.

## 2. The dataset

We use the term RoA not only to refer to the code that creates the precipitation estimates (detailed [in this section](#4-the-code)), but also for an existing dataset.

## 2.1. Data access

We are offering many years of RoA precipitation estimates through the Registry of Open Data on AWS at the following address: .

The data is stored as Zarr files in the following structure:
```
s3://rainoverafrica/
├── README.txt
└── data/
    ⋮
    ├── roa_2023.zarr/
    └── roa_2024.zarr/
```

We create one Zarr file per year and follow the pattern `roa_YYYY.zarr`. We are processing the data in batches.

If you need to explore further the directory tree, you can use the [AWS CLI](https://github.com/aws/aws-cli) or [fsspec's s3fs](https://github.com/fsspec/s3fs).

## 2.2. Reading RoA data

We recommend using [Xarray](https://docs.xarray.dev/en/stable/) with [s3fs](https://github.com/fsspec/s3fs). It is as simple as
```python
import xarray as xr

ds = xr.open_zarr(
    's3://rainoverafrica/data/roa_2024.zarr',
    chunks=None,
    storage_options={"anon": True},
)

# Select an arbitrary timestamp
ds_sel = ds.isel(sensing_end=0)

print(ds_sel)
```
```python
<xarray.Dataset> Size: 248MB
Dimensions:         (time: 1, y: 2688, x: 2304, quantile_level: 7)
Coordinates:
  * quantile_level  (quantile_level) float64 56B 0.05 0.16 0.25 ... 0.84 0.95
  * time            (time) datetime64[ns] 8B 2024-01-01T00:12:43.688000
  * x               (x) float64 18kB 4.783e+06 4.78e+06 ... -2.127e+06
  * y               (y) float64 22kB -4.033e+06 -4.03e+06 ... 4.027e+06 4.03e+06
Data variables:
    acq_time        (time, y) datetime64[ns] 22kB dask.array<chunksize=(1, 2688), meta=np.ndarray>
    latitude        (y, x) float32 25MB dask.array<chunksize=(336, 576), meta=np.ndarray>
    longitude       (y, x) float32 25MB dask.array<chunksize=(336, 576), meta=np.ndarray>
    platform        (time) <U11 44B dask.array<chunksize=(1,), meta=np.ndarray>
    y_hat_mu        (time, y, x) float32 25MB dask.array<chunksize=(1, 336, 576), meta=np.ndarray>
    y_hat_tau       (time, quantile_level, y, x) float32 173MB dask.array<chunksize=(1, 1, 672, 576), meta=np.ndarray>
Attributes:
    comment:      The attributes are based on the CF Conventions version 1.12...
    institution:  Chalmers University of Technology
    projection:   +proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +r...
    references:   Amell, A., Hee, L., Pfreundschuh, S., & Eriksson, P. (2025)...
    source:       Meteosat data processed with the `roa` library (version 0.3...
    title:        Rain over Africa
```
```python
# Load it to memory, i.e. transfer the data from AWS
ds_sel = ds_sel.compute()
```

More examples follow in the section [how to use the data](#3-how-to-use-the-data).

## 2.3. Dataset content

The variables in the dataset follow the [Climate and Forecast metadata conventions](https://en.wikipedia.org/wiki/Climate_and_Forecast_Metadata_Conventions). In any case, the table below compiles the meaning of each variable found in the input dataset.

| Variable | Meaning |
|--|--|
|`time` | end of the observation of the full Earth disc |
|`x` and `y` | coordinates, see note below |
|`quantile_level` | precipitation quantile level (relates to uncertainty) |
| `platform` | Source of the input image |
| `y_hat_mu` | Expected rain rate |
| `y_hat_tau` | Precipitation quantile (at level `tau = quantile_level`) |
| `acq_time` | Mean timestamp of the satellite scanline |
| `latitude` and `longitude` | lat and lon corresponding to the (`x`,`y`) coordinates |

Note that the data is provided on a projected Cartesian grid (`x` and `y`, rather than longitude and latitude). It corresponds to the native Meteosat (Second Generation) grid. The following [PROJ string](https://proj.org/) describes the projection:
```
+proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +rf=295.488065897001 +units=m +no_defs +type=crs
```

## 3. How to use the data

Note: The outputs with `TBC` and the examples will be updated once the data is in AWS.

We recommend to use Python and [Xarray](https://docs.xarray.dev/en/stable) to work with the data. In the example below, we show how we can:

- Compute the average precipitation in 2023, for every point in the grid

- Compute daily accumulations for 2023 for a specific location

- Reproduce the RoA diurnal cycles in fig. 2a from [Amell et al. (2025)](https://doi.org/10.1029/2025JD044595) 

We will use the the data stored in the [AWS S3 bucket](https://registry.opendata.aws/roa). For this, we will also need to have [fsspec's s3fs](https://github.com/fsspec/s3fs) installed. In this example, we are using Zarr version 3.1.3 (there are important differences between Zarr 2 and 3, see the [migration guide](https://zarr.readthedocs.io/en/main/user-guide/v3_migration)).

```python
# Import libraries
import numpy as np
from pyproj import Proj
import xarray as xr

# Open the data
ds = xr.open_zarr(
    f's3://rainoverafrica/data/roa_2023.zarr',
    chunks=None,
    storage_options={"anon": True}
)
ds.head()
```
```python
TBC
```
```python
# We can now compute the average for all of 2023, as simple as
ds_mean = ds.mean(dim='sensing_end')

# It will, however, only build a graph and not compute anything
# until we do ds_mean.compute()

# If we want to know the average precipitation at, for example,
# 0 °N and 0 °E (AKA the Null Island) for 2023, we then first need to find what `x` and `y`
# values are closest to the Null Island
roa_proj = Proj(
    '+proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +rf=295.488065897001 +units=m +no_defs +type=crs'
)
x, y = roa_proj.transform(0, 0)

ds_mean_at_null_island = ds_mean.sel(x=x, y=y, method='nearest')

# We now trigger computation
ds_mean_at_null_island = ds_mean_at_null_island.compute()
print(ds_mean_at_null_island)
```
```python
TBC
```
```python
# Let's compute the daily accumulations at the Null Island
# by integrating the rain rates
# To make this explicit, we will iterate through each day of the year
# and use some tricks with time
sensing_end_floored = ds.sensing_end.dt.floor('D').data
days_of_year = np.unique(sensing_end_floored)
ds_null_island = ds.sel(x=x, y=y, method='nearest')
daily_accumulations_at_null_island = np.full(days_of_year.size, np.nan)
for i, d in enumerate(days_of_year):
    ds_null_island_day_d = ds_null_island.sel(
        sensing_end=(sensing_end_floored == d)
    )
    # We extract the time as an integer representation of datetime64[ns]
    time = ds_null_island_day_d.sensing_end.astype('datetime64[ns]').astype(int)
    y_hat = ds_null_island_day_d.y_hat_mu.data
    finite_mask = np.isfinite(y_hat) # In case there's any invalid value
    daily_accumulations_at_null_island[i] = np.trapz(
        y_hat[finite_mask],
        # Below we divide by 3_600 x 10^9 to convert the nanosecond
        # representation to a fractional hour representation,
        # i.e. np.diff(time / 3600e9) is hours (used internally)
        # This matches the units of y_hat, as it is given in mm/h
        time / 3600e9
    )

print(daily_accumulations_at_null_island)
```
```python
TBC
```
```python
# We now want to compute the blue curve in fig. 2a from Amell et al. (2025)
# https://doi.org/10.1029/2025JD044595
#
# However, ds is in `x` and `y`, but we want to consider only the extent
# 20 - 25 °N and 0 - 5 °E
#
# The simplest is to extract a mask, as done below
xx, yy = np.broadcast_arrays(
    ds.x.data.reshape(1, -1),
    ds.y.data.reshape(-1, 1)
)

lon_xx, lat_yy = proj.transform(
    xx, yy,
    direction='INVERSE'
)

mask_lon = xr.DataArray(
    (0 <= lon_xx) & (lon_xx <= 5),
    coords={'y': ds.y, 'x': ds.x}
)
mask_lat = xr.DataArray(
    (20 <= lat_yy) & (lat_yy <= 25),
    coords={'y': ds.y, 'x': ds.x}
)

# We then select the June, July, and August months
ds_jja = ds.sel(sensing_end=slice('2023-06', '2023-08'))

# Before computing, we select only a subset of the data, to decrease the computational footprint
mask_x_crop = (mask == False).all('y') == False
mask_y_crop = (mask == False).all('x') == False
mask_cropped = mask.sel(x=mask_x_crop, y=mask_y_crop)

ds_jja_cropped = ds.sel(x=mask_x_crop, y=mask_y_crop)

# We now mask `ds_jja_cropped` to only pixels inside the designated area
ds_jja_cropped = ds_jja_cropped.where(mask_cropped)

# select `y_hat_mu`, the relevant variable
y_hat_mu_jja_cropped = ds_jja_cropped.sel(
    x=mask_x_crop, y=mask_y_crop
)

# and compute the mean in each 30 time bin, by using
# some tricks with time
y_hat_mu_jja_cropped_30min_mean['minute_of_day'] = xr.DataArray(
    minute_of_day,
    coords={'sensing_end', ds_jja_cropped.sensing_end}
)
y_hat_mu_jja_cropped_30min_mean = y_hat_mu_jja_cropped.groupby('minute_of_day').compute()
print(y_hat_mu_jja_cropped_30min_mean)
```
```
TBC
```

## 4. The code

For an [Apptainer container](https://apptainer.org/docs/admin/main/installation.html) containing all necessary libraries and libraries to execute a RoA retrieval, see https://doi.org/10.5281/zenodo.17193911 and the instructions therein.


Note that the RoA code is research code and can be affected by future updates in external resources. It is provided 'as-is'.

### 4.1. How the public dataset is produced

We use the [Apptainer container](https://apptainer.org/docs/admin/main/installation.html) we uploaded to [this Zenodo record](https://doi.org/10.5281/zenodo.17193911).

With

- a Linux machine,

- the container as `~/roa.sif`,

- a data directory at `/data/roa`,

- a 8 GiB NVIDIA Quadro RTX 4000 GPU to run inference,

- and EUMETSAT Data Store credentials saved in `~/.eumdac/credentials`,

we download data for one period (the full 2024, in this example):

```
$ apptainer run --bind /data/roa:/data ~/roa.sif \
    python /roa/scripts/MSG_download.py \
    --start 2024-01-01 --end 2025-01-01 \
    --output /data/MSG_data_2024.zarr
```
The end date is not included.

Note that some downloads can fail, so it is best to assert that all expected observations are successfully stored in the resulting Zarr file.

Afterwards, we run inference with
```
$ apptainer run --nv --bind /data/roa:/data ~/roa.sif \
    python /roa/scripts/inference_aws.py \
        --model /roa/data/network_CPU.pt \
        --input /data/MSG_data_2024.zarr \
        --output /data/roa_2024.zarr \
        --bs 100 \
        --quantiles 0.05 0.16 0.25 0.50 0.75 0.84 0.95 \
        --small_gpu
```
and that's it!

Note that different GPUs can introduce numerical differences when comparing to inference on CPUs.

### 4.2. How to produce your own precipitation estimates

You can either opt for running RoA with a similar approach as described in "[how the public dataset is produced](#41-how-the-public-dataset-is-produced)" or installing the RoA source code. Note that any software updates, for example support for the new Meteosat Third Generation, will likely not be reflected in the shared container. The basic requirements to install the `roa` repository are:
- Linux
- Python 3.10

To install it, execute:
```
$ pip install git+https://github.com/SEE-GEO/roa
```

You can also clone the repository and install locally. In any case, you need the file [`data/network_CPU.pt`](data/network_CPU.pt).

Note that many unused dependencies are installed as they are required by other dependencies. We do not guarantee that the PyTorch install correctly links to your GPU, if you have one.

For a walkthrough of a complete retrieval, check [`docs/example.ipynb`](docs/example.ipynb).

## 5. How to cite

Amell, A., Hee, L., Pfreundschuh, S., & Eriksson, P. (2025). Probabilistic near‐real‐time retrievals of Rain over Africa using deep learning. *Journal of Geophysical Research: Atmospheres, 130*, e2025JD044595. https://doi.org/10.1029/2025JD044595

If you also used data from the Registry of Open Data on AWS, consider using the statement "RoA data was accessed on [DATE] at registry.opendata.aws/roa"

## Acknowledgments

We would like to acknowledge:

- The [PyTroll](https://pytroll.github.io/) community.

- The National Academic Infrastructure for Supercomputing in Sweden ([NAISS](https://www.naiss.se)), partially funded by the Swedish Research Council through grant agreement no. 2022-06725, Chalmers e-Commons at Chalmers, and Chalmers AI Research Centre.

- The European Union’s HORIZON Research and Innovation Programme under grant agreement no. 101120657, project [ENFIELD](https://enfield-project.eu) (European Lighthouse to Manifest Trustworthy and Green AI).

- The [AWS Open Data Sponsorship Program](https://aws.amazon.com/opendata/open-data-sponsorship-program).
