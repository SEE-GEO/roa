# Rain over Africa

**Publication**: Amell, A., Hee, L., Pfreundschuh, S., & Eriksson, P. (2025). Probabilistic near‐real‐time retrievals of Rain over Africa using deep learning. *Journal of Geophysical Research: Atmospheres, 130*, e2025JD044595. https://doi.org/10.1029/2025JD044595

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
6. [Acknowledgements](#6-acknowledgements)

## 1. Background

Satellites are the only way to continuously monitor rainfall across all of Africa. However, current methods for estimating rain from space can take a long time because they combine data from multiple sources.

We introduce Rain over Africa (RoA) [in this publication](https://doi.org/10.1029/2025JD044595), a new public method that can provide near-real-time precipitation estimates for Africa. The approach works by downloading a [Meteosat image](https://data.eumetsat.int/data/map/EO:EUM:DAT:MSG:HRSEVIRI) and processing it with an artificial neural network trained on precipitation estimates from the calibration satellite in the Global Precipitation Measurement (GPM) mission. Note that GPM, despite being a constellation of satellites, has less continuous coverage of Africa than Meteosat.

We found that RoA estimates show good agreement with estimates from dedicated precipitation sensors. Moreover, while the latter are available every few hours at best, RoA estimates can be updated every 15 minutes (or even faster\*). This makes RoA valuable for disaster preparedness and water management. Additionally, RoA provides practical probabilities of rain to help predict different scenarios, delivered as precipitation quantiles.

\*The code supports the latest Meteosat generation which offers a better revisit time, but we have not evaluated its accuracy.

## 2. The dataset

We use the term RoA not only to refer to the code that creates the precipitation estimates (detailed [in this section](#4-the-code)), but also for an existing dataset.

## 2.1. Data access

We are offering many years of RoA precipitation estimates via the Registry of Open Data on AWS at the following address: https://registry.opendata.aws/roa.

The data is stored as Zarr files in the following structure:
```
s3://rainoverafrica/
├── README.txt
└── data/
    ⋮
    ├── roa_2023.zarr/
    └── roa_2024.zarr/
```

We create one Zarr file per year and follow the pattern `roa_YYYY.zarr`. We are processing the data in batches. See the notes on the coordinates in [Reading RoA data](#22-reading-roa-data).

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
ds_sel = ds.isel(time=0)

print(ds_sel)
```
```
<xarray.Dataset> Size: 248MB
Dimensions:         (y: 2688, x: 2304, quantile_level: 7)
Coordinates:
  * y               (y) float64 22kB -4.033e+06 -4.03e+06 ... 4.027e+06 4.03e+06
  * x               (x) float64 18kB 4.783e+06 4.78e+06 ... -2.127e+06
  * quantile_level  (quantile_level) float64 56B 0.05 0.16 0.25 ... 0.84 0.95
    time            datetime64[ns] 8B 2024-01-01T00:12:43.688000
Data variables:
    acq_time        (y) datetime64[ns] 22kB dask.array<chunksize=(2688,), meta=np.ndarray>
    latitude        (y, x) float32 25MB dask.array<chunksize=(336, 576), meta=np.ndarray>
    longitude       (y, x) float32 25MB dask.array<chunksize=(336, 576), meta=np.ndarray>
    platform        <U11 44B dask.array<chunksize=(), meta=np.ndarray>
    y_hat_mu        (y, x) float32 25MB dask.array<chunksize=(336, 576), meta=np.ndarray>
    y_hat_tau       (quantile_level, y, x) float32 173MB dask.array<chunksize=(1, 672, 576), meta=np.ndarray>
Attributes:
    comment:      The attributes are based on the CF Conventions version 1.12...
    institution:  Chalmers University of Technology
    projection:   +proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +r...
    references:   Amell, A., Hee, L., Pfreundschuh, S., & Eriksson, P. (2025)...
    source:       Meteosat data processed with the `roa` library (version 0.0...
    title:        Rain over Africa
```
```python
# Load it to memory, i.e. transfer the data from AWS
ds_sel = ds_sel.compute()
```

More examples follow in the section [how to use the data](#3-how-to-use-the-data).

**Note on the coordinates for data on AWS**: Retrievals that do not correspond to the first timestamp of the year can be misaligned by half a pixel (1.5 km). It is not documented which timestamps suffer this misalignment. To concatenate data from different years that is not aligned, [`xarray.Dataset.reindex`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.reindex.html) or [`xarray.align`](https://docs.xarray.dev/en/latest/generated/xarray.align.html) can, for example, be used.

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

We recommend to use Python and [Xarray](https://docs.xarray.dev/en/stable) to work with the data. In the example below, we show how we can:

- Compute the average precipitation in 2023, for every point in the grid

- Compute daily accumulations for 2023 for a specific location

- Reproduce the RoA diurnal cycles in fig. 2a from [Amell et al. (2025)](https://doi.org/10.1029/2025JD044595) 

We will use the the data stored in the [AWS S3 bucket](https://registry.opendata.aws/roa). For this, we will also need to have [fsspec's s3fs](https://github.com/fsspec/s3fs) installed. In this example, we are using Zarr version 3.1.3 (there are important differences between Zarr 2 and 3, see the [migration guide](https://zarr.readthedocs.io/en/main/user-guide/v3_migration)). We will also run the code using a small computer, a 2-CPU and 8-GB m7i-flex AWS EC2 free tier instance.

```python
# Import libraries
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr

# Open the data
ds = xr.open_zarr(
    f's3://rainoverafrica/data/roa_2023.zarr',
    storage_options={"anon": True}
)
ds.head()
```
```
<xarray.Dataset> Size: 4kB
Dimensions:         (time: 5, y: 5, x: 5, quantile_level: 5)
Coordinates:
  * time            (time) datetime64[ns] 40B 2023-01-01T00:12:43.100000 ... ...
  * y               (y) float64 40B -4.033e+06 -4.03e+06 ... -4.021e+06
  * x               (x) float64 40B 4.783e+06 4.78e+06 ... 4.774e+06 4.771e+06
  * quantile_level  (quantile_level) float64 40B 0.05 0.16 0.25 0.5 0.75
Data variables:
    acq_time        (time, y) datetime64[ns] 200B dask.array<chunksize=(1, 5), meta=np.ndarray>
    latitude        (y, x) float32 100B dask.array<chunksize=(5, 5), meta=np.ndarray>
    longitude       (y, x) float32 100B dask.array<chunksize=(5, 5), meta=np.ndarray>
    platform        (time) <U11 220B dask.array<chunksize=(1,), meta=np.ndarray>
    y_hat_mu        (time, y, x) float32 500B dask.array<chunksize=(1, 5, 5), meta=np.ndarray>
    y_hat_tau       (time, quantile_level, y, x) float32 2kB dask.array<chunksize=(1, 1, 5, 5), meta=np.ndarray>
Attributes:
    comment:      The attributes are based on the CF Conventions version 1.12...
    institution:  Chalmers University of Technology
    projection:   +proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +r...
    references:   Amell, A., Hee, L., Pfreundschuh, S., & Eriksson, P. (2025)...
    source:       Meteosat data processed with the `roa` library (version 0.0...
    title:        Rain over Africa
```
```python
# We can now compute the average for all of 2023, as simple as
ds_y_hat_mu_mean = ds.y_hat_mu.where((0<= ds.y_hat_mu) & (ds.y_hat_mu <= 200)).mean(dim='time')

# It will, however, only build a graph and not compute anything
# until we do `ds_y_hat_mu_mean.compute()`
# Also note that it's a safe practice to mask values below a certain threshold,
# hence our use of .where((0<= ds.y_hat_mu) & (ds.y_hat_mu <= 200))
# as they can be the result of numerical issues. See the note from section 4.1.

# If we want to know the average precipitation at, for example,
# 0 °N and 0 °E (AKA the Null Island) for 2023, we then first need to find what `x` and `y`
# values are closest to the Null Island
# We can use the auxiliary variables `latitude` and `longitude` for this,
# using an approximation by computing the Euclidean distance
i_y, i_x = np.unravel_index(
    (
        abs(ds.latitude - 0)**2 + abs(ds.longitude - 0)**2
    ).argmin().compute().item(),
    ds.latitude.shape
)

ds_y_hat_mu_mean_at_null_island = ds_y_hat_mu_mean.isel(y=i_y, x=i_x)

# We now trigger computation
with ProgressBar():
    ds_y_hat_mu_mean_at_null_island = ds_y_hat_mu_mean_at_null_island.compute()
```
```
[########################################] | 100% Completed | 26m 10s
```
```python
print(ds_mean_at_null_island)
```
```
<xarray.DataArray 'y_hat_mu' ()> Size: 4B
array(0.13609327, dtype=float32)
Coordinates:
    x        float64 8B 0.0
    y        float64 8B 0.0
Attributes:
    comment:        There can be negative values as well as outliers. These c...
    description:    Expected value from the rain rate retrieval distribution.
    long_name:      expected rain rate
    projection:     +proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 ...
    standard_name:  rainfall_rate
    units:          mm h-1
```
```python
# Let's compute the daily accumulations at the Null Island
# by integrating the rain rates
# To make this explicit, we will iterate through each day of the year
# and use some tricks with time
ds_null_island = ds.isel(y=i_y, x=i_x)
day_of_year = ds_null_island.time.dt.dayofyear.data
day_of_year_unique = np.unique(day_of_year)
daily_accumulations_at_null_island = np.full(
    day_of_year_unique.size,
    np.nan
)
print("Processing daily accumulations at the Null Island\n")
for i, d in enumerate(day_of_year_unique):
    print(f"Day {d}", end='\r')
    ds_null_island_day_d = ds_null_island.sel(
        time=(day_of_year == d)
    )
    # We extract the time as an integer representation of datetime64[ns]
    time = ds_null_island_day_d.time.astype('datetime64[ns]').astype(int).values
    y_hat = ds_null_island_day_d.y_hat_mu.where(
        (0 <= ds_null_island_day_d.y_hat_mu) &
        (ds_null_island_day_d.y_hat_mu <= 200)
    ).values
    finite_mask = np.isfinite(y_hat) # In case there's any invalid value
    daily_accumulations_at_null_island[i] = np.trapezoid(
        y_hat[finite_mask],
        # Below we divide by 3_600 x 10^9 to convert the nanosecond
        # representation to a fractional hour representation,
        # i.e. np.diff(time / 3600e9) is hours (used internally)
        # This matches the units of y_hat, as it is given in mm/h
        time[finite_mask] / 3600e9
    )

# Show the first 5 accumulations
print('\nFirst five daily accumulations:\n', daily_accumulations_at_null_island[:5])
```
```
First five daily accumulations:
 [0.10510297 0.63085772 1.39207075 4.37375558 0.54879636]
```
```python
# We now want to compute the blue curve in fig. 2a from Amell et al. (2025)
# https://doi.org/10.1029/2025JD044595
# 
# We want to consider only the extent 20 - 25 °N and 0 - 5 °E,
# and we can make use of the auxiliary latitude and longitude variables
mask_lat = xr.DataArray(
    (20 <= ds.latitude) & (ds.latitude <= 25),
    coords={'y': ds.y, 'x': ds.x}
)
mask_lon = xr.DataArray(
    (0 <= ds.longitude) & (ds.longitude <= 5),
    coords={'y': ds.y, 'x': ds.x}
)
mask = mask_lat & mask_lon

# Select June, July, and August months
ds_jja = ds.sel(time=slice('2023-06', '2023-08'))

# Crop to the mask, where NaNs will populate elements outside the area
# .compute() required for the index
ds_jja_cropped = ds_jja.where(mask.compute(), drop=True)

# Select the relevant variable
y_hat_mu_jja_cropped = ds_jja_cropped.y_hat_mu
# Handle any possible conflicting value
y_hat_mu_jja_cropped = y_hat_mu_jja_cropped.where(
    (0 <= y_hat_mu_jja_cropped) & (y_hat_mu_jja_cropped <= 200)
)

# and compute the mean in each 30 time bin, by using
# some tricks with time
y_hat_mu_jja_cropped['minute_of_day'] = y_hat_mu_jja_cropped.time.dt.floor('30min').pipe(lambda x: x.dt.hour * 60 + x.dt.minute)
y_hat_mu_jja_cropped = y_hat_mu_jja_cropped.swap_dims({'time': 'minute_of_day'}).drop_vars('time')
y_hat_mu_jja_cropped_30min_mean = y_hat_mu_jja_cropped.groupby(
    'minute_of_day'
).mean(
    ['minute_of_day', 'x', 'y']
)

with ProgressBar():
    y_hat_mu_jja_cropped_30min_mean = y_hat_mu_jja_cropped_30min_mean.compute()
```
```
[########################################] | 100% Completed | 266.23 s
```
```python
print(y_hat_mu_jja_cropped_30min_mean)
```
```
<xarray.DataArray 'y_hat_mu' (minute_of_day: 48)> Size: 192B
array([0.03006615, 0.02757636, 0.02362311, 0.02234554, 0.02091054,
       0.01890199, 0.01593699, 0.01270751, 0.01106855, 0.00951752,
       0.00843452, 0.00691009, 0.00584777, 0.00527937, 0.00428771,
       0.00387396, 0.00305142, 0.00227556, 0.00178786, 0.00121699,
       0.00087936, 0.00063457, 0.00056154, 0.00068697, 0.00151117,
       0.00427322, 0.00925402, 0.01717995, 0.02930276, 0.04330863,
       0.05582752, 0.06514046, 0.07072729, 0.07282268, 0.07208262,
       0.06770573, 0.06500953, 0.06011817, 0.05709378, 0.052485  ,
       0.04922819, 0.04561045, 0.04238486, 0.03960717, 0.03687198,
       0.03620769, 0.03495797, 0.03342935], dtype=float32)
Coordinates:
  * minute_of_day  (minute_of_day) int64 384B 0 30 60 90 ... 1320 1350 1380 1410
Attributes:
    comment:        There can be negative values as well as outliers. These c...
    description:    Expected value from the rain rate retrieval distribution.
    long_name:      expected rain rate
    projection:     +proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 ...
    standard_name:  rainfall_rate
    units:          mm h-1
```

## 4. The code

For an [Apptainer container](https://apptainer.org/docs/admin/main/installation.html) containing all necessary libraries and libraries to execute a RoA retrieval, see https://doi.org/10.5281/zenodo.17193911 and the instructions therein.


Note that the RoA code is research code and can be affected by future updates in external resources. It is provided 'as-is'.

### 4.1. How the public dataset is produced

We use the [Apptainer container](https://apptainer.org/docs/admin/main/installation.html) we uploaded to [this Zenodo record](https://doi.org/10.5281/zenodo.17193911). The `roa` library in the container can be updated as necessary.

With

- a Linux machine,

- the container as `~/roa.sif`,

- a data directory at `/data/roa`,

- a 8 GiB NVIDIA Quadro RTX 4000 GPU to run inference (alternatively, an A40 or A100 increasing the batch size and removing `--small-gpu`),

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
and that's it! The dataset is uploaded with `scripts/upload.py`.

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

## Acknowledgements

We would like to acknowledge:

- The [PyTroll](https://pytroll.github.io/) community.

- The National Academic Infrastructure for Supercomputing in Sweden ([NAISS](https://www.naiss.se)), partially funded by the Swedish Research Council through grant agreement no. 2022-06725, Chalmers e-Commons at Chalmers, and Chalmers AI Research Centre.

- The European Union’s HORIZON Research and Innovation Programme under grant agreement no. 101120657, project [ENFIELD](https://enfield-project.eu) (European Lighthouse to Manifest Trustworthy and Green AI).

- The [AWS Open Data Sponsorship Program](https://aws.amazon.com/opendata/open-data-sponsorship-program).
