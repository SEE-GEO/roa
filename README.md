# Rain over Africa

**Publication**: Amell, A., Hee, L., Pfreundschuh, S., & Eriksson, P. (2025). Probabilistic near‐real‐time retrievals of Rain over Africa using deep learning. *Journal of Geophysical Research: Atmospheres, 130*, e2025JD044595. https://doi.org/10.1029/2025JD044595

## Contents of this page
1. [Background](#1-background)
2. [The dataset](#2-the-dataset)
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

We are offering many years of RoA precipitation estimates through the Registry of Open Data on AWS at the following address: https://registry.opendata.aws/roa.

The data is stored as Zarr files in the following structure:
```
s3://roa/
├── README.txt
└── data
    ⋮
    ├── roa_2023.zarr
    └── roa_2024.zarr
```

The variables follow the [Climate and Forecast metadata conventions](https://en.wikipedia.org/wiki/Climate_and_Forecast_Metadata_Conventions). In any case, the table below compiles the meaning of each variable found in the input dataset.

However, the table below compiles the meaning of each variable:
| Variable | Meaning |
|--|--|
|`sensing_end` | stop acquisition time of the full Earth disc |
|`x` and `y` | coordinates, see note below |
|`quantile_level` | precipitation quantile level (relates to uncertainty) |
| `meteosat` | Source of the input image |
| `y_hat_mu` | Expected rain rate |
| `y_hat_tau` | Precipitation quantile (at level `tau = quantile_level`) |
| `acq_time` | Timestamp of the satellite scanline |

Note that the data is offered on a projected Cartesian grid (`x` and `y`, rather than longitude and latitude). IT corresponds to the native Meteosat (Second Generation) grid, and the following PROJ string describes the projection:
```
+proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +rf=295.488065897001 +units=m +no_defs +type=crs
```

## 3. How to use the data

This is to be completed

## 4. The code

For an [Apptainer container](https://apptainer.org/docs/admin/main/installation.html) containing all necessary libraries and libraries to execute a RoA retrieval, see https://doi.org/10.5281/zenodo.17193911 and the instructions therein.


Note that the RoA code is research code and can be affected by future updates in external resources. It is provided 'as-is'.

### 4.1. How the public dataset is produced

We use the [Apptainer container](https://apptainer.org/docs/admin/main/installation.html) we uploaded to [this Zenodo record](https://doi.org/10.5281/zenodo.17193911).

With

- a Linux machine,

- the container as `~/roa.sif`,

- a data directory at `/data/roa`,

- a 40 GiB NVIDIA A100 GPU to run inference,

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
    python /roa/scripts/inference_cf.py \
        --model /roa/data/network_CPU.pt \
        --input /data/MSG_data_2024.zarr \
        --output /data/roa_2024.zarr \
        --bs 256 \
        --quantiles 0.16 0.84
```
and that's it!

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

TBC

## Acknowledgments

We would like to acknowledge:

- The [PyTroll](https://pytroll.github.io/) community.

- The National Academic Infrastructure for Supercomputing in Sweden ([NAISS](https://www.naiss.se)), partially funded by the Swedish Research Council through grant agreement no. 2022-06725, Chalmers e-Commons at Chalmers, and Chalmers AI Research Centre.

- The European Union’s HORIZON Research and Innovation Programme under grant agreement no. 101120657, project [ENFIELD](https://enfield-project.eu) (European Lighthouse to Manifest Trustworthy and Green AI).

- The [AWS Open Data Sponsorship Program](https://aws.amazon.com/opendata/open-data-sponsorship-program)
