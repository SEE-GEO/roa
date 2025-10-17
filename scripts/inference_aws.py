"""
This script is a copy of inference.py
with more metadata to prepare publicly sharing the data on AWS.

The metadata is based on the CF Conventions version 1.12.

It does not support FCI inference, only MSG.
"""

import argparse
import datetime
from pathlib import Path
import warnings

import numpy as np
from quantnn.quantiles import posterior_mean, sample_posterior
import torch
import tqdm
import xarray as xr

import roa
from roa.data import (
    prepare_dataset_for_network,
    QUANTILES,
    SEVIRI_0DEG_AREADEF
)
from roa.tiler import Tiler
from roa.utils import mask_invalid_rates

class CFConventions:

    projection = '+proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +rf=295.488065897001 +units=m +no_defs +type=crs'
    
    def __init__(self, ds: xr.Dataset, copy: bool = False):
        self.ds = ds.copy(deep=copy)
        self._nat_file = ds.nat_file.item()
        assert 'latitude' in self.ds
        assert 'longitude' in self.ds

    def apply(self) -> xr.Dataset:
        self._rename_vars()
        self.ds['time'] = [self.get_sensing_end(self._nat_file)]
        self.ds['platform'] = ('time', np.array([f'Meteosat-{self._get_meteosat()}'], dtype='<U11'))
        self._set_dim_order()
        self._set_data_attrs()
        self._set_global_attrs()
        self._fix_time_encoding()
        return self.ds
    
    def _fix_time_encoding(self):
        self.ds['time'].encoding['units'] = 'milliseconds since 1970-01-01'
        self.ds['acq_time'].encoding['units'] = 'milliseconds since 1970-01-01'

    @staticmethod
    def get_sensing_end(nat_file) -> datetime.datetime:
        sensing_end = datetime.datetime.strptime(
            nat_file.split('-')[-2],
            '%Y%m%d%H%M%S.%f000Z'
        )
        return sensing_end
    
    def _get_meteosat(self) -> np.uint8:
        return np.ubyte(self._nat_file.split('-')[0][-1])
    
    def _rename_vars(self):
        self.ds = self.ds.rename({'nat_file': 'time'})
        for v in ['y_mu', 'y_tau', 'y_sample']:
            if v in self.ds:
                self.ds = self.ds.rename({v: v.replace('y_', 'y_hat_')})

    def _set_dim_order(self):
        self.ds = self.ds.transpose('time', 'quantile_level', 'y', 'x')

    def _set_global_attrs(self):
        self.ds.attrs['title'] = 'Rain over Africa'
        self.ds.attrs['institution'] = 'Chalmers University of Technology'
        self.ds.attrs['source'] = f'Meteosat data processed with the `roa` library (version {roa.__version__}).'
        self.ds.attrs['references'] = 'Amell, A., Hee, L., Pfreundschuh, S., & Eriksson, P. (2025). Probabilistic near-real-time retrievals of Rain over Africa using deep learning. Journal of Geophysical Research: Atmospheres, 130, e2025JD044595. https://doi.org/10.1029/2025JD044595' + '\n' + 'https://github.com/SEE-GEO/roa'
        # new_history = ' '.join(
        #     [
        #         datetime.datetime.now(datetime.timezone.utc).isoformat(),
        #         '-',
        #         f'Processed {self._nat_file} with roa {roa.__version__}'
        #     ]
        # )

        # if 'history' in self.ds.attrs:
        #     self.ds.attrs['history'] += '\n' + new_history
        # else:
        #     self.ds.attrs['history'] = new_history
        self.ds.attrs['comment'] = '\n'.join(
            [
                'The attributes are based on the CF Conventions version 1.12.',
                'The data are provided on a projection grid centred on the sub-satellite point of the Meteosat satellite, defined by the `projection` attribute.',
                'There is no history attribute as the processing pipeline does not support appending to the history attribute.'
            ]
        )
        self.ds.attrs['projection'] = self.projection

    def _set_data_attrs(self):
        if 'y_hat_mu' in self.ds:
            self.ds['y_hat_mu'].attrs = {
                'long_name': 'expected rain rate',
                'standard_name': 'rainfall_rate',
                'units': 'mm h-1',
                'description': 'Expected value from the rain rate retrieval distribution.',
                'projection': self.projection,
                'comment': 'There can be negative values as well as outliers. These can be caused by numerical issues and are safe to ignore. One can only include, for example, only values between 0 and 200 mm h-1 for practical applications.'
            }
        
        if 'y_hat_tau' in self.ds:
            self.ds['y_hat_tau'].attrs = {
                'long_name': 'Rain rate quantile',
                'standard_name': 'rainfall_rate',
                'units': 'mm h-1',
                'description': 'Quantile of the rain rate retrieval distribution.',
                'projection': self.projection
            }

        if 'y_hat_sample' in self.ds:
            self.ds['y_hat_sample'].attrs = {
                'long_name': 'Rain rate sample',
                'standard_name': 'rainfall_rate',
                'units': 'mm h-1',
                'description': 'Random sample from the rain rate retrieval distribution.',
                'projection': self.projection
            }
        
        for v in ['x', 'y']:
            self.ds[v].attrs = {
                'units': 'metres',
                'long_name': 'Metres from sub-satellite point',
                'standard_name': f'projection_{v}_coordinate',
                'description': f'Distance in metres from the sub-satellite point in the {v}-direction.',
                'ancilliary_variables': 'latitude longitude',
                'projection': self.projection
            }
        
        self.ds['latitude'].attrs = {
            'long_name': 'Latitude',
            'standard_name': 'latitude',
            'description': 'Latitude coordinate corresponding to each grid point.',
            'units': 'degrees_north'
        }

        self.ds['longitude'].attrs = {
            'long_name': 'Longitude',
            'standard_name': 'longitude',
            'description': 'Longitude coordinate corresponding to each grid point.',
            'units': 'degrees_east'
        }

        self.ds['platform'].attrs = {
            'long_name': 'Meteosat satellite number',
            'standard_name': 'platform_name',
            'description': 'Identifier for the Meteosat satellite from which the data was acquired.',
            'units': '1'
        }

        self.ds['time'].attrs = {
            'long_name': 'Observation end time',
            'standard_name': 'time',
            'description': 'Time at which the observation was completed.',
        }

        self.ds['acq_time'].attrs = {
            'long_name': 'Mean scanline acquisition time',
            'standard_name': 'time',
            'description': 'Mean acquisition time for each scanline in the observation.',
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--left_index_fraction', type=float, default=0)
    parser.add_argument('--right_index_fraction', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quantiles', type=float, nargs='*')
    parser.add_argument('--bs', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--small_gpu', action='store_true', help='Use little GPU memory')
    parser.add_argument(
        '--file_dim', type=str, default='nat_file',
        choices=['nat_file', 'file_id'],
        help='Dimension name for the file index in the input dataset'
    )
    parser.add_argument(
        '--subset',
        help='File listing subset of files to process (or a single file name)'
    )
    parser.add_argument(
        '--mask_invalid_rates', type=float,
        help='Mask out pixels with rates above this threshold'
    )

    args = parser.parse_args()

    quantile_levels = np.array(args.quantiles) if args.quantiles else None
    if quantile_levels is not None:
        # Follows the convention of RoA, where quantiles are 0.01, 0.02, ..., 0.99
        quantile_levels_idx = np.round(quantile_levels * 100).astype(int) - 1

    device = torch.device(args.device)
    device_cpu = torch.device('cpu')
    if args.small_gpu and (device.type == 'cuda'):
        device_y_hat = device_cpu
    else:
        device_y_hat = device
    quantiles_torch = torch.from_numpy(QUANTILES).to(device_y_hat, dtype=torch.float32)

    bs = args.bs

    # Load network
    model = torch.jit.load(
        args.model,
        map_location=device
    )

    # Open input data (force coordinate to be `nat_file` for compatibility with posterior projects)
    ds_input = xr.open_zarr(args.input).rename({args.file_dim: 'nat_file'})
    ds_input = ds_input.drop_duplicates('nat_file')

    # Sort by end observation time
    ds_input = ds_input.isel(nat_file=np.argsort([CFConventions.get_sensing_end(nf) for nf in ds_input.nat_file.data]))

    if args.subset:
        if Path(args.subset).exists():
            with open(args.subset) as f:
                subset = f.read().splitlines()
        else:
            assert not (' ' in args.subset)
            subset = [args.subset]
        ds_input = ds_input.sel(nat_file=subset)

    left_index = int(args.left_index_fraction * ds_input.nat_file.size)
    right_index = int(args.right_index_fraction * ds_input.nat_file.size)

    # Compute latitude and longitude coordinates before any processing
    lon, lat = SEVIRI_0DEG_AREADEF.get_lonlat_from_projection_coordinates(
        *np.broadcast_arrays(
            ds_input.x.data.reshape(1, -1),
            ds_input.y.data.reshape(-1, 1)
        )
    )
    mask_lonlat = ~(np.isfinite(lon) & np.isfinite(lat))
    lon[mask_lonlat] = np.nan
    lat[mask_lonlat] = np.nan

    lon = xr.DataArray(
        lon.astype('float32'),
        coords={'y': ds_input.y.data, 'x': ds_input.x.data},
        name='longitude'
    )
    lat = xr.DataArray(
        lat.astype('float32'),
        coords={'y': ds_input.y.data, 'x': ds_input.x.data},
        name='latitude'
    )

    for nat_file in tqdm.tqdm(ds_input.nat_file.values[left_index:right_index], ncols=80):
        # Prepare the dataset for the network
        ds_in = ds_input.sel(nat_file=nat_file)
        x = prepare_dataset_for_network(ds_in)[None, ...]
        tiler = Tiler(x)

        # Get tiles and put them in the batch dimension
        x = torch.stack(
                [tiler.get_tile(i,j) for i in range(tiler.M) for j in range(tiler.N)],
                dim=0
            )
        x = x.view(-1, *x.shape[2:])

        # Run inference
        dtype = x.dtype
        # Preallocate output tensor
        y_hat = torch.full(
            (len(x), QUANTILES.size, *tiler.tile_size),
            torch.nan,
            dtype=dtype,
            device=device_y_hat
        )

        for i in range(0, len(x), bs):
            y_hat[i:(i+bs)] = model(x[i:(i+bs)].to(device)).to(device_y_hat)

        # Assemble the tiles
        y_hat = y_hat.view(tiler.M, tiler.N, 1, *y_hat.shape[1:])
        y_hat = tiler.assemble(y_hat)

        # Compute statistics
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            y_mu = posterior_mean(
                y_hat,
                quantiles_torch,
                quantile_axis=1
            )

            if args.mask_invalid_rates is not None:
                y_mu = mask_invalid_rates(y_mu, args.mask_invalid_rates)

            if args.sample:
                y_sample = sample_posterior(
                    y_hat,
                    quantiles_torch,
                    quantile_axis=1
                )
                if args.mask_invalid_rates is not None:
                    y_sample = mask_invalid_rates(y_sample, args.mask_invalid_rates)

            if args.mask_invalid_rates is not None:
                y_hat = mask_invalid_rates(y_hat, args.mask_invalid_rates)

        ds_out = xr.Dataset(
            data_vars={
                "y_mu": (
                    ["nat_file", "y", "x"],
                    y_mu.cpu().numpy()
                ),
            } | (
                {"y_sample": (["nat_file", "y", "x"], y_sample.squeeze(axis=1).cpu().numpy())} if args.sample else {}
            ),
            coords={
                "x": ds_input.x.data,
                "y": ds_input.y.data,
                "nat_file": [nat_file],
            }
        )
        if quantile_levels is not None:
            ds_out = xr.merge(
                (
                    ds_out,
                    xr.Dataset(
                        data_vars={
                            "y_tau": (["nat_file", "quantile_level", "y", "x"], y_hat[:, quantile_levels_idx].cpu().numpy()),
                        },
                        coords={
                            "x": ds_input.x.data,
                            "y": ds_input.y.data,
                            "nat_file": [nat_file],
                            "quantile_level": quantile_levels,
                        }
                    )
                )
            )

        # Copy acq_time to output
        ds_out = xr.merge((ds_out, ds_in.acq_time.expand_dims({'nat_file': [nat_file]})))

        # Add latitude and longitude coordinates
        ds_out = xr.merge((ds_out, lon, lat))

        ds_out = CFConventions(ds_out).apply()

        ds_out.to_zarr(
            args.output,
            append_dim='time' if Path(args.output).exists() else None
        )