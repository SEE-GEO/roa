import argparse
from pathlib import Path
import warnings

import numpy as np
from quantnn.quantiles import posterior_mean, sample_posterior
import torch
import tqdm
import xarray as xr

from roa.data import (
    prepare_dataset_for_network,
    QUANTILES,
)
from roa.tiler import Tiler

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
    ds_input = ds_input.sortby('nat_file')

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

            if args.sample:
                y_sample = sample_posterior(
                    y_hat,
                    quantiles_torch,
                    quantile_axis=1
                )

        ds_out = xr.Dataset(
            data_vars={
                "y_mu": (["nat_file", "y", "x"], y_mu.cpu().numpy()),
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

        ds_out = ds_out.rename({'nat_file': args.file_dim})
        ds_out.to_zarr(
            args.output,
            append_dim=args.file_dim if Path(args.output).exists() else None
        )