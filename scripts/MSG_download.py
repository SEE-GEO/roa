"""
Script to download MSG data into a Zarr file,
one month resolution.
"""

import argparse
import concurrent.futures
from pathlib import Path
import shutil
import tempfile
from threading import Lock
import time
import warnings

import eumdac
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from roa.data import MSGNative

def download_file(product, lock, output_file, first=False, area_extent=None):
    # The `while True` is a hack to cope with EUMETSAT not serving data
    nat_file = f'{str(product)}.nat'
    # Use /dev/shm to avoid writing to disk, in the current system
    # it backed by RAM
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdirname:
        file_path = Path(tmpdirname) / nat_file
        while True:
            while True:
                try:
                    with product.open(entry=nat_file) as fsrc, \
                        open(file_path, mode='wb') as handle:
                        shutil.copyfileobj(fsrc, handle)
                        break
                except:
                    time.sleep(60)
                    continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", 
                        message="^The quality flag for this file indicates not OK."
                    )
                    if area_extent:
                        ds = MSGNative(file_path).get_dataset(area_extent=area_extent).load()
                    else:
                        ds = MSGNative(file_path).get_dataset().load()
                break
            except:
                continue
    
    ds = ds.assign_coords({"nat_file": nat_file}).expand_dims("nat_file")

    # Clear attributes that are not serializable
    for v in ds:
        ds[v].attrs = {}

    with lock:
        if first:
            ds.to_zarr(output_file)
        else:
            ds.to_zarr(output_file, append_dim="nat_file")

def get_area_extent(area_extent):
    return {
        'y': np.arange(area_extent[1], area_extent[3]),
        'x': np.arange(area_extent[0], area_extent[2])
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MSG data")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date")
    parser.add_argument("--credentials", type=str,
                        default=Path("~/.eumdac/credentials").expanduser(),
                        help="EUMDAC credentials file")
    parser.add_argument("--output", type=str, help="Output Zarr", required=True)
    parser.add_argument("--workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--area_extent", nargs=4, type=int)
    parser.add_argument("--files", type=str,
                        help="File with list of files to download, one .nat file per line")
    args = parser.parse_args()

    # Fix area extent
    area_extent_idxs = get_area_extent(args.area_extent) if args.area_extent else None

    # Get access to EUMETSAT Data Store
    with open(args.credentials) as handle:
        token = eumdac.AccessToken(handle.read().split(','))
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI')

    if not args.files:
        # Range of dates to search for (dtstart, dtend)
        # Use 1 month at a time otherwise it collapses
        dt_pairs = [
            (t1.to_pydatetime(), t2.to_pydatetime())
            for t1, t2 in zip(
                pd.date_range(args.start, args.end, freq='1MS', inclusive='left'),
                pd.date_range(args.start, args.end, freq='1MS', inclusive='right')
            )
        ]

        if (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days < 28:
            dt_pairs = [(pd.Timestamp(args.start).to_pydatetime(), pd.Timestamp(args.end).to_pydatetime())]

        expected_nat_files = [
            f
            for dt in dt_pairs
            for f in list(selected_collection.search(dtstart=dt[0], dtend=dt[1]))
        ]
    else:
        with open(args.files) as handle:
            files = handle.read().splitlines()

        expected_nat_files = [
            _f
            for f in tqdm.tqdm(files, ncols=80, desc="Searching for files")
            for _f in list(selected_collection.search(title=f[:-4]))
        ]

    print(f"Found {len(expected_nat_files)}")

    output_file = Path(args.output)

    if output_file.exists():
        ds = xr.open_zarr(output_file)
        existing_nat_files = ds.nat_file.values
        expected_nat_files = [
            f
            for f in expected_nat_files
            if f'{str(f)}.nat' not in existing_nat_files
        ]
        first = False
    else:
        first = True

    print(f"Will download {len(expected_nat_files)}")

    lock = Lock()
    if first:
        first_file = expected_nat_files.pop(0)
        download_file(first_file, lock, output_file, first=first, area_extent=area_extent_idxs)

    # If many workers are used EUMDAC complains and blocks
    workers = args.workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_file, f, lock, output_file, area_extent=area_extent_idxs) for f in expected_nat_files]

        with tqdm.tqdm(total=len(futures), ncols=80) as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update()