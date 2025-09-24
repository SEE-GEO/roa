"""
Script to download MTG-FCI data into a Zarr file,
one month resolution.
"""

import argparse
import concurrent.futures
from pathlib import Path
import shutil
import tempfile
from threading import Lock
import time
import zipfile

import eumdac
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from roa.data import FCI2SEVIRI

def download_file(product, lock, output_file, first=False, area_extent=None, tmpdir=None, max_download_attempts=3, max_retries=3):
    fci_file_zip = f'{str(product)}.zip'
    for _ in range(max_retries):
        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
            file_path = Path(tmpdirname) / fci_file_zip
            for _ in range(max_download_attempts):
                try:
                    with product.open() as fsrc, \
                        open(file_path, mode='wb') as handle:
                        shutil.copyfileobj(fsrc, handle)
                        break
                except:
                    time.sleep(60)
                    continue
            try:
                with zipfile.ZipFile(file_path) as zf:
                    zf.extractall(tmpdirname)
            except:
                continue
            # Remove the zip file after extraction to free up space
            file_path.unlink(missing_ok=True)

            # Find all FCI files in the temporary directory
            # and read them
            fci_files = list(Path(tmpdirname).glob('*BODY*.nc'))
            try:
                if area_extent:
                    ds = FCI2SEVIRI(fci_files).get_dataset(area_extent=area_extent)
                else:
                    ds = FCI2SEVIRI(fci_files).get_dataset()
                break
            except:
                continue
        # If we reached this point, we have the data we need
        break
    else:
        print(f"Failed to download and read {fci_file_zip} after {max_retries} retries")
        return

    ds = ds.expand_dims({"file_id": [str(product)]})

    # Clear attributes that are not serializable
    for v in ds:
        ds[v].attrs = {}

    with lock:
        if first:
            ds.to_zarr(output_file)
        else:
            ds.to_zarr(output_file, append_dim="file_id")

def get_area_extent(area_extent):
    return {
        'y': np.arange(area_extent[1], area_extent[3]),
        'x': np.arange(area_extent[0], area_extent[2])
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MSG data")
    parser.add_argument("--start", type=str, default="2024-09-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-07-01", help="End date")
    parser.add_argument("--credentials", type=str,
                        default=Path("~/.eumdac/credentials").expanduser(),
                        help="EUMDAC credentials file")
    parser.add_argument("--output", type=str, help="Output Zarr", required=True)
    parser.add_argument("--workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--area_extent", nargs=4, type=int)
    parser.add_argument("--files", type=str,
                        help="File with list of files to download, one .zip file per line")
    parser.add_argument("--tmpdir", type=str, default=None,
                        help="Temporary directory to use for downloads")
    args = parser.parse_args()

    # Fix area extent
    area_extent_idxs = get_area_extent(args.area_extent) if args.area_extent else None

    # Get access to EUMETSAT Data Store
    with open(args.credentials) as handle:
        token = eumdac.AccessToken(handle.read().split(','))
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection('EO:EUM:DAT:0662')

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

        expected_files = [
            f
            for dt in dt_pairs
            for f in list(selected_collection.search(dtstart=dt[0], dtend=dt[1]))
        ]
    else:
        with open(args.files) as handle:
            files = handle.read().splitlines()

        expected_files = [
            _f
            for f in tqdm.tqdm(files, ncols=80, desc="Searching for files")
            for _f in list(selected_collection.search(title=f))
        ]

    # Filter out files without the expected content in the Data Store
    expected_files = [
        e
        for e in expected_files
        if (len([f for f in e.entries if (('BODY' in f) and f.endswith('.nc'))]) == 40)
    ]

    print(f"Found {len(expected_files)} files")

    output_file = Path(args.output)

    if output_file.exists():
        ds = xr.open_zarr(output_file)
        existing_files = ds.file_id.values
        expected_files = [
            f
            for f in expected_files
            if str(f) not in existing_files
        ]
        first = False
    else:
        first = True

    print(f"Will download {len(expected_files)} files")

    lock = Lock()
    if first:
        first_file = expected_files.pop(0)
        download_file(first_file, lock, output_file, first=first, area_extent=area_extent_idxs, tmpdir=args.tmpdir)

    # If many workers are used EUMDAC complains and blocks
    workers = args.workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_file, f, lock, output_file, area_extent=area_extent_idxs, tmpdir=args.tmpdir) for f in expected_files]

        with tqdm.tqdm(total=len(futures), ncols=80) as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update()