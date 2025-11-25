import argparse
from pathlib import Path

from dask.diagnostics import ProgressBar
import xarray as xr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', type=str)
    args = parser.parse_args()

    files = [Path(f) for f in args.files]
    assert all(f.exists() for f in files), "One or more input files do not exist."

    storage_options = dict(profile='roa')

    for file in files:
        year = file.name.split('_')[-1].split('.')[0]
        print(f"Uploading file: {file} (year: {year})")
        ds = xr.open_zarr(file)
        
        ds.time.encoding['chunks'] = (ds.time.size,)

        write_job = ds.to_zarr(
            f's3://rainoverafrica/data/roa_{year}.zarr',
            storage_options=storage_options,
            compute=False
        )
        with ProgressBar():
            write_job.compute()

    print("Completed script")
