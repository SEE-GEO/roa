import argparse
from pathlib import Path

from dask.diagnostics import ProgressBar
import zarr
import xarray as xr

def combine_parts(file):
    file = Path(file)
    year = file.name
    expected_files = [f'roa_{year}_{i}.zarr' for i in range(1, 13)]
    
    assert all([(file / f).exists() for f in expected_files]), "One or more expected part files are missing."
    
    datasets = [
        xr.open_zarr(str(file / f)) for f in expected_files
    ]

    first_ds = datasets[0]
    static_coords = {
        'latitude': first_ds.latitude,
        'longitude': first_ds.longitude,
        'x': first_ds.x,
        'y': first_ds.y
    }

    combined = (
        xr.concat(
            [
                d.drop_vars(['latitude', 'longitude'])
                for d in datasets
            ],
            dim='time'
        )
        .sortby('time')
    )

    combined = combined.assign_coords(static_coords).reset_coords(['latitude', 'longitude'])

    return combined


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    files = [Path(f) for f in args.files]
    assert all(f.exists() for f in files), "One or more input files do not exist."

    storage_options = dict(profile='roa')

    for file in files:
        year = file.name.split('_')[-1].split('.')[0]
        print(f"Uploading file: {file} (year: {year})")
        if file.suffix == '.zip':
            store = zarr.ZipStore(str(file), mode='r')
            ds = xr.open_zarr(store)
        elif file.suffix == '.zarr':
            ds = xr.open_zarr(str(file))
        else:
            ds = combine_parts(file)
        
        ds.time.encoding['chunks'] = (ds.time.size,)

        write_job = ds.to_zarr(
            f's3://rainoverafrica/data/roa_{year}.zarr',
            mode="w" if args.overwrite else "w-",
            storage_options=storage_options,
            compute=False
        )
        with ProgressBar():
            write_job.compute()

        if file.suffix == '.zip':
            store.close()

    print("Completed script")
