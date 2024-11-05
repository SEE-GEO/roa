# Rain over Africa

This README is to be completed, it contains minimal information.

The code for running retrievals is likely final, pending the RoA paper review. We don’t plan to make backwards-incompatible updates, but additional code for analyses can be updated or added.

Requirements:
- Linux
- Python 3.10

Installation of RoA:
```shell
$ pip install git+https://github.com/SEE-GEO/roa
```

You can also clone the repository and install locally. In any case, you need the file [`data/network_CPU.pt`](data/network_CPU.pt) from this repository.

Note that many unused dependencies are installed as they are required by other dependencies.

For an illustrative example of complete retrievals, check [`docs/example.ipynb`](docs/example.ipynb).