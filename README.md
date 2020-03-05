# easyPEASI
> Software implementing methods from: David Nahmias and Kimberly Kontson. Easy Perturbation EEG Algorithm for Spectral Importance (easyPEASI):
A simple method to identifyimportant spectral features of EEG in deep learning models. In Review. (2020)

This work identifies frequency bands that are important in EEG-driven deep learning classifications.

## Usage

To train and save deep learning models for all classifications
```sh
.\deepShell.sh
```
with files in _data_ subfolder.

Once models are generated,
```sh
python testModels.py
```

## Development setup

Developed and tested on Python 3.5, packages used found in software header.
Deep learning implemented using _PyTorch_ and _Braindecode_ package.
Make file coming soon.


## Release History

* 0.0.1
    * Initial commit of stable project

## Meta

David Nahmias – [Website](dnahmias.com) – david.nahmias@fda.hhs.gov

Distributed under the public domain license. See ``LICENSE`` for more information.

[https://github.com/dbp-osel](https://github.com/dbp-osel/)


## Citation
Summited to conference, full citation forthcoming. 

David Nahmias and Kimberly Kontson. Easy Perturbation EEG Algorithm for Spectral Importance (easyPEASI):
A simple method to identifyimportant spectral features of EEG in deep learning models. In Review (2020)


