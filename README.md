# Project README
Here is the codebase associated with the paper titled *A Lightweight Probabilistic Sequence Model for Efficient Nonlinear LED Channel Equalization*. 

Lab website: [UW Photonics Lab](https://sites.google.com/uw.edu/photonics-lab)

First Author/codebase maintainer ddj123[at]uw[dot]edu

# Critical:
In order to generate the figures associated with this paper and inspect the models/datasets, you have to get the associated files from our Zenodo DOI for this project. Download the latest project_data.zip and unzip in the project's root directory. This will create two folders with the following structure:
```
 data/
    channel_measurements/
        zarr files
        cached data checkpoints
    plots/
        plotting cached pkl files
        svg figures
    validation_measurements/
        zarr files
        cached data checkpoints
 models/
    channel_models/
        model-name/
            config.json
            model.pth
    encoder_decoders/
        model-name/
            config.json
            model.pth
```
Because .zarr files are enormous, the original zarr files were converted to cached data checkpoints of the data tensors. The code will check to see if checkpoints exist and open them first. 


