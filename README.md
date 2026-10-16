# Project README
Here is the codebase associated with the paper titled *A Lightweight Probabilistic Sequence Model for Efficient Nonlinear LED Channel Equalization*. 

Lab website: [UW Photonics Lab](https://sites.google.com/uw.edu/photonics-lab)

First Author/codebase maintainer ddj123[at]uw[dot]edu

# Setup
Create the conda environment (Python 3.11). This installs torch, the [PyFlux](https://github.com/UW-Photonics-Lab/PyFlux) hardware/experiment package, and this repo itself as an editable install, so `import modules` and `import pyflux` resolve from any working directory:

```
conda env create -f environment.yml
conda activate prob_tcn_for_LED
```

The figure scripts default to CPU (`SUMMARIZE_DEVICE=cpu`); no GPU is required to regenerate figures.

# Critical:
In order to generate the figures associated with this paper and inspect the models/datasets, you have to get the associated files from our Zenodo DOI for this project. Download the latest prob_tcn_for_led_data.tar.gz and extract in repo root. This creates the following `data/` tree (one directory per DC offset stage):

```
data/
  sweeps/   collected OFDM datasets, one .zarr per DC offset
    prime_coast_dc0.05A_..._20260724_2101.zarr   (50 mA)
    fair_ledge_dc0.06A_..._20260726_1115.zarr    (60 mA)
    calm_heath_dc0.08A_..._20260729_1339.zarr    (80 mA)
    mild_star_dc0.12A_..._20260802_2119.zarr     (120 mA)

  experiments/train_and_validate/
    nice_rain_channel_models_20260915_2032/      50 mA, channel-model grid search
    nice_rain_encoder_decoder_20260916_0654/     50 mA, encoder/decoder grid search
    nice_rain_ed_validation_20260917_1929/       50 mA, E/D live-channel validation
    tidy_vale_channel_models_20260921_1015/      60 mA
    tidy_vale_encoder_decoder_20260921_2135/     60 mA
    tidy_vale_ed_validation_20260923_1322/       60 mA
    rough_peak_channel_models_20260928_1103/     80 mA
    rough_peak_encoder_decoder_20260928_2300/    80 mA
    rough_peak_ed_validation_20260930_1701/      80 mA
    firm_fern_channel_models_20261005_1518/      120 mA
    firm_fern_encoder_decoder_20261006_0352/     120 mA
    firm_fern_ed_validation_20261008_0006/       120 mA

    calm_rain_find_regularization_20260817_1952/ regularization sweep
      stats.csv    per-cell paired stats (drives the plots)
      pairs.csv    per-seed EVM pairs
    raw_storm_channel_models_20260808_1526/      channel models used by the regularization sweep (50 mA)
    tiny_cliff_channel_models_20260810_1622/     60 mA
    fleet_sand_channel_models_20260812_1903/     80 mA
    light_sea_channel_models_20260813_2059/      120 mA

  logs/   experiment run logs
```

Then, you will be able to generate the manuscript's key figures by running

```
python experiments/summarize_results.py
```

```
python experiments/plot_find_regularization.py <path_to_repo>/prob_tcn_for_LED/data/experiments/train_and_validate/calm_rain_find_regularization_20260817_1952/stats.csv
```

All models and data used for these experiments can be found in the associated Zenodo files and logs included. 

Here's how the zenodo tarball was generated:

```

tar -czf ../prob_tcn_for_led_data.tar.gz \
  data/experiments/train_and_validate/nice_rain_channel_models_20260915_2032 \
  data/experiments/train_and_validate/nice_rain_encoder_decoder_20260916_0654 \
  data/experiments/train_and_validate/nice_rain_ed_validation_20260917_1929 \
  data/sweeps/prime_coast_dc0.05A_fmin1e+06_fmax7.6e+06_20260724_2101.zarr \
  data/experiments/train_and_validate/tidy_vale_channel_models_20260921_1015 \
  data/experiments/train_and_validate/tidy_vale_encoder_decoder_20260921_2135 \
  data/experiments/train_and_validate/tidy_vale_ed_validation_20260923_1322 \
  data/sweeps/fair_ledge_dc0.06A_fmin1e+06_fmax9.2e+06_20260726_1115.zarr \
  data/experiments/train_and_validate/rough_peak_channel_models_20260928_1103 \
  data/experiments/train_and_validate/rough_peak_encoder_decoder_20260928_2300 \
  data/experiments/train_and_validate/rough_peak_ed_validation_20260930_1701 \
  data/sweeps/calm_heath_dc0.08A_fmin1e+06_fmax1.08e+07_20260729_1339.zarr \
  data/experiments/train_and_validate/firm_fern_channel_models_20261005_1518 \
  data/experiments/train_and_validate/firm_fern_encoder_decoder_20261006_0352 \
  data/experiments/train_and_validate/firm_fern_ed_validation_20261008_0006 \
  data/sweeps/mild_star_dc0.12A_fmin1e+06_fmax1.3e+07_20260802_2119.zarr \
  data/experiments/train_and_validate/calm_rain_find_regularization_20260817_1952/stats.csv \
  data/experiments/train_and_validate/calm_rain_find_regularization_20260817_1952/pairs.csv \
  data/experiments/train_and_validate/raw_storm_channel_models_20260808_1526 \
  data/experiments/train_and_validate/tiny_cliff_channel_models_20260810_1622 \
  data/experiments/train_and_validate/fleet_sand_channel_models_20260812_1903 \
  data/experiments/train_and_validate/light_sea_channel_models_20260813_2059 \
  data/logs
```
and to unpack, go to your project repo's root dir and run:

```
tar -xzf prob_tcn_for_led_data.tar.gz
```
