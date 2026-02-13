# STARSS2023 Acoustic Images

To install:

```
poetry install
```

To generate acoustic images:

```
poetry run python starss_representations/generate_acoustic_image_dataset.py
```

To standardise acoustic images:

```
poetry run python starss_representations/standardise_dataset.py
```

## Data

Ensure that:
- 32-channel Eigenmike data is downloaded to `data/eigen_dev/dev-<split>-<location`, e.g. `data/eigen_dev/dev-train-tau`
- Metadata is extracted to `data/metadata_dev`

Acoustic images will be saved as `outputs/apgd_dev` and `outputs/apgd_dev_std` (after standardisation).
