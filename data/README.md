# Data Files

This directory contains the data files needed to run the Climate QA Agent.

## Required Files

### RAG Corpus
- **File**: `climate_corpus.jsonl`
- **Description**: JSONL file containing climate documentation chunks for RAG retrieval
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1LvFAJB50VCwAsRfKTxWLxqBHiMUjXS_B/view?usp=sharing)

### NetCDF Climate Data (ERA5)
Download the following files from [Google Drive](https://drive.google.com/drive/folders/12xHyBDXGHiQIMAMn-F0AY50X2qpx3rWu?usp=sharing):

| File | Variable | Description |
|------|----------|-------------|
| `t2m.nc` | t2m | 2m air temperature (daily mean) |
| `d2m.nc` | d2m | 2m dewpoint temperature (daily mean) |
| `u10.nc` | u10 | 10m U wind component |
| `v10.nc` | v10 | 10m V wind component |
| `msl.nc` | msl | Mean sea level pressure |
| `tp.nc` | tp | Total precipitation |

## Directory Structure

After downloading, your `data/` directory should look like:

```
data/
├── README.md
├── climate_corpus.jsonl
├── t2m.nc
├── d2m.nc
├── u10.nc
├── v10.nc
├── msl.nc
└── tp.nc
```

## Data Source

The NetCDF files contain ERA5 reanalysis data from the Copernicus Climate Data Store:
- **Spatial coverage**: NYC bounding box (approx. 40.5°N - 40.9°N, 74.1°W - 73.7°W)
- **Temporal coverage**: January 2026 (daily records)
- **Resolution**: Daily statistics

## Notes

- The `.nc` files are excluded from git (see `.gitignore`) due to their size
- Make sure to set the `DATA_DIR` environment variable if your data is in a different location
