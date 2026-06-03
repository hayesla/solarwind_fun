# swpipeline

This is the data layer. It takes raw spacecraft CDF files and turns them into
clean, hourly, physics enriched solar wind datasets ready for analysis.

Every spacecraft goes through the same four stages, and each stage lives in its
own file:

```
download  ->  read  ->  clean  ->  enrich  ->  save
download.py   io.py   clean.py   enrich.py    io.py
```

- **config.py** is the one place that holds the dataset IDs, raw and standard
  column names, quality cuts, and SPICE kernel URLs. Every other file reads from
  it, so this is where you go to change a threshold or add a dataset.
- **download.py** fetches the raw CDFs and caches them on disk. PSP, SolO and
  ACE come from CDAWeb through Fido; Helios comes straight from SPDF. Files
  already downloaded are never fetched again.
- **io.py** reads those CDFs into a single time indexed DataFrame, expands the
  field vectors into components, and turns fill values into NaN. It also saves
  and loads the parquet files, stamping the version, dataset IDs and kernel name
  into the metadata so every file describes how it was built.
- **clean.py** applies the quality cuts per instrument, renames everything to
  the standard columns and units, and holds the helpers to resample to hourly
  and merge field with plasma.
- **enrich.py** adds spacecraft position (SPICE for PSP and SolO, fixed L1 for
  ACE, already in the file for Helios) and the derived physics: plasma beta,
  Alfven speed and Alfven Mach number.

`pipeline.py` wires these together. Use `build_dataset` for PSP, SolO or ACE,
and `build_helios_dataset` for Helios (its files already merge field, plasma and
position, so there is no merge or SPICE step). Both write a hourly and an
enriched parquet into `data_processed/` and return the enriched DataFrame.

Because the stages do not reach across each other, you can also call any one on
its own when you want to see the data between steps. The orchestrators just
chain these calls in order, so reading one is the quickest way to follow the
full path for a spacecraft.
