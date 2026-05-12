# Interactive FTIR Workstation

A single-file Flask web app for loading real FTIR CSV data, extracting editable absorbance peaks, regenerating a synthetic spectrum, and exporting the regenerated data.

## Features

- Upload raw FTIR CSV files with `cm-1` / `%T` columns or common aliases such as `Wavenumber` / `Transmittance`.
- Upload peak-model CSV files with `Position`, `Amplitude`, `FWHM`, and optional `Baseline` columns.
- Display the original spectrum with extracted peak markers.
- Edit, add, and remove peak parameters in the browser.
- Regenerate a synthetic spectrum against the uploaded spectrum's original wavenumber axis.
- Download regenerated spectrum CSV, edited peak-model CSV, and a PNG overlay when Kaleido is available.
- Safer download routing that serves only generated files from the app output directory.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open <http://127.0.0.1:5050>.

## CSV formats

### Raw spectrum CSV

```csv
cm-1,%T
4000,98.4
3999,98.3
...
```

### Peak-model CSV

```csv
Position,Amplitude,FWHM,Baseline
1715,0.18,35,98.5
2920,0.06,42,98.5
```
