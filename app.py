"""Interactive FTIR workstation.

Run with:
    python app.py

The app accepts CSV spectra containing wavenumber and transmittance columns,
extracts editable peak parameters, regenerates a synthetic spectrum, and exports
both the regenerated spectrum and plot image.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, jsonify, render_template_string, request, send_from_directory, url_for
from scipy.signal import find_peaks, peak_widths, savgol_filter
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

SPECTRA: dict[str, dict[str, object]] = {}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Interactive FTIR Workstation</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {
            color-scheme: light;
            --bg: #eef2f7;
            --panel: #ffffff;
            --ink: #111827;
            --muted: #6b7280;
            --line: #d1d5db;
            --primary: #111827;
            --primary-soft: #e5e7eb;
            --danger: #b91c1c;
            --ok: #047857;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            padding: 28px;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: var(--bg);
            color: var(--ink);
        }

        .container {
            max-width: 1480px;
            margin: 0 auto;
            background: var(--panel);
            border-radius: 18px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
            overflow: hidden;
        }

        header {
            padding: 28px 32px;
            border-bottom: 1px solid var(--line);
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        }

        h1 { margin: 0 0 8px; font-size: clamp(1.8rem, 3vw, 2.7rem); }
        h2 { margin: 0 0 12px; font-size: 1.25rem; }
        p { color: var(--muted); line-height: 1.5; }

        main { padding: 28px 32px 34px; }

        .grid {
            display: grid;
            grid-template-columns: minmax(330px, 420px) 1fr;
            gap: 24px;
            align-items: start;
        }

        .card {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 18px;
            background: #fff;
        }

        .upload-box {
            display: grid;
            gap: 12px;
        }

        input[type="file"], input[type="number"] {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 10px;
            background: #fff;
        }

        input[type="number"] { text-align: right; }

        button, .download-link {
            appearance: none;
            border: 0;
            border-radius: 10px;
            padding: 10px 14px;
            font-weight: 700;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        button.primary { background: var(--primary); color: #fff; }
        button.secondary, .download-link { background: var(--primary-soft); color: var(--ink); }
        button.danger { background: #fee2e2; color: var(--danger); }
        button:disabled, .disabled { opacity: 0.48; pointer-events: none; }

        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }

        .status {
            display: none;
            margin-top: 12px;
            padding: 10px 12px;
            border-radius: 10px;
            background: #eff6ff;
            color: #1d4ed8;
        }

        .status.error { background: #fee2e2; color: var(--danger); }
        .status.ok { background: #dcfce7; color: var(--ok); }
        .status.show { display: block; }

        .table-wrap {
            max-height: 430px;
            overflow: auto;
            border: 1px solid var(--line);
            border-radius: 12px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 540px;
        }

        th, td {
            border-bottom: 1px solid var(--line);
            padding: 9px;
            text-align: center;
        }

        th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #f9fafb;
            font-size: 0.9rem;
        }

        td:last-child { width: 88px; }

        .graph-card { min-width: 0; }
        #graph { width: 100%; min-height: 660px; }
        .empty-state { padding: 48px 20px; text-align: center; border: 2px dashed var(--line); border-radius: 16px; }
        .meta { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0; color: var(--muted); }
        .pill { border: 1px solid var(--line); border-radius: 999px; padding: 5px 10px; background: #f9fafb; }

        @media (max-width: 980px) {
            body { padding: 14px; }
            .grid { grid-template-columns: 1fr; }
            main, header { padding: 20px; }
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Interactive FTIR Workstation</h1>
        <p>Upload a real FTIR CSV, inspect extracted absorbance peaks, edit the model, regenerate a synthetic spectrum, and download clean outputs.</p>
    </header>
    <main>
        <div class="grid">
            <section class="card">
                <h2>1. Load spectrum</h2>
                <form method="POST" enctype="multipart/form-data" class="upload-box">
                    <input type="file" name="file" accept=".csv,text/csv" required>
                    <button class="primary" type="submit">Load CSV</button>
                </form>
                {% if error %}<div class="status error show">{{ error }}</div>{% endif %}

                {% if loaded %}
                <div class="meta">
                    <span class="pill">{{ point_count }} points</span>
                    <span class="pill">{{ peaks|length }} peaks</span>
                    <span class="pill">Baseline {{ '%.2f'|format(baseline) }} %T</span>
                </div>

                <h2>2. Edit peaks</h2>
                <p>Position is in cm⁻¹, amplitude is absorbance above baseline, and FWHM is in cm⁻¹.</p>
                <div class="table-wrap">
                    <table id="peakTable">
                        <thead>
                            <tr><th>Position</th><th>Amplitude</th><th>FWHM</th><th></th></tr>
                        </thead>
                        <tbody>
                            {% for peak in peaks %}
                            <tr>
                                <td><input type="number" step="0.01" value="{{ '%.2f'|format(peak.position) }}"></td>
                                <td><input type="number" step="0.0001" value="{{ '%.4f'|format(peak.amplitude) }}"></td>
                                <td><input type="number" step="0.01" min="0.01" value="{{ '%.2f'|format(peak.fwhm) }}"></td>
                                <td><button type="button" class="danger" onclick="removeRow(this)">Remove</button></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>

                <div class="actions">
                    <button type="button" class="primary" onclick="regenerateSpectrum()">Regenerate</button>
                    <button type="button" class="secondary" onclick="addPeak()">Add peak</button>
                    <a id="csvLink" class="download-link disabled" href="#">Spectrum CSV</a>
                    <a id="peakCsvLink" class="download-link disabled" href="#">Peak CSV</a>
                    <a id="pngLink" class="download-link disabled" href="#">PNG</a>
                </div>
                <div id="status" class="status"></div>
                {% else %}
                <div class="empty-state">
                    <h2>No spectrum loaded</h2>
                    <p>Expected columns include <strong>cm-1</strong> (or Wavenumber) and <strong>%T</strong> (or Transmittance). Peak-model CSVs with Position, Amplitude, FWHM, and optional Baseline are also supported.</p>
                </div>
                {% endif %}
            </section>

            <section class="card graph-card">
                <h2>3. Inspect overlay</h2>
                <div id="graph"></div>
            </section>
        </div>
    </main>
</div>

<script>
const spectrumId = {{ spectrum_id|tojson }};
const initialGraph = {{ graph_json|safe }};

if (initialGraph) {
    Plotly.newPlot("graph", initialGraph.data, initialGraph.layout, {responsive: true});
}

function setStatus(message, type = "ok") {
    const el = document.getElementById("status");
    if (!el) return;
    el.textContent = message;
    el.className = `status ${type} show`;
}

function collectPeaks() {
    const rows = document.querySelectorAll("#peakTable tbody tr");
    return Array.from(rows).map((row, index) => {
        const cells = row.querySelectorAll("input");
        const peak = {
            position: Number.parseFloat(cells[0].value),
            amplitude: Number.parseFloat(cells[1].value),
            fwhm: Number.parseFloat(cells[2].value)
        };
        if (!Number.isFinite(peak.position) || !Number.isFinite(peak.amplitude) || !Number.isFinite(peak.fwhm) || peak.fwhm <= 0) {
            throw new Error(`Peak row ${index + 1} has invalid numeric values.`);
        }
        return peak;
    });
}

function updateDownload(id, href) {
    const link = document.getElementById(id);
    if (!link) return;
    if (href) {
        link.href = href;
        link.classList.remove("disabled");
    } else {
        link.href = "#";
        link.classList.add("disabled");
    }
}

async function regenerateSpectrum() {
    if (!spectrumId) return;
    let peaks;
    try {
        peaks = collectPeaks();
    } catch (error) {
        setStatus(error.message, "error");
        return;
    }

    setStatus("Regenerating spectrum…", "ok");
    const response = await fetch("/regenerate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({spectrum_id: spectrumId, peaks})
    });
    const data = await response.json();

    if (!response.ok) {
        setStatus(data.error || "Regeneration failed.", "error");
        return;
    }

    Plotly.react("graph", data.graph.data, data.graph.layout, {responsive: true});
    updateDownload("csvLink", data.csv);
    updateDownload("peakCsvLink", data.peak_csv);
    updateDownload("pngLink", data.png);
    setStatus(data.message || "Regenerated successfully.", data.png ? "ok" : "error");
}

function addPeak() {
    const tbody = document.querySelector("#peakTable tbody");
    const row = document.createElement("tr");
    row.innerHTML = `
        <td><input type="number" step="0.01" value="1500.00"></td>
        <td><input type="number" step="0.0001" value="0.0500"></td>
        <td><input type="number" step="0.01" min="0.01" value="40.00"></td>
        <td><button type="button" class="danger" onclick="removeRow(this)">Remove</button></td>`;
    tbody.appendChild(row);
}

function removeRow(button) {
    button.closest("tr").remove();
}
</script>
</body>
</html>
"""

COLUMN_ALIASES = {
    "wavenumber": "cm-1",
    "wavenumbercm1": "cm-1",
    "wavenumbercm-1": "cm-1",
    "cm-1": "cm-1",
    "cm1": "cm-1",
    "x": "cm-1",
    "%t": "%T",
    "t": "%T",
    "transmittance": "%T",
    "percenttransmittance": "%T",
    "transmission": "%T",
    "position": "Position",
    "peakposition": "Position",
    "amplitude": "Amplitude",
    "height": "Amplitude",
    "fwhm": "FWHM",
    "width": "FWHM",
    "baseline": "Baseline",
}


def normalize_column_name(name: object) -> str:
    text = (
        str(name)
        .strip()
        .replace("⁻¹", "-1")
        .replace("⁻1", "-1")
        .replace("−", "-")
    )
    key = re.sub(r"[\s_%()\[\]/^]+", "", text).lower()
    return COLUMN_ALIASES.get(key, text)


def read_csv_flexibly(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for skiprows in (0, 1, 2, 3, 4, 5):
        try:
            frame = pd.read_csv(path, skiprows=skiprows)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            errors.append(str(exc))
            continue
        frame.columns = [normalize_column_name(c) for c in frame.columns]
        if {"cm-1", "%T"}.issubset(frame.columns) or "Position" in frame.columns:
            return frame
    raise ValueError("Could not find supported FTIR columns. Use cm-1/%T or Position/Amplitude/FWHM.")


def clean_numeric_series(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.dropna(subset=list(columns))


def as_python_float_array(values: np.ndarray) -> list[float]:
    return [float(v) for v in values]


def smooth_signal(values: np.ndarray, preferred_window: int, polyorder: int) -> np.ndarray:
    length = len(values)
    if length <= polyorder + 2:
        return values
    window = min(preferred_window, length if length % 2 else length - 1)
    window = max(window, polyorder + 3)
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 else length - 1
    return savgol_filter(values, window, polyorder)


def transmittance_to_absorbance(transmittance: np.ndarray) -> np.ndarray:
    return 2 - np.log10(np.clip(transmittance, 1e-5, 100))


def extract_peaks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    frame = clean_numeric_series(df, ["cm-1", "%T"]).sort_values("cm-1", ascending=False)
    if len(frame) < 5:
        raise ValueError("At least five spectrum points are required for peak extraction.")

    x_real = frame["cm-1"].to_numpy(dtype=float)
    y_real = frame["%T"].to_numpy(dtype=float)
    target_baseline = float(np.percentile(y_real, 98))
    y_abs = transmittance_to_absorbance(y_real)
    bg_abs = float(transmittance_to_absorbance(np.array([target_baseline]))[0])
    y_abs_search = smooth_signal(y_abs, 15, 2)

    absorbance_range = float(np.nanmax(y_abs_search) - np.nanmin(y_abs_search))
    prominence = max(0.002, absorbance_range * 0.04)
    distance = max(3, len(frame) // 180)
    peak_indices, _ = find_peaks(y_abs_search, prominence=prominence, distance=distance)

    if len(peak_indices) == 0:
        return np.array([]), np.array([]), np.array([]), target_baseline

    widths_raw, _, _, _ = peak_widths(y_abs_search, peak_indices, rel_height=0.5)
    resolution = float(np.median(np.abs(np.diff(np.sort(x_real)))))
    positions: list[float] = []
    amplitudes: list[float] = []
    widths: list[float] = []

    for raw_width, idx in zip(widths_raw, peak_indices):
        amp = max(float(y_abs[idx] - bg_abs), 0.0)
        positions.append(float(x_real[idx]))
        amplitudes.append(amp)
        widths.append(max(float(raw_width * resolution * (1 + 0.30 * amp)), resolution * 2, 1.0))

    order = np.argsort(np.array(positions))[::-1]
    return np.array(positions)[order], np.array(amplitudes)[order], np.array(widths)[order], target_baseline


def load_input(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    df = read_csv_flexibly(path)

    if {"cm-1", "%T"}.issubset(df.columns):
        frame = clean_numeric_series(df, ["cm-1", "%T"]).sort_values("cm-1", ascending=False)
        if frame.empty:
            raise ValueError("The spectrum CSV does not contain numeric cm-1/%T rows.")
        positions, amplitudes, widths, baseline = extract_peaks(frame)
        return (
            frame["cm-1"].to_numpy(dtype=float),
            frame["%T"].to_numpy(dtype=float),
            positions,
            amplitudes,
            widths,
            baseline,
        )

    if {"Position", "Amplitude", "FWHM"}.issubset(df.columns):
        frame = clean_numeric_series(df, ["Position", "Amplitude", "FWHM"])
        if frame.empty:
            raise ValueError("The peak CSV does not contain numeric peak rows.")
        baseline = float(pd.to_numeric(df.get("Baseline", pd.Series([98.5])), errors="coerce").dropna().iloc[0])
        positions = frame["Position"].to_numpy(dtype=float)
        amplitudes = frame["Amplitude"].to_numpy(dtype=float)
        widths = frame["FWHM"].to_numpy(dtype=float)
        x_real, y_real = generate_spectrum(positions, amplitudes, widths, baseline)
        return x_real, y_real, positions, amplitudes, widths, baseline

    raise ValueError("Invalid CSV format. Expected cm-1/%T or Position/Amplitude/FWHM columns.")


def generate_spectrum(
    positions: np.ndarray,
    amplitudes: np.ndarray,
    widths: np.ndarray,
    baseline: float = 98.5,
    x_axis: np.ndarray | None = None,
    noise_level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    wn_range = np.array(x_axis, dtype=float) if x_axis is not None else np.linspace(4000, 600, 3500)
    bg_abs = float(transmittance_to_absorbance(np.array([baseline]))[0])
    absorbance = np.full_like(wn_range, bg_abs, dtype=float)

    for pos, amp, fwhm in zip(positions, amplitudes, widths):
        safe_width = max(float(fwhm), 1e-6)
        exponent = -4 * np.log(2) * ((wn_range - float(pos)) ** 2) / (safe_width**2)
        absorbance += max(float(amp), 0.0) * np.exp(exponent)

    transmittance = 100 * (10 ** (-absorbance))
    if noise_level > 0:
        rng = np.random.default_rng(42)
        transmittance += rng.normal(0, noise_level, len(wn_range))
    transmittance = smooth_signal(transmittance, 11, 2)
    return wn_range, np.round(np.clip(transmittance, 0, 100), 4)


def create_plot(
    original_x: np.ndarray,
    original_y: np.ndarray,
    positions: np.ndarray,
    synthetic_x: np.ndarray | None = None,
    synthetic_y: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_x, y=original_y, mode="lines", name="Original spectrum", line={"color": "#111827", "width": 2}))

    if synthetic_x is not None and synthetic_y is not None:
        fig.add_trace(go.Scatter(x=synthetic_x, y=synthetic_y, mode="lines", name="Regenerated spectrum", line={"color": "#dc2626", "width": 2}))

    if len(positions):
        sort_order = np.argsort(original_x)
        peak_y = np.interp(positions, original_x[sort_order], original_y[sort_order])
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=peak_y,
                mode="markers",
                marker={"size": 9, "color": "#2563eb", "line": {"width": 1, "color": "#ffffff"}},
                name="Editable peaks",
            )
        )

    fig.update_layout(
        title="FTIR Spectrum",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="% Transmittance",
        template="plotly_white",
        height=660,
        margin={"l": 70, "r": 30, "t": 70, "b": 70},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(autorange="reversed")
    return fig


def peak_records(positions: np.ndarray, amplitudes: np.ndarray, widths: np.ndarray) -> list[dict[str, float]]:
    return [
        {"position": float(position), "amplitude": float(amplitude), "fwhm": float(fwhm)}
        for position, amplitude, fwhm in zip(positions, amplitudes, widths)
    ]


def parse_peak_payload(payload: object) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    spectrum_id = str(payload.get("spectrum_id", ""))
    peaks = payload.get("peaks")
    if spectrum_id not in SPECTRA:
        raise ValueError("Upload a spectrum before regenerating.")
    if not isinstance(peaks, list):
        raise ValueError("Peaks must be a list.")

    positions: list[float] = []
    amplitudes: list[float] = []
    widths: list[float] = []
    for index, peak in enumerate(peaks, start=1):
        if not isinstance(peak, dict):
            raise ValueError(f"Peak row {index} is not an object.")
        position = float(peak["position"])
        amplitude = float(peak["amplitude"])
        fwhm = float(peak["fwhm"])
        if not all(np.isfinite([position, amplitude, fwhm])) or fwhm <= 0:
            raise ValueError(f"Peak row {index} has invalid values.")
        positions.append(position)
        amplitudes.append(amplitude)
        widths.append(fwhm)

    return spectrum_id, np.array(positions), np.array(amplitudes), np.array(widths)


@app.route("/", methods=["GET", "POST"])
def home():
    context: dict[str, object] = {"loaded": False, "error": None, "graph_json": "null", "spectrum_id": None}

    if request.method == "POST":
        uploaded = request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            context["error"] = "Choose a CSV file to upload."
        else:
            filename = secure_filename(uploaded.filename) or "spectrum.csv"
            spectrum_id = uuid.uuid4().hex
            upload_path = UPLOAD_FOLDER / f"{spectrum_id}_{filename}"
            uploaded.save(upload_path)

            try:
                original_x, original_y, positions, amplitudes, widths, baseline = load_input(upload_path)
                SPECTRA[spectrum_id] = {
                    "original_x": as_python_float_array(original_x),
                    "original_y": as_python_float_array(original_y),
                    "baseline": float(baseline),
                }
                fig = create_plot(original_x, original_y, positions)
                context.update(
                    {
                        "loaded": True,
                        "spectrum_id": spectrum_id,
                        "point_count": len(original_x),
                        "baseline": float(baseline),
                        "peaks": peak_records(positions, amplitudes, widths),
                        "graph_json": json.dumps(json.loads(fig.to_json())),
                    }
                )
            except (KeyError, ValueError, IndexError, pd.errors.EmptyDataError) as exc:
                context["error"] = str(exc)

    return render_template_string(HTML_PAGE, **context)


@app.route("/regenerate", methods=["POST"])
def regenerate():
    try:
        spectrum_id, positions, amplitudes, widths = parse_peak_payload(request.get_json(silent=True))
        stored = SPECTRA[spectrum_id]
        original_x = np.array(stored["original_x"], dtype=float)
        original_y = np.array(stored["original_y"], dtype=float)
        baseline = float(stored["baseline"])

        synthetic_x, synthetic_y = generate_spectrum(positions, amplitudes, widths, baseline, x_axis=original_x)
        fig = create_plot(original_x, original_y, positions, synthetic_x, synthetic_y)

        output_id = uuid.uuid4().hex
        csv_name = f"{output_id}_spectrum.csv"
        peak_csv_name = f"{output_id}_peaks.csv"
        png_name = f"{output_id}_overlay.png"

        pd.DataFrame({"cm-1": synthetic_x, "%T": synthetic_y}).to_csv(OUTPUT_FOLDER / csv_name, index=False)
        pd.DataFrame(
            {
                "Position": positions,
                "Amplitude": amplitudes,
                "FWHM": widths,
                "Baseline": baseline,
            }
        ).to_csv(OUTPUT_FOLDER / peak_csv_name, index=False)

        png_url = None
        message = "Regenerated successfully."
        try:
            pio.write_image(fig, OUTPUT_FOLDER / png_name, width=1400, height=760, scale=2)
            png_url = url_for("download_output", filename=png_name)
        except (ValueError, RuntimeError, OSError) as exc:
            message = f"Regenerated CSV files. PNG export needs Plotly Kaleido support: {exc}"

        return jsonify(
            {
                "graph": json.loads(fig.to_json()),
                "csv": url_for("download_output", filename=csv_name),
                "peak_csv": url_for("download_output", filename=peak_csv_name),
                "png": png_url,
                "message": message,
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/download/<path:filename>")
def download_output(filename: str):
    safe_name = secure_filename(filename)
    if safe_name != filename:
        return jsonify({"error": "Invalid download filename."}), 400
    return send_from_directory(OUTPUT_FOLDER, safe_name, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)
