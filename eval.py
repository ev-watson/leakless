import argparse
import csv
import math
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

import config


def require_healpy():
    try:
        import healpy as hp
    except ModuleNotFoundError as exc:
        raise SystemExit("Healpy is required for evaluation. Run this in the project scientific environment.") from exc
    return hp


def nside_from_alm_len(alm_len: int) -> int:
    return int((math.sqrt(8 * alm_len + 1) - 1) // 6)


def lmax_from_alm_len(alm_len: int) -> int:
    return int(3 * nside_from_alm_len(alm_len) - 1)


def recombine(channel_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e_comb = channel_tensor[:, 0] + 1j * channel_tensor[:, 1]
    b_comb = channel_tensor[:, 2] + 1j * channel_tensor[:, 3]
    return e_comb, b_comb


@dataclass(frozen=True)
class Band:
    lmin: int
    lmax_exclusive: int

    @property
    def label(self) -> str:
        return f"{self.lmin}-{self.lmax_exclusive - 1}"


def parse_bands(value: Optional[str], lmax: int) -> list[Band]:
    if value is None:
        return [Band(int(lo), int(hi)) for lo, hi in config.BANDS]

    bands = []
    for item in value.split(","):
        lo, hi = item.split(":", 1)
        bands.append(Band(int(lo), int(hi)))

    for band in bands:
        if band.lmin < 0 or band.lmax_exclusive <= band.lmin or band.lmax_exclusive > lmax + 1:
            raise ValueError(f"Invalid band {band.label}; require 0 <= lmin < lmax_exclusive <= {lmax + 1}")
    return bands


def select_samples(data_path: str, trials: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 3 or data.shape[-1] != 2 * config.INPUT_DIM:
        raise ValueError(
            f"Expected stack shape (samples, alm_len, {2 * config.INPUT_DIM}); got {data.shape}"
        )

    rng = np.random.default_rng(seed)
    n = min(trials, len(data))
    indices = rng.choice(len(data), size=n, replace=False)
    samples = np.asarray(data[indices], dtype=np.float32)
    return indices, samples[..., :config.INPUT_DIM], samples[..., config.INPUT_DIM:]


def alm_parts(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.empty(vectors.shape[:1] + vectors.shape[1:2], dtype=np.complex128)
    b = np.empty_like(e)
    for i in range(len(vectors)):
        e[i], b[i] = recombine(vectors[i])
    return e, b


def bb_cls(vectors: np.ndarray, lmax: int) -> np.ndarray:
    hp = require_healpy()
    _, b_alms = alm_parts(vectors)
    cls = np.empty((len(vectors), lmax + 1), dtype=np.float64)
    for i, b_alm in enumerate(b_alms):
        cls[i] = hp.alm2cl(b_alm, lmax=lmax)
    return cls


def rho_cls(source: np.ndarray, target: np.ndarray, lmax: int, eps: float) -> np.ndarray:
    hp = require_healpy()
    _, source_b = alm_parts(source)
    _, target_b = alm_parts(target)
    rho = np.empty((len(source), lmax + 1), dtype=np.float64)
    for i, (src, tgt) in enumerate(zip(source_b, target_b)):
        src_cl = hp.alm2cl(src, lmax=lmax)
        tgt_cl = hp.alm2cl(tgt, lmax=lmax)
        cross = hp.alm2cl(src, tgt, lmax=lmax)
        rho[i] = cross / np.sqrt(np.maximum(src_cl * tgt_cl, eps))
    return rho


def band_mean(values: np.ndarray, band: Band) -> np.ndarray:
    return np.mean(values[:, band.lmin:band.lmax_exclusive], axis=1)


def relative_errors(source_cls: np.ndarray, target_cls: np.ndarray, bands: Iterable[Band], eps: float) -> dict[str, np.ndarray]:
    out = {}
    for band in bands:
        source = band_mean(source_cls, band)
        target = band_mean(target_cls, band)
        out[band.label] = (source - target) / np.maximum(np.abs(target), eps)
    return out


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    if n_bootstrap <= 0 or len(values) < 2:
        mean = float(np.mean(values))
        return mean, mean
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = np.mean(draws, axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def predict_model(ckpt_path: str, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for checkpoint evaluation. Run this in the project training environment.") from exc

    from models import Leakless

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = Leakless.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    outputs = np.empty_like(inputs, dtype=np.float32)
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        batch = torch.from_numpy(inputs[start:stop]).to(device=device, dtype=torch.float32)
        outputs[start:stop] = model.predict(batch).detach().cpu().numpy().astype(np.float32)
    return outputs


def alm_vectors_to_qu(vectors: np.ndarray, nside: int, lmax: int) -> list[tuple[np.ndarray, np.ndarray]]:
    hp = require_healpy()
    e_alms, b_alms = alm_parts(vectors)
    zero_t = np.zeros_like(e_alms[0])
    maps = []
    for e_alm, b_alm in zip(e_alms, b_alms):
        _, q_map, u_map = hp.alm2map([zero_t, e_alm, b_alm], nside=nside, lmax=lmax, pol=True)
        maps.append((q_map, u_map))
    return maps


def namaster_bb_by_band(vectors: np.ndarray, nside: int, lmax: int, bands: list[Band]) -> np.ndarray:
    hp = require_healpy()
    import pymaster as nmt

    mask = hp.read_map(config.MASK_FILE, field=1)
    mask = hp.ud_grade(mask, nside_out=nside)
    window = nmt.mask_apodization(mask, aposize=config.MASK_APOSCALE, apotype=config.MASK_APOTYPE)

    out = np.empty((len(vectors), len(bands)), dtype=np.float64)
    for i, (q_map, u_map) in enumerate(alm_vectors_to_qu(vectors, nside, lmax)):
        field = nmt.NmtField(window, [q_map, u_map], purify_b=True)
        bb = nmt.compute_coupled_cell(field, field)[3]
        for j, band in enumerate(bands):
            out[i, j] = np.mean(bb[band.lmin:band.lmax_exclusive])
    return out


def metric_rows(
    source_name: str,
    source_cls: np.ndarray,
    target_cls: np.ndarray,
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    bands: list[Band],
    rng: np.random.Generator,
    n_bootstrap: int,
    eps: float,
) -> list[dict[str, object]]:
    rows = []
    rel_errors = relative_errors(source_cls, target_cls, bands, eps)
    rho = rho_cls(source_vectors, target_vectors, target_cls.shape[1] - 1, eps)
    for band in bands:
        rel = rel_errors[band.label]
        abs_rel = np.abs(rel)
        ci_lo, ci_hi = bootstrap_ci(abs_rel, rng, n_bootstrap)
        rows.append({
            "metric": "alm_bb",
            "source": source_name,
            "band": band.label,
            "mean_relative_error": float(np.mean(rel)),
            "mean_abs_relative_error": float(np.mean(abs_rel)),
            "abs_relative_error_ci95_low": ci_lo,
            "abs_relative_error_ci95_high": ci_hi,
            "mean_rho": float(np.mean(band_mean(rho, band))),
            "n_samples": len(source_cls),
        })
    return rows


def namaster_rows(
    source_name: str,
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    nside: int,
    lmax: int,
    bands: list[Band],
    rng: np.random.Generator,
    n_bootstrap: int,
    eps: float,
) -> list[dict[str, object]]:
    source_bb = namaster_bb_by_band(source_vectors, nside, lmax, bands)
    target_bb = namaster_bb_by_band(target_vectors, nside, lmax, bands)
    rows = []
    for i, band in enumerate(bands):
        rel = (source_bb[:, i] - target_bb[:, i]) / np.maximum(np.abs(target_bb[:, i]), eps)
        abs_rel = np.abs(rel)
        ci_lo, ci_hi = bootstrap_ci(abs_rel, rng, n_bootstrap)
        rows.append({
            "metric": "namaster_pure_b_coupled",
            "source": source_name,
            "band": band.label,
            "mean_relative_error": float(np.mean(rel)),
            "mean_abs_relative_error": float(np.mean(abs_rel)),
            "abs_relative_error_ci95_low": ci_lo,
            "abs_relative_error_ci95_high": ci_hi,
            "mean_rho": "",
            "n_samples": len(source_vectors),
        })
    return rows


def write_rows(rows: list[dict[str, object]], output_path: Optional[str]) -> None:
    fieldnames = [
        "metric",
        "source",
        "band",
        "mean_relative_error",
        "mean_abs_relative_error",
        "abs_relative_error_ci95_low",
        "abs_relative_error_ci95_high",
        "mean_rho",
        "n_samples",
    ]
    stream = open(output_path, "w", newline="") if output_path else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output_path:
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Leakless alm-stack outputs against target BB and NaMaster pure-B baselines."
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Optional Leakless checkpoint to evaluate.")
    parser.add_argument("--data", type=str, default=config.DATA_FILE, help="Path to stacks.npy-style data.")
    parser.add_argument("--trials", type=int, default=64, help="Number of stack rows to sample.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Prediction batch size.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed for sample/bootstrap draws.")
    parser.add_argument("--bands", type=str, default=None, help="Comma-separated half-open bands, e.g. 2:20,20:40.")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap draws for CI columns; 0 disables.")
    parser.add_argument("--namaster", action="store_true", help="Include experimental coupled pure-B NaMaster rows.")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _, inputs, targets = select_samples(args.data, args.trials, args.seed)
    alm_len = inputs.shape[1]
    nside = nside_from_alm_len(alm_len)
    lmax = lmax_from_alm_len(alm_len)
    bands = parse_bands(args.bands, lmax)

    target_cls = bb_cls(targets, lmax)
    input_cls = bb_cls(inputs, lmax)
    rows = metric_rows("input", input_cls, target_cls, inputs, targets, bands, rng, args.bootstrap, eps=1e-30)

    output_vectors = None
    if args.ckpt:
        output_vectors = predict_model(args.ckpt, inputs, args.batch_size)
        output_cls = bb_cls(output_vectors, lmax)
        rows.extend(metric_rows("model", output_cls, target_cls, output_vectors, targets, bands, rng, args.bootstrap, eps=1e-30))

    shuffled = output_vectors if output_vectors is not None else inputs
    shuffled = shuffled[rng.permutation(len(shuffled))]
    shuffled_cls = bb_cls(shuffled, lmax)
    rows.extend(metric_rows("shuffled_model" if output_vectors is not None else "shuffled_input",
                            shuffled_cls, target_cls, shuffled, targets, bands, rng, args.bootstrap, eps=1e-30))

    if args.namaster:
        rows.extend(namaster_rows("input", inputs, targets, nside, lmax, bands, rng, args.bootstrap, eps=1e-30))
        if output_vectors is not None:
            rows.extend(namaster_rows("model", output_vectors, targets, nside, lmax, bands, rng, args.bootstrap, eps=1e-30))

    write_rows(rows, args.output)

    if args.namaster:
        print(
            "# Note: NaMaster rows use purify_b=True coupled spectra reconstructed from stored cut-sky E/B alms. "
            "The report still recommends preserving or regenerating original masked Q/U for a decoupled MASTER baseline.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
