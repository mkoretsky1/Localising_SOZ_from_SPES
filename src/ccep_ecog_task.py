"""
PyHealth task for SOZ localization from SPES/CCEP data.

Each sample represents one electrode from one subject. The model predicts
whether that electrode lies within the Seizure Onset Zone (SOZ).

This module owns the full signal-processing pipeline:
  1. Load raw EEG + events + channels TSVs for every run.
  2. Band-pass filter, epoch, baseline-correct, and resample (StimulationDataProcessor).
  3. Average and compute std across trials; combine across runs (combine_stats).
  4. Filter stim-recording pairs by Euclidean distance > min_distance_mm.
  5. Sort remaining pairs by distance (distance becomes the last feature column).
  6. Yield one sample dict per electrode that appears in both recording and
     stimulation roles.

Usage::

    from ccep_ecog_dataset import CCEPECoGDataset
    from ccep_ecog_task import SOZPredictionTask

    dataset = CCEPECoGDataset(root="./data/ds004080")
    sample_dataset = dataset.set_task(SOZPredictionTask())
    sample = sample_dataset[0]
    print(sample["label"])
"""

import logging
from typing import Any, Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

_REGIONS = ["Frontal", "Insula", "Limbic", "Occipital", "Parietal", "Temporal", "Unknown"]
_REGION_TO_IDX = {r: i for i, r in enumerate(_REGIONS)}


class SOZPredictionTask(BaseTask):
    """Binary classification task: predict whether each electrode is in the SOZ.

    For each subject this task:
      1) Loads the raw BrainVision EEG for every run.
      2) Applies band-pass filtering (1–150 Hz), extracts SPES epochs
         (tmin–tmax seconds post-stimulus), baseline-corrects, and resamples
         to 512 Hz.
      3) Averages and computes trial-level std; combines statistics across runs
         for matching stim/recording pairs.
      4) Filters stim-recording pairs closer than ``min_distance_mm`` mm.
      5) Sorts remaining pairs by ascending Euclidean distance; distance is
         stored as the last column of each feature array.
      6) Returns one sample per electrode that appears in both a recording and
         a stimulation role.

    Each returned sample contains:
      - ``X_recording_mean``: torch.FloatTensor, shape (n_stim_pairs, time_steps + 1)
      - ``X_stim_mean``:      torch.FloatTensor, shape (n_stim_pairs, time_steps + 1)
      - ``X_recording_std``:  torch.FloatTensor, shape (n_stim_pairs, time_steps + 1)
      - ``X_stim_std``:       torch.FloatTensor, shape (n_stim_pairs, time_steps + 1)
      - ``coords``:           torch.FloatTensor, shape (3,)
      - ``lobe``:             int
      - ``label``:            int  (1 = SOZ, 0 = not SOZ)

    Examples::

        >>> from ccep_ecog_dataset import CCEPECoGDataset
        >>> from ccep_ecog_task import SOZPredictionTask
        >>> dataset = CCEPECoGDataset(root="./data/ds004080")
        >>> sample_dataset = dataset.set_task(SOZPredictionTask())
        >>> sample = sample_dataset[0]
        >>> print(sample["label"])
    """

    task_name: str = "soz_prediction"
    input_schema: Dict[str, str] = {
        "X_recording_mean": "tensor",
        "X_stim_mean":      "tensor",
        "X_recording_std":  "tensor",
        "X_stim_std":       "tensor",
        "coords":           "tensor",
        "lobe":             "tensor",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        tmin: float = 0.009,
        tmax: float = 1.0,
        destrieux_path: str = "../destrieux.rda",
        min_distance_mm: float = 13.0,
    ) -> None:
        """
        Args:
            tmin: Epoch start in seconds relative to stimulation onset.
                The first ``tmin`` seconds are excluded to remove the
                stimulation artefact. Defaults to 0.009 (9 ms).
            tmax: Epoch end in seconds relative to stimulation onset.
                Defaults to 1.0.
            destrieux_path: Path to ``destrieux.rda`` for lobe mapping.
            min_distance_mm: Minimum stim-recording Euclidean distance (mm)
                to retain. Defaults to 13.
        """
        super().__init__()
        self.tmin = tmin
        self.tmax = tmax
        self.min_distance_mm = min_distance_mm

        # Load Destrieux atlas once so it isn't re-read for every subject
        import pyreadr
        self._destrieux_df = pyreadr.read_r(destrieux_path)["destrieux"]

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_run(
        header_file: str,
        events_file: str,
        channels_file: str,
    ) -> Tuple[mne.io.BaseRaw, pd.DataFrame, pd.DataFrame]:
        """Load one BrainVision run and its paired events/channels TSVs.

        Args:
            header_file: Path to the ``.vhdr`` file.
            events_file: Path to the ``*_events.tsv`` file.
            channels_file: Path to the ``*_channels.tsv`` file.

        Returns:
            Tuple of (raw EEG, events DataFrame, channels DataFrame).
        """
        eeg = mne.io.read_raw_brainvision(header_file, preload=True, verbose=False)
        events_df   = pd.read_csv(events_file,   sep="\t", index_col=0)
        channels_df = pd.read_csv(channels_file, sep="\t", index_col=0)
        return eeg, events_df, channels_df

    @staticmethod
    def filter_by_distance(
        response_df: pd.DataFrame,
        electrodes_df: pd.DataFrame,
        min_distance_mm: float,
    ) -> pd.DataFrame:
        """Remove pairs closer than ``min_distance_mm`` and sort by distance.

        Distance is appended as the last numeric column so that
        ``select_dtypes(include='number')`` naturally includes it, matching
        the ``X[:, :, -1]`` convention used in dataset.py.

        Args:
            response_df: Single-metric response DataFrame.
            electrodes_df: Electrode metadata with x, y, z columns.
            min_distance_mm: Distance threshold in mm.

        Returns:
            Filtered and distance-sorted DataFrame, or the original DataFrame
            unchanged if coordinate lookup fails.
        """
        try:
            def _coords(electrodes):
                return np.array([
                    [electrodes_df.loc[e].x,
                     electrodes_df.loc[e].y,
                     electrodes_df.loc[e].z]
                    for e in electrodes
                ])

            stim_coords = (
                _coords(response_df.stim_1.values) +
                _coords(response_df.stim_2.values)
            ) / 2
            rec_coords = _coords(response_df.recording.values)
            distances = np.sqrt(np.sum((stim_coords - rec_coords) ** 2, axis=1))

            response_df = response_df[distances > min_distance_mm].copy()
            response_df["distances"] = distances[distances > min_distance_mm]
            response_df = response_df.sort_values("distances", ascending=True)

        except Exception as e:
            logger.warning(f"Distance filtering failed ({e}); skipping.")

        return response_df

    def _get_lobe_index(self, label: int) -> int:
        """Map a Destrieux integer label to a lobe index (0–6)."""
        if label == 0:
            return _REGION_TO_IDX["Unknown"]
        lobe_name = self._destrieux_df[
            self._destrieux_df.index == label - 1
        ].lobe.values[0]
        return _REGION_TO_IDX.get(lobe_name, _REGION_TO_IDX["Unknown"])

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one patient and return per-electrode SOZ prediction samples.

        Args:
            patient: A PyHealth ``Patient`` object returned by
                ``CCEPECoGDataset.get_patient``.  ``patient.data_source`` is
                a Polars DataFrame with one row per run containing
                ``header_file``, ``events_file``, ``channels_file``,
                ``electrodes_file``, and ``has_soz`` columns.

        Returns:
            List of sample dicts (one per electrode), or an empty list if the
            subject has no SOZ labels or no usable EEG data.
        """
        # Materialise the Polars DataFrame for this patient
        data = patient.data_source
        subject = patient.patient_id

        if not data["has_soz"][0]:
            return []

        from create_dataset import StimulationDataProcessor, combine_stats

        mne.set_log_level("WARNING")
        stim_processor = StimulationDataProcessor(tmin=self.tmin, tmax=self.tmax)

        # --- Step 1: process all runs ---
        run_dfs = []
        for row in data.iter_rows(named=True):
            if not all(row.get(k) for k in ("header_file", "events_file", "channels_file")):
                logger.warning(f"{subject} run {row.get('run_id')}: missing file path(s), skipping")
                continue
            try:
                eeg, events_df, channels_df = self.load_run(
                    row["header_file"], row["events_file"], row["channels_file"]
                )
                df = stim_processor.process_run_data(eeg, events_df, channels_df, subject)
                if df is not None:
                    run_dfs.append(df)
            except Exception as e:
                logger.warning(f"{subject} run {row.get('run_id')}: {e}")

        if not run_dfs:
            logger.warning(f"{subject}: no usable runs, returning no samples")
            return []

        # --- Step 2: combine stats across runs ---
        patient_df = pd.concat(run_dfs, ignore_index=True)
        grouped = patient_df.groupby(["recording", "stim_1", "stim_2"])
        patient_df = pd.concat(
            [pd.concat(combine_stats(g)) for _, g in grouped],
            ignore_index=True,
        )

        # --- Step 3: load electrode metadata ---
        # Each run row stores the session-specific electrode file; use the
        # first row (all runs in the same session share the same file).
        electrodes_file = data["electrodes_file"][0]
        try:
            electrodes_df = pd.read_csv(electrodes_file, sep="\t", index_col=0)
        except Exception as e:
            logger.warning(f"{subject}: could not load electrodes file: {e}")
            return []

        # --- Step 4: apply distance filtering per metric ---
        response_mean = self.filter_by_distance(
            patient_df[patient_df.metric == "mean"].copy(),
            electrodes_df, self.min_distance_mm,
        )
        response_std = self.filter_by_distance(
            patient_df[patient_df.metric == "std"].copy(),
            electrodes_df, self.min_distance_mm,
        )

        # --- Step 5: identify electrodes in both recording and stim roles ---
        stim_channels = (
            set(response_mean.stim_1.unique()) |
            set(response_mean.stim_2.unique())
        )
        recording_stim_channels = set(response_mean.recording.unique()) & stim_channels

        if not recording_stim_channels:
            logger.warning(f"{subject}: no channels appear in both recording and stim roles")
            return []

        # --- Step 6: build one sample per electrode ---
        samples: List[Dict[str, Any]] = []

        for channel in recording_stim_channels:
            rec_mean  = response_mean[response_mean.recording == channel].select_dtypes(include="number")
            rec_std   = response_std[response_std.recording   == channel].select_dtypes(include="number")

            stim_mask_mean = response_mean.stim_1.eq(channel) | response_mean.stim_2.eq(channel)
            stim_mask_std  = response_std.stim_1.eq(channel)  | response_std.stim_2.eq(channel)
            stim_mean = response_mean[stim_mask_mean].select_dtypes(include="number")
            stim_std  = response_std[stim_mask_std].select_dtypes(include="number")

            try:
                label = int(electrodes_df.loc[channel, "soz"].lower() == "yes")
            except (KeyError, AttributeError):
                logger.warning(f"{subject}/{channel}: missing SOZ label, skipping")
                continue

            try:
                coords = torch.FloatTensor([
                    electrodes_df.loc[channel, "x"],
                    electrodes_df.loc[channel, "y"],
                    electrodes_df.loc[channel, "z"],
                ])
            except KeyError:
                coords = torch.full((3,), float("nan"))

            try:
                lobe = self._get_lobe_index(
                    int(electrodes_df.loc[channel, "Destrieux_label"])
                )
            except (KeyError, IndexError):
                lobe = _REGION_TO_IDX["Unknown"]

            samples.append(
                {
                    "patient_id":       subject,
                    "electrode_name":   channel,
                    "X_recording_mean": torch.FloatTensor(rec_mean.values),
                    "X_stim_mean":      torch.FloatTensor(stim_mean.values),
                    "X_recording_std":  torch.FloatTensor(rec_std.values),
                    "X_stim_std":       torch.FloatTensor(stim_std.values),
                    "coords":           coords,
                    "lobe":             lobe,
                    "label":            label,
                }
            )

        logger.info(f"{subject}: built {len(samples)} electrode samples")
        return samples
