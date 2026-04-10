"""
PyHealth dataset for the CCEP ECoG dataset.
Dataset link:
    https://openneuro.org/datasets/ds004080
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import mne_bids

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class CCEPECoGDataset(BaseDataset):
    """Dataset class for the CCEP ECoG dataset.

    Dataset is organized in BIDS format. This class indexes subjects who have
    all electrodes labeled, including at least one electrode in the Seizure
    Onset Zone (SOZ).  All signal preprocessing is handled by the task
    function (see ccep_ecog_task.py).

    After initialization, ``self.global_event_df`` is a Polars LazyFrame
    with one row per run and columns:

    * ``patient_id``, ``session_id``, ``task_id``, ``run_id``
    * ``header_file``    – path to the ``.vhdr`` signal file
    * ``events_file``    – path to the ``*_events.tsv`` for that run
    * ``channels_file``  – path to the ``*_channels.tsv`` for that run
    * ``electrodes_file``– path to the ``*_electrodes.tsv`` for that run's session
    * ``has_soz``        – bool, True if the subject has a complete SOZ column

    This is the structure ``BaseDataset.get_patient`` filters to build each
    ``Patient`` object passed to the task.

    Attributes:
        root (str): Root directory of the raw BIDS data.
        dataset_name (str): Name of the dataset.
    """

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes the CCEP ECoG dataset.

        Args:
            root (str): Root directory of the raw BIDS data.
            config_path (Optional[str]): Path to a PyHealth config YAML.

        Raises:
            FileNotFoundError: If ``root`` does not exist.
            ValueError: If the directory lacks the expected BIDS structure.

        Example::

            >>> dataset = CCEPECoGDataset(root="./data/ds004080")
            >>> patient = dataset.get_patient("ccepAgeUMCU01")
            >>> patient.data_source["has_soz"][0]
            True
        """
        self._verify_data(root)
        meta = self._index_data(root)

        super().__init__(
            root=root,
            tables=["ecog"],
            dataset_name="ccep_ecog",
            config_path=config_path,
            **kwargs,
        )

        # Populate the Polars LazyFrame and patient ID list that
        # BaseDataset.get_patient relies on.
        self.global_event_df = pl.from_pandas(meta).lazy()
        self.unique_patient_ids = meta["patient_id"].unique().tolist()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _verify_data(self, root: str) -> None:
        """Verifies the presence and structure of the dataset directory.

        Args:
            root (str): Root directory of the raw data.

        Raises:
            FileNotFoundError: If the dataset path does not exist.
            ValueError: If the dataset lacks subjects or core BIDS files.
        """
        if not os.path.exists(root):
            msg = f"Dataset path '{root}' does not exist"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not list(Path(root).glob("sub-*")):
            msg = f"BIDS root '{root}' contains no 'sub-*' subject folders"
            logger.error(msg)
            raise ValueError(msg)

        if not any(Path(root).rglob("*.vhdr")):
            msg = f"BIDS root '{root}' contains no '.vhdr' signal files"
            logger.error(msg)
            raise ValueError(msg)

        if not any(Path(root).rglob("*_electrodes.tsv")):
            msg = f"BIDS root '{root}' contains no '*_electrodes.tsv' files"
            logger.error(msg)
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Metadata indexing
    # ------------------------------------------------------------------

    def _index_data(self, root: str) -> pd.DataFrame:
        """Parses BIDS directory and indexes all run-level file paths.

        One row per (subject, run).  Columns:

        * ``patient_id``, ``session_id``, ``task_id``, ``run_id``
        * ``header_file``   – path to ``.vhdr``
        * ``events_file``   – path to ``*_events.tsv`` for that run
        * ``channels_file`` – path to ``*_channels.tsv`` for that run
        * ``electrodes_file``– path to ``*_electrodes.tsv`` for that run's session
        * ``has_soz``       – whether the subject has a complete SOZ column

        Writes the table to ``<root>/ccep_ecog-metadata-pyhealth.csv`` and
        returns it.

        Args:
            root (str): Root directory of the raw data.

        Returns:
            pd.DataFrame: The metadata table described above.
        """
        try:
            subjects = mne_bids.get_entity_vals(root, "subject")
        except FileNotFoundError:
            subjects = []

        rows = []
        root_path = Path(root)

        for sub in subjects:
            patient_dir = root_path / f"sub-{sub}"

            # Pre-pass: check whether any session for this subject has a fully
            # labeled SOZ column.  We scan all electrode files up front so the
            # has_soz flag is correct before we start appending run rows.
            has_soz = False
            for tsv_file in patient_dir.rglob("*electrodes.tsv"):
                try:
                    df = pd.read_csv(tsv_file, sep="\t")
                    cols = [c.lower() for c in df.columns]
                    if "soz" in cols:
                        col_series = df["soz"].str.lower()
                        if (col_series == "yes").any() and col_series.isin(["yes", "no"]).all():
                            has_soz = True
                            break
                except Exception as e:
                    logger.warning(f"Skipping electrode file {tsv_file}: {e}")

            # --- one row per run ---
            for header_file in patient_dir.rglob("*.vhdr"):
                entities = mne_bids.get_entities_from_fname(str(header_file))

                # All sibling TSV files are resolved from the same BIDSPath so
                # each run gets files from its own session, not the first session
                # found for the subject.
                run_bids_path = mne_bids.BIDSPath(
                    root=root,
                    subject=sub,
                    session=entities.get("session"),
                    task=entities.get("task"),
                    run=entities.get("run"),
                    datatype="ieeg",
                )

                # Session-specific electrode file (no task / run in the path)
                try:
                    electrodes_file = str(
                        run_bids_path.copy()
                        .update(task=None, run=None, extension=".tsv", suffix="electrodes")
                        .match()[0].fpath
                    )
                except Exception:
                    electrodes_file = ""

                try:
                    events_file = str(
                        run_bids_path.copy()
                        .update(extension=".tsv", suffix="events")
                        .match()[0].fpath
                    )
                except Exception:
                    events_file = ""

                try:
                    channels_file = str(
                        run_bids_path.copy()
                        .update(extension=".tsv", suffix="channels")
                        .match()[0].fpath
                    )
                except Exception:
                    channels_file = ""

                rows.append(
                    {
                        "patient_id":     sub,
                        "session_id":     entities.get("session", ""),
                        "task_id":        entities.get("task", ""),
                        "run_id":         entities.get("run", ""),
                        "header_file":    str(header_file),
                        "events_file":    events_file,
                        "channels_file":  channels_file,
                        "electrodes_file": electrodes_file,
                        "has_soz":        has_soz,
                    }
                )

        if not rows:
            logger.warning(
                "No valid BIDS ECoG header files (.vhdr) found. "
                "Ensure the root directory follows the BIDS structure "
                "(sub-*/ses-*/ieeg/*.vhdr)."
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values("patient_id", inplace=True)
            df.reset_index(drop=True, inplace=True)

        output_path = os.path.join(root, "ccep_ecog-metadata-pyhealth.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Wrote metadata index to {output_path}")

        return df

