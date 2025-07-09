"""Data processing utilities for COX-2 molecular activity dataset preparation.

This module provides functions for downloading, cleaning, and validating
molecular activity data from ChEMBL database using pure Polars for high performance.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# Suppress RDKit warnings
warnings.filterwarnings("ignore")
logging.getLogger("rdkit").setLevel(logging.ERROR)

# Setup logger for this module
logger = logging.getLogger(__name__)


class COX2DataProcessor:
    """A class for processing COX-2 molecular activity data from ChEMBL.

    This class provides methods to download, clean, and validate molecular
    activity data specifically for the COX-2 target (ChEMBL ID: CHEMBL230).

    All methods use pure Polars for high-performance data processing without pandas dependency.
    Downloaded data is automatically cached to avoid repeated API calls.
    """

    def __init__(
        self,
        target_chembl_id: str = "CHEMBL230",
        # Lipinski/physicochemical thresholds
        mw_max: float = 500.0,
        logp_max: float = 5.0,
        hbd_max: int = 5,
        hba_max: int = 10,
        tpsa_max: float = 140.0,
        rotatable_bonds_max: int = 10,
        ro5: bool = True,
        require_pchembl: bool = True,
    ) -> None:
        """Create a new dataset processor instance.

        Args:
            target_chembl_id: Target identifier in ChEMBL.
            mw_max, logp_max, hbd_max, hba_max: thresholds of Lipinski's rule of five.
            tpsa_max, rotatable_bonds_max: additional Veber-like constraints.
            ro5: If *False*, Lipinski filters are skipped (only TPSA/rotatable/positive activity kept).
        """
        self.target_chembl_id = target_chembl_id
        self.activity_client = new_client.activity
        self.molecule_client = new_client.molecule

        # Setup cache directory
        self.cache_dir = Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata for integrity checking
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"

        # Store thresholds
        self.mw_max = mw_max
        self.logp_max = logp_max
        self.hbd_max = hbd_max
        self.hba_max = hba_max
        self.tpsa_max = tpsa_max
        self.rotatable_bonds_max = rotatable_bonds_max
        self.ro5 = ro5
        self.require_pchembl = require_pchembl

    def download_activity_data(self, activity_type: str = "IC50", limit: int | None = None) -> pl.DataFrame:
        """Download activity data for COX-2 from ChEMBL database with caching.

        Downloads biological activity data from ChEMBL API and caches the results
        to avoid repeated API calls. Handles mixed data types appropriately for Polars.

        Args:
            activity_type: Type of biological activity to filter (default: IC50).
            limit: Maximum number of records to download (None for all available data).

        Returns:
            Polars DataFrame containing raw activity data from ChEMBL.

        Raises:
            ValueError: If no data is found for the specified parameters.
            RuntimeError: If data download or processing fails.
        """
        # Generate cache filename; include flag so filtered/unfiltered don’t mix
        limit_str = f"_limit_{limit}" if limit else "_all"
        pchembl_tag = "_pchembl" if self.require_pchembl else "_nopchembl"
        cache_filename = f"{self.target_chembl_id}_{activity_type}{limit_str}{pchembl_tag}.json"
        cache_path = self.cache_dir / cache_filename

        # Check if cached data exists
        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            try:
                with open(cache_path, encoding="utf-8") as f:
                    activities_list = json.load(f)
                logger.info(f"Loaded {len(activities_list)} cached activity records")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-downloading...")
                activities_list = None
        else:
            activities_list = None

        # Download if no valid cache
        if activities_list is None:
            try:
                logger.info(f"Downloading {activity_type} activity data for target {self.target_chembl_id}...")

                # Build filter parameters dynamically
                _flt = {
                    "target_chembl_id": self.target_chembl_id,
                    "standard_type": activity_type,
                }
                if self.require_pchembl:
                    _flt["pchembl_value__isnull"] = False

                activities = self.activity_client.filter(**_flt)

                if limit:
                    activities = activities[:limit]

                # Convert to list
                activities_list = list(activities)
                logger.info(f"Downloaded {len(activities_list)} activity records")

                if not activities_list:
                    msg = f"No {activity_type} data found for target {self.target_chembl_id}"
                    raise ValueError(msg)

                # Save to cache
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(activities_list, f, indent=2, default=str)
                    logger.info(f"Data cached to {cache_path}")

                    # Save cache metadata for integrity checking
                    self._save_cache_metadata(
                        cache_filename,
                        {
                            "target_chembl_id": self.target_chembl_id,
                            "activity_type": activity_type,
                            "limit": limit,
                            "record_count": len(activities_list),
                            "download_date": datetime.now().isoformat(),
                            "file_size_bytes": cache_path.stat().st_size,
                            "cache_file": cache_filename,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")

            except Exception as e:
                logger.error(f"Error downloading data: {str(e)}")
                msg = f"Failed to download ChEMBL data: {str(e)}"
                raise RuntimeError(msg)

        # Create Polars DataFrame with proper schema handling
        try:
            # Use increased infer_schema_length to handle mixed types
            # Convert to string first to handle mixed types, then cast appropriately
            df = pl.DataFrame(activities_list, infer_schema_length=10000)
            logger.debug(f"Initial dataset shape: {df.shape}")
            logger.debug(f"Dataset columns: {df.columns}")

            return df

        except Exception as e:
            logger.error(f"Error creating Polars DataFrame: {str(e)}")
            logger.info("Attempting alternative DataFrame creation...")

            # Alternative approach: let Polars infer from all data
            try:
                df = pl.DataFrame(activities_list, infer_schema_length=None)
                logger.info("Successfully created DataFrame with full schema inference")
                return df
            except Exception as e2:
                logger.error(f"Alternative DataFrame creation also failed: {str(e2)}")
                msg = f"Failed to create Polars DataFrame: {str(e)} / {str(e2)}"
                raise RuntimeError(msg)

    def _save_cache_metadata(self, cache_filename: str, metadata: dict[str, Any]) -> None:
        """Save metadata about cached file for integrity checking.

        Args:
            cache_filename: Name of the cache file.
            metadata: Dictionary with cache metadata.
        """
        try:
            # Load existing metadata or create new
            if self.cache_metadata_file.exists():
                with open(self.cache_metadata_file, encoding="utf-8") as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}

            # Update metadata for this cache file
            all_metadata[cache_filename] = metadata

            # Save updated metadata
            with open(self.cache_metadata_file, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def remove_useless_columns(self, df: pl.DataFrame, min_fill_rate: float = 0.05) -> pl.DataFrame:
        """Remove columns with too many missing values or low information content.

        Identifies and removes columns that have excessive null values and would not
        contribute meaningfully to analysis or modeling.

        Args:
            df: Polars DataFrame with potentially useless columns.
            min_fill_rate: Minimum fraction of non-null values required to keep column (default: 5%).

        Returns:
            Polars DataFrame with useless columns removed.

        Note:
            Critical columns (SMILES, activity values) are never removed regardless of fill rate.
        """
        initial_columns = len(df.columns)

        # Critical columns that should never be removed
        critical_columns = {
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_type",
            "standard_value",
            "standard_units",
            "standard_value_nm",
            "pchembl_value",
            "activity_class",
            "is_active",
            "pic50",
        }

        columns_to_remove = []
        analysis_results = []

        for col in df.columns:
            null_count = df[col].null_count()
            total_count = len(df)
            fill_rate = (total_count - null_count) / total_count

            analysis_results.append(
                {"column": col, "null_count": null_count, "fill_rate": fill_rate, "is_critical": col in critical_columns}
            )

            # Remove if fill rate is too low and not critical
            if fill_rate < min_fill_rate and col not in critical_columns:
                columns_to_remove.append(col)

        # Log analysis results
        logger.info("Column fill rate analysis:")
        for result in sorted(analysis_results, key=lambda x: x["fill_rate"]):
            status = "CRITICAL" if result["is_critical"] else ("REMOVE" if result["fill_rate"] < min_fill_rate else "KEEP")
            logger.info(f"  {result['column']}: {result['fill_rate']:.1%} filled - {status}")

        # Remove identified columns
        if columns_to_remove:
            df_cleaned = df.drop(columns_to_remove)
            logger.info(f"Removed {len(columns_to_remove)} useless columns: {columns_to_remove}")
        else:
            df_cleaned = df
            logger.info("No useless columns found to remove")

        logger.info(f"Column count: {initial_columns} → {len(df_cleaned.columns)}")
        return df_cleaned

    def select_relevant_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Select and rename relevant columns for analysis.

        Filters the raw ChEMBL activity data to include only the columns
        necessary for downstream analysis and model building.

        Args:
            df: Raw ChEMBL activity Polars DataFrame with all available columns.

        Returns:
            Polars DataFrame with selected relevant columns for analysis.

        Note:
            Only columns that exist in the input DataFrame will be selected.
        """
        base_columns = [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_type",
            "standard_value",
            "standard_units",
            "pchembl_value",
        ]

        # Always keep base columns if they exist
        selected = [col for col in base_columns if col in df.columns]

        # Heuristic: keep additional columns whose null% < 80 %
        null_stats = df.null_count().row(0)
        total_rows = len(df)
        for col, nulls in zip(df.null_count().columns, null_stats, strict=False):
            if col in selected:
                continue
            if (nulls / total_rows) < 0.8:
                selected.append(col)

        df_selected = df.select(selected)

        logger.info(f"Column selection: kept {len(selected)} of {len(df.columns)} columns (≥20% non-null)")
        return df_selected

    def standardize_activity_units(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert all activity values to nanomolar (nM) units.

        Standardizes activity measurements from various units (μM, mM, M, pM, fM)
        to nanomolar for consistent analysis. Creates a new standardized column
        while preserving original data.

        Args:
            df: Polars DataFrame with activity data containing 'standard_value'
                and 'standard_units' columns.

        Returns:
            Polars DataFrame with additional 'standard_value_nm' column containing
            all activity values converted to nanomolar units.

        Note:
            Unknown units are left unconverted. Conversion factors are logged
            for verification.
        """
        # Step 1: Normalise unit strings for robust matching
        #  - Strip whitespace
        #  - Replace the micro sign ("µ" U+00B5) and Greek mu ("μ" U+03BC) with ASCII "u"
        #  - Convert to upper-case for case-insensitive comparison
        df = df.with_columns(
            [
                pl.when(pl.col("standard_units").is_not_null())
                .then(pl.col("standard_units").str.strip_chars().str.replace_all("[µμ]", "u").str.to_uppercase())
                .otherwise(None)
                .alias("_units_norm")
            ]
        )

        # Step 2: Define conversion factors (all to nM, keys must be upper-case after normalisation)
        unit_conversions = {
            "NM": 1.0,
            "UM": 1_000.0,  # micro-molar
            "MM": 1_000_000.0,  # milli-molar
            "M": 1_000_000_000.0,  # molar
            "PM": 0.001,  # pico-molar
            "FM": 0.000_001,  # femto-molar
        }

        # Step 3: Cast activity value to float once
        df = df.with_columns(pl.col("standard_value").cast(pl.Float64).alias("standard_value_nm"))

        # Step 4: Apply vectorised conversion using when/otherwise chain
        # Build expression dynamically to avoid Python loops over rows
        expr = pl.col("standard_value_nm")  # base expression
        for unit, factor in unit_conversions.items():
            expr = pl.when(pl.col("_units_norm") == unit).then(expr * factor).otherwise(expr)

        df = df.with_columns(expr.alias("standard_value_nm"))

        # Step 5: Logging statistics
        conversion_stats = df.group_by("_units_norm").count().filter(pl.col("_units_norm").is_in(list(unit_conversions.keys())))
        logger.info("Unit conversion summary (records per recognised unit):")
        for row in conversion_stats.iter_rows():
            logger.info(f"  {row[0]}: {row[1]}")

        # Identify unrecognised units for transparency
        unknown_units = (
            df.filter(~pl.col("_units_norm").is_in(list(unit_conversions.keys())) & pl.col("_units_norm").is_not_null())
            .select("_units_norm")
            .unique()
            .to_series()
            .to_list()
        )
        if unknown_units:
            logger.warning(f"Encountered unsupported activity units: {unknown_units}. These rows will be removed.")
            before_rows = len(df)
            df = df.filter((pl.col("_units_norm").is_null()) | (pl.col("_units_norm").is_in(list(unit_conversions.keys()))))
            logger.info(f"Removed {before_rows - len(df)} records with unsupported units.")

        # Step 6: Set final standardised unit column and drop helper column
        df = df.with_columns(pl.lit("nM").alias("standard_units_standardized")).drop("_units_norm")

        logger.info(f"All {len(df)} activity values converted/standardised to nM (where applicable)")

        return df

    def clean_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove records with missing critical values.

        Filters out records that lack essential data for analysis, specifically
        those missing SMILES structures or activity values. Reports the number
        of records removed for each criteria.

        Args:
            df: Polars DataFrame with activity data that may contain null values.

        Returns:
            Polars DataFrame with all records containing both valid SMILES
            and activity measurements.

        Note:
            Records removed are logged for transparency. This step is critical
            for ensuring data quality in downstream analysis.
        """
        initial_count = len(df)

        # Remove rows with missing SMILES
        df = df.filter(pl.col("canonical_smiles").is_not_null())
        smiles_removed = initial_count - len(df)

        # Remove rows with missing activity values
        df = df.filter(pl.col("standard_value_nm").is_not_null())
        activity_removed = initial_count - smiles_removed - len(df)

        logger.info(f"Removed {smiles_removed} records with missing SMILES")
        logger.info(f"Removed {activity_removed} records with missing activity values")
        logger.info(f"Remaining records: {len(df)}")

        return df

    def remove_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate records based on molecule and activity.

        Args:
            df: Polars DataFrame with activity data.

        Returns:
            Polars DataFrame with duplicates removed.
        """
        initial_count = len(df)

        # Remove exact duplicates
        df = df.unique()
        exact_duplicates_removed = initial_count - len(df)

        # Decide grouping key (prefer molecule_chembl_id when available)
        group_key = "molecule_chembl_id" if "molecule_chembl_id" in df.columns else "canonical_smiles"

        # Build aggregation expressions dynamically to preserve **all** columns
        agg_exprs: list[pl.Expr] = []
        for col, dtype in zip(df.columns, df.dtypes, strict=False):
            if col == group_key:
                # grouping key is retained automatically
                continue

            if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                agg_exprs.append(pl.col(col).median().alias(col))
            else:
                agg_exprs.append(pl.col(col).first().alias(col))

        df_grouped = df.group_by(group_key).agg(agg_exprs)

        molecule_duplicates_removed = len(df) - len(df_grouped)
        df = df_grouped

        logger.info(f"Removed {exact_duplicates_removed} exact duplicate records")
        logger.info(f"Removed {molecule_duplicates_removed} molecule duplicates (kept median activity)")
        logger.info(f"Final dataset size: {len(df)}")

        return df

    def validate_molecules(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate SMILES strings and calculate basic molecular properties.

        Args:
            df: Polars DataFrame with SMILES data.

        Returns:
            Polars DataFrame with validated molecules and molecular properties.
        """
        logger.info("Validating SMILES and calculating molecular properties...")

        valid_molecules = []
        invalid_count = 0

        # Get all rows as dictionaries for iteration
        rows = df.to_dicts()

        for row in tqdm(rows, desc="Validating molecules"):
            smiles = row["canonical_smiles"]

            try:
                mol = Chem.MolFromSmiles(smiles)

                if mol is not None:
                    # Calculate basic properties and add to row
                    row["mol_weight"] = Descriptors.MolWt(mol)
                    row["log_p"] = Descriptors.MolLogP(mol)
                    row["hbd"] = Descriptors.NumHDonors(mol)
                    row["hba"] = Descriptors.NumHAcceptors(mol)
                    row["tpsa"] = Descriptors.TPSA(mol)
                    row["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
                    row["aromatic_rings"] = Descriptors.NumAromaticRings(mol)
                    row["is_valid_molecule"] = True

                    valid_molecules.append(row)
                else:
                    invalid_count += 1

            except Exception:
                invalid_count += 1

        # Create Polars DataFrame directly from list of dictionaries
        df_valid = pl.DataFrame(valid_molecules)

        logger.info(f"Valid molecules: {len(df_valid)}")
        logger.info(f"Invalid molecules removed: {invalid_count}")

        return df_valid

    def apply_drug_like_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Lipinski's Rule of Five and other drug-likeness filters.

        Args:
            df: Polars DataFrame with molecular properties.

        Returns:
            Polars DataFrame with drug-like molecules only.
        """
        initial_count = len(df)

        # Build list of filter conditions
        criteria: list[tuple[str, pl.Expr]] = []

        if self.ro5:
            criteria.extend(
                [
                    (f"MW ≤ {self.mw_max}", pl.col("mol_weight") <= self.mw_max),
                    (f"logP ≤ {self.logp_max}", pl.col("log_p") <= self.logp_max),
                    (f"HBD ≤ {self.hbd_max}", pl.col("hbd") <= self.hbd_max),
                    (f"HBA ≤ {self.hba_max}", pl.col("hba") <= self.hba_max),
                ]
            )

        criteria.extend(
            [
                (f"TPSA ≤ {self.tpsa_max}", pl.col("tpsa") <= self.tpsa_max),
                (f"RotB ≤ {self.rotatable_bonds_max}", pl.col("rotatable_bonds") <= self.rotatable_bonds_max),
                ("Activity value > 0", pl.col("standard_value_nm") > 0),
            ]
        )

        # Combine conditions safely
        combined_expr = criteria[0][1]
        for _, expr in criteria[1:]:
            combined_expr = combined_expr & expr

        df_filtered = df.filter(combined_expr)

        # Logging per-condition removal statistics
        logger.info("Applied drug-likeness filters:")
        for desc, expr in criteria:
            removed = df.filter(~expr).shape[0]
            logger.info(f"  {desc}  → removed {removed}")

        logger.info(f"Total removed: {initial_count - len(df_filtered)}")
        logger.info(f"Drug-like molecules retained: {len(df_filtered)}")

        return df_filtered

    def add_activity_classes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add activity classification columns (activity_class, is_active, pic50).

        Args:
            df: Polars DataFrame with standardized activity values.

        Returns:
            Polars DataFrame with additional activity-related columns.
        """
        if "standard_value_nm" not in df.columns:
            logger.warning("'standard_value_nm' column not found – skipping activity classification")
            return df

        # Define activity thresholds (in nM)
        highly_active_threshold = 100
        moderately_active_threshold = 1000

        df = df.with_columns(
            [
                pl.when(pl.col("standard_value_nm") <= highly_active_threshold)
                .then(pl.lit("Highly Active"))
                .when(pl.col("standard_value_nm") <= moderately_active_threshold)
                .then(pl.lit("Moderately Active"))
                .otherwise(pl.lit("Low Activity"))
                .alias("activity_class"),
                pl.when(pl.col("standard_value_nm") <= moderately_active_threshold).then(1).otherwise(0).alias("is_active"),
                # pic50: negative log10(IC50 in molar units)
                pl.when(pl.col("standard_value_nm") > 0)
                .then(-pl.col("standard_value_nm").cast(pl.Float64).mul(1e-9).log10())
                .otherwise(None)
                .alias("pic50"),
            ]
        )

        # Log class distribution
        logger.info("Activity classification summary:")
        for cls, count in df.group_by("activity_class").count().iter_rows():
            logger.info(f"  {cls}: {count}")

        return df

    def save_dataset(self, df: pl.DataFrame, filename: str = "cox2_processed_dataset.parquet", save_csv: bool = False) -> None:
        """Save the processed dataset to Parquet file in ./data directory.

        Args:
            df: Processed Polars DataFrame to save.
            filename: Output filename (should end with .parquet).
            save_csv: Additionally save a CSV version (nested columns will be flattened to JSON strings).
        """
        # Ensure data directory exists
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)

        # Full path for the output file
        output_path = data_dir / filename

        # Save to Parquet (more efficient than CSV)
        df.write_parquet(output_path)
        logger.info(f"Dataset saved to {output_path}")

        if save_csv:
            csv_name = filename.rsplit(".", 1)[0] + ".csv"
            csv_path = data_dir / csv_name

            df_flat = self.flatten_nested_columns(df)
            df_flat.write_csv(csv_path)
            logger.info(f"Dataset saved to {csv_path} (flattened nested columns)")

    def flatten_nested_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert any List/Struct columns to JSON strings for CSV compatibility.

        Args:
            df: Polars DataFrame possibly containing nested columns.

        Returns:
            DataFrame with nested columns converted to pl.String.
        """
        nested_cols = [c for c, dt in df.schema.items() if dt.is_nested()]
        if not nested_cols:
            return df

        def _to_json_safe(val):
            if val is None:
                return None
            try:
                return json.dumps(val, default=lambda o: o.to_list() if isinstance(o, pl.Series) else str(o), ensure_ascii=False)
            except TypeError:
                # Fallback to string representation
                return str(val)

        flatten_exprs = [pl.col(c).map_elements(_to_json_safe, return_dtype=pl.String).alias(c) for c in nested_cols]
        logger.info(f"Flattened nested columns for CSV: {nested_cols}")
        return df.with_columns(flatten_exprs)
