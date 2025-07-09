"""Visualization utilities for COX-2 molecular activity dataset analysis.

This module provides functions for creating comprehensive visualizations
and statistical analysis of the processed molecular activity data using pure Polars.
"""

import logging
from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# Setup logger for this module
logger = logging.getLogger(__name__)


class COX2DataVisualizer:
    """A class for creating visualizations of COX-2 molecular activity data.

    This class provides methods to create various plots and statistical
    visualizations for molecular activity analysis.
    """

    def __init__(self, width: int = 1200, height: int = 800) -> None:
        """Initialize the visualizer with default plotting parameters.

        Args:
            width: Default width for plotly plots.
            height: Default height for plotly plots.
        """
        self.width = width
        self.height = height

        # Color schemes
        self.colors = {
            "primary": "#2E8B57",
            "secondary": "#4682B4",
            "accent": "#FF6B6B",
            "warning": "#FFB347",
            "success": "#98FB98",
        }

        # Default plotly template
        self.template = "plotly_white"

        # Ensure data directory exists for saving plots
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)

    def plot_activity_distribution(
        self, df: pl.DataFrame, save_path: str | None = None, title: str = "COX-2 Activity Distribution Analysis"
    ) -> go.Figure:
        """Plot the distribution of molecular activity values.

        Creates a comprehensive visualization of molecular activity including IC50 and pIC50
        distributions, activity class comparisons, and log-scale transformations.

        Args:
            df: Polars DataFrame with activity data containing 'standard_value_nm' column.
            save_path: Optional path to save the plot as PNG or HTML file.
            title: Title for the plot.

        Returns:
            Plotly figure object with interactive visualizations.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        logger.info("Creating activity distribution plots...")

        # Validate required columns
        if "standard_value_nm" not in df.columns:
            raise ValueError("DataFrame must contain 'standard_value_nm' column")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Distribution of IC50 Values",
                "Log-scale Distribution of IC50 Values",
                "IC50 by Activity Class",
                "Distribution of pIC50 Values",
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
        )

        # Get IC50 values (remove nulls)
        ic50_values = df.filter(pl.col("standard_value_nm").is_not_null())["standard_value_nm"].to_numpy()

        if len(ic50_values) > 0:
            fig.add_trace(
                go.Histogram(
                    x=ic50_values, nbinsx=50, name="IC50 Distribution", marker_color=self.colors["primary"], opacity=0.7
                ),
                row=1,
                col=1,
            )

            # Log-scale histogram
            log_ic50_values = np.log10(ic50_values[ic50_values > 0])
            if len(log_ic50_values) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=log_ic50_values,
                        nbinsx=50,
                        name="Log IC50 Distribution",
                        marker_color=self.colors["secondary"],
                        opacity=0.7,
                    ),
                    row=1,
                    col=2,
                )

        # Box plot by activity class
        if "activity_class" in df.columns and len(ic50_values) > 0:
            # Get unique activity classes
            activity_classes = df["activity_class"].unique().to_list()

            for activity_class in activity_classes:
                # Filter data for this activity class
                activity_data = df.filter(
                    (pl.col("activity_class") == activity_class) & pl.col("standard_value_nm").is_not_null()
                )["standard_value_nm"].to_numpy()

                if len(activity_data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=activity_data, name=str(activity_class), boxpoints="outliers", marker_color=self.colors["accent"]
                        ),
                        row=2,
                        col=1,
                    )

        # pIC50 distribution
        if "pic50" in df.columns:
            pic50_values = df.filter(pl.col("pic50").is_not_null())["pic50"].to_numpy()
            if len(pic50_values) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=pic50_values, nbinsx=50, name="pIC50 Distribution", marker_color=self.colors["accent"], opacity=0.7
                    ),
                    row=2,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            title=title, title_x=0.5, height=self.height, width=self.width, template=self.template, showlegend=False
        )

        # Update axes labels
        fig.update_xaxes(title_text="IC50 (nM)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="log10(IC50 nM)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Activity Class", row=2, col=1)
        fig.update_yaxes(title_text="IC50 (nM)", type="log", row=2, col=1)
        fig.update_xaxes(title_text="pIC50", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        if save_path:
            # Handle save path - if already contains data directory, use as is
            full_path = Path(save_path) if save_path.startswith(("./data/", "data/")) else self.data_dir / save_path

            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as HTML for interactivity, or PNG for static
            if str(full_path).endswith(".html"):
                fig.write_html(full_path)
            else:
                fig.write_image(full_path, width=1200, height=800)

            logger.info(f"Activity distribution plot saved to {full_path}")

        return fig

    def plot_molecular_properties(
        self, df: pl.DataFrame, save_path: str | None = None, title: str = "Molecular Properties Distribution"
    ) -> go.Figure:
        """Plot molecular property distributions with drug-likeness thresholds.

        Creates histograms for key molecular descriptors including molecular weight,
        LogP, hydrogen bond donors/acceptors, TPSA, and rotatable bonds with
        Lipinski's Rule of Five thresholds marked.

        Args:
            df: Polars DataFrame with molecular properties.
            save_path: Optional path to save the plot as PNG or HTML file.
            title: Title for the plot.

        Returns:
            Plotly figure object with property distributions.

        Raises:
            ValueError: If insufficient molecular properties are available.
        """
        properties = ["mol_weight", "log_p", "hbd", "hba", "tpsa", "rotatable_bonds"]
        available_props = [prop for prop in properties if prop in df.columns]

        if len(available_props) < 4:
            raise ValueError("At least 4 molecular properties are required for visualization")

        logger.info("Creating molecular properties distribution plots...")

        # Create subplots
        rows = 2
        cols = 3
        subplot_titles = [prop.replace("_", " ").title() for prop in available_props[:6]]

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        # Drug-like thresholds
        thresholds = {"mol_weight": 500, "log_p": 5, "hbd": 5, "hba": 10}

        for i, prop in enumerate(available_props[:6]):
            row = (i // cols) + 1
            col = (i % cols) + 1

            # Get property values
            prop_values = df.filter(pl.col(prop).is_not_null())[prop].to_numpy()

            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=prop_values,
                    nbinsx=30,
                    name=f"{prop.replace('_', ' ').title()}",
                    marker_color=self.colors["primary"],
                    opacity=0.7,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add threshold line if applicable
            if prop in thresholds:
                threshold_val = thresholds[prop]
                fig.add_vline(
                    x=threshold_val, line_dash="dash", line_color="red", annotation_text="Lipinski limit", row=row, col=col
                )

        # Update layout
        fig.update_layout(title=title, title_x=0.5, height=self.height, width=self.width, template=self.template)

        # Update axes labels
        for i, prop in enumerate(available_props[:6]):
            row = (i // cols) + 1
            col = (i % cols) + 1
            fig.update_xaxes(title_text=prop.replace("_", " ").title(), row=row, col=col)
            fig.update_yaxes(title_text="Frequency", row=row, col=col)

        if save_path:
            # Handle save path - if already contains data directory, use as is
            full_path = Path(save_path) if save_path.startswith(("./data/", "data/")) else self.data_dir / save_path

            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as HTML for interactivity, or PNG for static
            if str(full_path).endswith(".html"):
                fig.write_html(full_path)
            else:
                fig.write_image(full_path, width=self.width, height=self.height)

            logger.info(f"Molecular properties plot saved to {full_path}")

        return fig

    def plot_property_vs_activity(
        self, df: pl.DataFrame, save_path: str | None = None, title: str = "Molecular Properties vs Activity"
    ) -> go.Figure:
        """Plot molecular properties vs biological activity relationships.

        Creates scatter plots showing correlations between molecular descriptors
        and biological activity, colored by activity class if available.

        Args:
            df: Polars DataFrame with properties and activity data.
            save_path: Optional path to save the plot as PNG or HTML file.
            title: Title for the plot.

        Returns:
            Plotly figure object with property-activity correlations.

        Raises:
            ValueError: If insufficient data columns are available.
        """
        properties = ["mol_weight", "log_p", "tpsa", "rotatable_bonds"]
        available_props = [prop for prop in properties if prop in df.columns]

        if len(available_props) < 2 or "standard_value_nm" not in df.columns:
            raise ValueError("Need at least 2 molecular properties and activity data")

        logger.info("Creating molecular properties vs activity plots...")

        # Create subplots
        rows = 2
        cols = 2
        subplot_titles = [f"{prop.replace('_', ' ').title()} vs Activity" for prop in available_props[:4]]

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        # ==================================================================
        # Build colour palette for activity classes (consistent across plots)
        # ==================================================================
        if "activity_class" in df.columns:
            # Desired logical order
            logical_order = ["Low Activity", "Moderately Active", "Highly Active"]
            unique_classes = df["activity_class"].unique().to_list()
            # Preserve logical order, drop missing
            activity_classes_master = [cls for cls in logical_order if cls in unique_classes] + [
                cls for cls in unique_classes if cls not in logical_order
            ]

            palette = pc.qualitative.Plotly  # categorical palette ~10 colours
            color_map = {cls: palette[idx % len(palette)] for idx, cls in enumerate(activity_classes_master)}
        else:
            color_map = {}

        for i, prop in enumerate(available_props[:4]):
            row = (i // cols) + 1
            col = (i % cols) + 1

            if "activity_class" in df.columns:
                # Scatter plot colored by activity class
                activity_classes = activity_classes_master

                for activity_class in activity_classes:
                    # Filter data for this activity class
                    class_data = df.filter(
                        (pl.col("activity_class") == activity_class)
                        & pl.col(prop).is_not_null()
                        & pl.col("standard_value_nm").is_not_null()
                    )

                    if len(class_data) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=class_data[prop].to_numpy(),
                                y=class_data["standard_value_nm"].to_numpy(),
                                mode="markers",
                                name=str(activity_class),
                                legendgroup=str(activity_class),
                                opacity=0.6,
                                marker={"size": 8, "color": color_map.get(activity_class)},
                                showlegend=(i == 0),  # add legend item only once per group
                            ),
                            row=row,
                            col=col,
                        )
            else:
                # Filter out null values
                valid_data = df.filter(pl.col(prop).is_not_null() & pl.col("standard_value_nm").is_not_null())

                if len(valid_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_data[prop].to_numpy(),
                            y=valid_data["standard_value_nm"].to_numpy(),
                            mode="markers",
                            name=prop.replace("_", " ").title(),
                            opacity=0.6,
                            marker={"size": 8},
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

        # Update layout
        fig.update_layout(title=title, title_x=0.5, height=self.height, width=self.width, template=self.template)

        # Update axes labels
        for i, prop in enumerate(available_props[:4]):
            row = (i // cols) + 1
            col = (i % cols) + 1
            fig.update_xaxes(title_text=prop.replace("_", " ").title(), row=row, col=col)
            fig.update_yaxes(title_text="IC50 (nM)", type="log", row=row, col=col)

        if save_path:
            # Handle save path - if already contains data directory, use as is
            full_path = Path(save_path) if save_path.startswith(("./data/", "data/")) else self.data_dir / save_path

            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as HTML for interactivity, or PNG for static
            if str(full_path).endswith(".html"):
                fig.write_html(full_path)
            else:
                fig.write_image(full_path, width=1200, height=800)

            logger.info(f"Property vs activity plot saved to {full_path}")

        return fig

    def plot_correlation_matrix(
        self,
        df: pl.DataFrame,
        save_path: str | None = None,
        title: str = "Correlation Matrix of Molecular Properties and Activity",
    ) -> go.Figure:
        """Plot correlation matrix of molecular properties and activity.

        Creates a heatmap showing correlations between numerical molecular descriptors
        and biological activity values.

        Args:
            df: Polars DataFrame with numerical columns.
            save_path: Optional path to save the plot as PNG or HTML file.
            title: Title for the plot.

        Returns:
            Plotly figure object with correlation heatmap.

        Raises:
            ValueError: If insufficient numerical columns are available.
        """
        logger.info("Creating correlation matrix plot...")

        # Get numerical column types from Polars
        numerical_cols = []
        for col in df.columns:
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                numerical_cols.append(col)

        # Remove non-informative columns
        exclude_cols = ["is_valid_molecule", "is_active"]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if len(numerical_cols) < 3:
            raise ValueError("Need at least 3 numerical columns for correlation analysis")

        # Calculate correlation matrix using Polars
        # Convert to numpy for correlation calculation
        data_matrix = df.select(numerical_cols).fill_null(0).to_numpy()
        corr_matrix = np.corrcoef(data_matrix.T)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = np.where(~mask, corr_matrix, np.nan)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix_masked,
                x=numerical_cols,
                y=numerical_cols,
                colorscale="RdBu_r",
                zmid=0,
                text=np.around(corr_matrix_masked, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                showscale=True,
            )
        )

        fig.update_layout(
            title=title,
            title_x=0.5,
            width=min(800, self.width),
            height=min(800, self.height),
            template=self.template,
            xaxis={"side": "bottom"},
            yaxis={"side": "left"},
        )

        if save_path:
            # Handle save path - if already contains data directory, use as is
            full_path = Path(save_path) if save_path.startswith(("./data/", "data/")) else self.data_dir / save_path

            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as HTML for interactivity, or PNG for static
            if str(full_path).endswith(".html"):
                fig.write_html(full_path)
            else:
                fig.write_image(full_path, width=min(800, self.width), height=min(800, self.height))

            logger.info(f"Correlation matrix plot saved to {full_path}")

        return fig

    def generate_summary_report(self, df: pl.DataFrame) -> str:
        """Generate a comprehensive summary report of the dataset.

        Args:
            df: Processed Polars DataFrame.

        Returns:
            Formatted summary report as string.
        """
        report = []
        report.append("=" * 60)
        report.append("COX-2 DATASET SUMMARY REPORT")
        report.append("=" * 60)

        # Basic statistics
        report.append("\nDATASET OVERVIEW")
        report.append(f"Total molecules: {len(df):,}")
        report.append(f"Dataset shape: {df.shape}")

        # Activity statistics
        if "standard_value_nm" in df.columns:
            report.append("\nACTIVITY STATISTICS")
            report.append(
                f"IC50 range: {df.select(pl.col('standard_value_nm').min()).item():.2f} - {df.select(pl.col('standard_value_nm').max()).item():.2f} nM"
            )
            report.append(f"Median IC50: {df.select(pl.col('standard_value_nm').median()).item():.2f} nM")
            report.append(f"Mean IC50: {df.select(pl.col('standard_value_nm').mean()).item():.2f} nM")

        # Activity classification
        if "activity_class" in df.columns:
            report.append("\nACTIVITY CLASSIFICATION")
            activity_counts = df.group_by("activity_class").len()
            for row in activity_counts.iter_rows():
                class_name, count = row[0], row[1]
                percentage = (count / len(df)) * 100
                report.append(f"{class_name}: {count:,} ({percentage:.1f}%)")

        # Molecular properties
        if "mol_weight" in df.columns:
            report.append("\nMOLECULAR PROPERTIES")
            report.append(f"Average molecular weight: {df.select(pl.col('mol_weight').mean()).item():.1f} Da")

        if "log_p" in df.columns:
            report.append(f"Average LogP: {df.select(pl.col('log_p').mean()).item():.2f}")

        if "tpsa" in df.columns:
            report.append(f"Average TPSA: {df.select(pl.col('tpsa').mean()).item():.1f} Ų")

        # Lipinski compliance
        if all(col in df.columns for col in ["mol_weight", "log_p", "hbd", "hba"]):
            lipinski_compliant = df.filter(
                (pl.col("mol_weight") <= 500) & (pl.col("log_p") <= 5) & (pl.col("hbd") <= 5) & (pl.col("hba") <= 10)
            ).height

            compliance_rate = (lipinski_compliant / len(df)) * 100
            report.append("\nDRUG-LIKENESS")
            report.append(f"Lipinski compliant: {lipinski_compliant:,} ({compliance_rate:.1f}%)")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
