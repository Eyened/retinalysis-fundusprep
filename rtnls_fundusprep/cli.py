from pathlib import Path

import click
import pandas as pd

from rtnls_fundusprep.preprocessor import parallel_preprocess


@click.group(name="fundusprep")
def cli():
    pass


def _run_preprocessing(
    files, ids=None, rgb_path=None, ce_path=None, bounds_path=None, n_jobs=4
):
    """Common preprocessing function used by CLI commands.

    Args:
        files: List of Path objects to process
        ids: Optional list of IDs to use for the files
        rgb_path: Output path for RGB images
        ce_path: Output path for Contrast Enhanced images
        bounds_path: Output path for CSV with image bounds data
        n_jobs: Number of preprocessing workers
    """
    # Handle optional paths
    rgb_output = Path(rgb_path) if rgb_path else None
    ce_output = Path(ce_path) if ce_path else None

    if not files:
        click.echo("No valid files to process")
        return

    click.echo(f"Found {len(files)} files to process")

    # Run preprocessing
    bounds = parallel_preprocess(
        files,
        ids=ids,
        rgb_path=rgb_output,
        ce_path=ce_output,
        n_jobs=n_jobs,
    )

    # Save bounds if a path was provided
    if bounds_path:
        df_bounds = pd.DataFrame(bounds).set_index("id")
        bounds_output = Path(bounds_path)
        df_bounds.to_csv(bounds_output)
        click.echo(f"Saved bounds data to {bounds_output}")

    click.echo("Preprocessing complete")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--rgb_path", type=click.Path(), help="Output path for RGB images")
@click.option(
    "--ce_path", type=click.Path(), help="Output path for Contrast Enhanced images"
)
@click.option(
    "--bounds_path", type=click.Path(), help="Output path for CSV with image bounds"
)
@click.option("--n_jobs", type=int, default=4, help="Number of preprocessing workers")
def prep(input_path, rgb_path, ce_path, bounds_path, n_jobs):
    """Preprocess fundus images for inference.

    INPUT_PATH can be either:
    - A directory containing fundus images to process
    - A CSV/TXT file with a 'path' column containing image file paths

    If a CSV file is provided and it contains an 'id' column, those values will be used
    as image identifiers instead of automatically generating them from filenames.
    """
    input_path = Path(input_path)

    # Check if input is a file (CSV) or directory
    if input_path.is_file() and input_path.suffix.lower() in [".csv", ".txt"]:
        # Handle CSV/TXT file input
        try:
            df = pd.read_csv(input_path)
            if "path" not in df.columns:
                return click.echo("Error: CSV must contain a 'path' column")
        except Exception as e:
            return click.echo(f"Error reading CSV file: {e}")

        # Get file paths and convert to Path objects
        files = [Path(p) for p in df["path"]]

        # Check if 'id' column exists and prepare ids list if it does
        ids = None
        if "id" in df.columns:
            # Create a list of IDs for files that exist
            ids = [str(f) for f in df["id"]]
            click.echo("Using IDs from 'id' column in CSV")

    elif input_path.is_dir():
        # Handle directory input
        files = list(input_path.glob("*"))
        ids = None

    else:
        return click.echo(f"Error: {input_path} must be a directory or a CSV/TXT file")

    existing_files = [f for f in files if f.exists()]

    if len(existing_files) < len(files):
        missing_count = len(files) - len(existing_files)
        click.echo(f"Warning: {missing_count} input files do not exist")

    if len(existing_files) == 0:
        return click.echo("No valid files found.")

    _run_preprocessing(
        files=files,
        ids=ids,
        rgb_path=rgb_path,
        ce_path=ce_path,
        bounds_path=bounds_path,
        n_jobs=n_jobs,
    )
