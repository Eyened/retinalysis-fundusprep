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
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--rgb_path", type=click.Path(), help="Output path for RGB images")
@click.option(
    "--ce_path", type=click.Path(), help="Output path for Contrast Enhanced images"
)
@click.option(
    "--bounds_path", type=click.Path(), help="Output path for CSV with image bounds"
)
@click.option("--n_jobs", type=int, default=4, help="Number of preprocessing workers")
def preprocess_folder(data_path, rgb_path, ce_path, bounds_path, n_jobs):
    """Preprocess fundus images for inference.

    DATA_PATH is the directory containing the original images to process.
    """
    data_path = Path(data_path)

    # Get all files in the data directory
    files = list(data_path.glob("*"))
    if not files:
        click.echo(f"No files found in {data_path}")
        return

    _run_preprocessing(
        files=files,
        rgb_path=rgb_path,
        ce_path=ce_path,
        bounds_path=bounds_path,
        n_jobs=n_jobs,
    )


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--rgb_path", type=click.Path(), help="Output path for RGB images")
@click.option(
    "--ce_path", type=click.Path(), help="Output path for Contrast Enhanced images"
)
@click.option(
    "--bounds_path", type=click.Path(), help="Output path for CSV with image bounds"
)
@click.option("--n_jobs", type=int, default=4, help="Number of preprocessing workers")
def preprocess_csv(csv_path, rgb_path, ce_path, bounds_path, n_jobs):
    """Preprocess fundus images listed in a CSV file.

    CSV_PATH is the path to a CSV file with a 'path' column containing file paths.

    If an 'id' column exists in the CSV, those values will be used as image identifiers
    instead of automatically generating them from filenames.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        if "path" not in df.columns:
            click.echo("Error: CSV must contain a 'path' column")
            return
    except Exception as e:
        click.echo(f"Error reading CSV file: {e}")
        return

    # Get file paths and convert to Path objects
    files = [Path(p) for p in df["path"]]
    existing_files = [f for f in files if f.exists()]

    if len(existing_files) == 0:
        click.echo("No valid files found in the CSV")
        return

    if len(existing_files) < len(files):
        missing_count = len(files) - len(existing_files)
        click.echo(f"Warning: {missing_count} files from the CSV do not exist")

    # Check if 'id' column exists and prepare ids list if it does
    ids = None
    if "id" in df.columns:
        # Create a list of IDs for files that exist
        path_to_id_map = dict(zip(df["path"], df["id"]))
        ids = [path_to_id_map[str(f)] for f in existing_files]
        click.echo("Using IDs from 'id' column in CSV")

    _run_preprocessing(
        files=existing_files,
        ids=ids,
        rgb_path=rgb_path,
        ce_path=ce_path,
        bounds_path=bounds_path,
        n_jobs=n_jobs,
    )
