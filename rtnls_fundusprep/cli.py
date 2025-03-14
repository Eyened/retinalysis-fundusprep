from pathlib import Path

import click
import pandas as pd

from rtnls_fundusprep.preprocessor import parallel_preprocess


@click.group(name="fundusprep")
def cli():
    pass


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

    # Handle optional paths
    rgb_output = Path(rgb_path) if rgb_path else None
    ce_output = Path(ce_path) if ce_path else None

    # Get all files in the data directory
    files = list(data_path.glob("*"))
    if not files:
        click.echo(f"No files found in {data_path}")
        return

    click.echo(f"Found {len(files)} files to process")

    # Run preprocessing
    bounds = parallel_preprocess(
        files,
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
    # Handle optional paths
    rgb_output = Path(rgb_path) if rgb_path else None
    ce_output = Path(ce_path) if ce_path else None

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

    click.echo(f"Found {len(existing_files)} files to process")

    # Check if 'id' column exists and prepare ids list if it does
    ids = None
    if "id" in df.columns:
        # Create a list of IDs for files that exist
        path_to_id_map = dict(zip(df["path"], df["id"]))
        ids = [path_to_id_map[str(f)] for f in existing_files]
        click.echo("Using IDs from 'id' column in CSV")

    # Run preprocessing
    bounds = parallel_preprocess(
        existing_files,
        ids=ids,  # Pass ids if available, otherwise None
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
