"""Kaggle dataset download utility."""
import logging
import subprocess
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

def download_dataset(dest_dir: Path, kaggle_dataset: str, kaggle_file: str) -> Path:
    """Download and extract a dataset from Kaggle.
    
    Args:
        dest_dir: Destination directory for the dataset.
        kaggle_dataset: Kaggle dataset identifier (e.g., 'user/dataset').
        kaggle_file: Specific file to download from the dataset.
        
    Returns:
        Path to the extracted dataset file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(kaggle_file).name
    output_file = dest_dir / filename
    
    if output_file.exists():
        logger.info(f"Dataset already exists at {output_file}. Skipping download.")
        return output_file
        
    logger.info(f"Downloading dataset {kaggle_dataset}/{kaggle_file} to {dest_dir}...")
    
    try:
        # Run kaggle cli to download the file. It usually downloads as a zip.
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-f", kaggle_file, "-p", str(dest_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Download output: {result.stdout}")
        
        # Look for the downloaded zip file
        zip_path = dest_dir / (filename + ".zip")
        if not zip_path.exists():
             # Kaggle might download it with a different name based on dataset. Let's just find the zip.
             zips = list(dest_dir.glob("*.zip"))
             if zips:
                 zip_path = zips[0]
             else:
                 logger.error("Could not find downloaded zip file.")
                 # Maybe it downloaded the csv directly
                 if output_file.exists():
                     return output_file
                 raise FileNotFoundError("Downloaded file not found.")
                 
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
            
        # Clean up zip
        zip_path.unlink()
        
        if not output_file.exists():
            logger.error(f"Expected file {output_file} not found after extraction.")
            # Let's try to find it
            csvs = list(dest_dir.rglob(filename))
            if csvs:
                 # Move it to the root of dest_dir
                 csvs[0].rename(output_file)
            else:
                 raise FileNotFoundError(f"File {filename} not found in archive.")
        
        # Basic validation
        if output_file.stat().st_size < 1024 * 1024:
            logger.warning(f"Downloaded file {output_file} is suspiciously small (< 1MB).")
            
        logger.info(f"Successfully downloaded and extracted {output_file}.")
        return output_file
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download dataset via Kaggle CLI. Error: {e.stderr}")
        logger.error("Please ensure you have configured your Kaggle API credentials (~/.kaggle/kaggle.json).")
        logger.error(f"Alternatively, manually download the file and place it at {output_file}.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during download: {e}")
        raise
