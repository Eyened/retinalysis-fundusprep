# Retinalysis fundus preprocessing

Fundus bounds extraction, cropping and contrast enhancement


## Basic usage: running from the command line

We include command line utilities for running fundus preprocessing. Two commands:

- `preprocess-folder` use for running on a folder with input RGB images. Will not recurse into children of the input folder:

    ```bash
    fundusprep preprocess-folder <data_path> [OPTIONS]
    ```

- `preprocess-csv` use for more advanced usage to provide arbitrary filepaths and specific filenames (or IDs) for the outputs. Provide an input CSV file with columns `path` (for filepath to an RGB file) and `id` (optional) to name/identify the outputs

    ```bash
    fundusprep preprocess-csv ./image_list.csv \
    --rgb_path ./processed_rgb \
    --ce_path ./contrast_enhanced \
    --bounds_path ./metadata/bounds.csv
    ```

### Options

Both commands share the same options:

- `--rgb_path PATH`: Directory where processed RGB images will be saved
- `--ce_path PATH`: Directory where contrast-enhanced images will be saved
- `--bounds_path PATH`: Path to save a CSV file containing image bounds information
- `--n_jobs INTEGER`: Number of parallel processing workers (default: 4)


### Notes

- All output paths are optional - files will only be written when the corresponding path is provided
- Missing image files will be reported but won't stop the processing of other images
- The bounds CSV contains information about how images were cropped for standardization
- All output images are saved in PNG format with the same filename as the input image.


### Examples

#### Processing Folder with RGB Images

To process a folder of fundus images and save only the RGB versions along with the bounds information:

```bash
fundusprep preprocess-folder ./original_images \
  --rgb_path ./processed_rgb \
  --bounds_path ./metadata/bounds.csv
```

#### Processing with Contrast Enhancement

To process images with both RGB and contrast enhancement:

```bash
fundusprep preprocess-folder ./original_images \
  --rgb_path ./processed_rgb \
  --ce_path ./contrast_enhanced \
  --bounds_path ./metadata/bounds.csv
```


#### Processing Images Listed in a CSV (No Custom IDs)

Example CSV:
```
path
/data/images/patient1.jpg
/data/images/patient2.jpg
/data/images/patient3.png

To process images listed in a CSV file:

```bash
fundusprep preprocess-csv ./image_list.csv \
  --rgb_path ./processed_rgb \
  --ce_path ./contrast_enhanced \
  --bounds_path ./metadata/bounds.csv
```

The outputs will use the same filenames as the input images. For example, the RGB output for `/data/images/patient2.jpg` will be `./preprocessed_rgb/patient2.png`. Note that all outputs will be stored in a single folder, and therefore filenames should be unique. If filenames are not unique, use custom image IDs.

#### Using Custom Image IDs

The CSV file must include:
- A `path` column with absolute or relative paths to the image files
- an `id` column to specify custom identifiers for each image

Example CSV:
```
path,id
/data/images/patient1.jpg,P1_left
/data/images/patient2.jpg,P2_right
```

Processing is done in the same way:

```bash
fundusprep preprocess-csv ./patient_images.csv \
  --rgb_path ./processed_rgb \
  --ce_path ./contrast_enhanced \
  --bounds_path ./metadata/bounds.csv
```

The RGB output for `/data/images/patient2.jpg` will be `./preprocessed_rgb/P2_right.png`.