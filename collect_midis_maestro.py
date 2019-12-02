import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def collect_midis(base_dir, collection_dir):
    """
    Collects .midi files from raw data into processed data folders as .midi
    """

    if not os.path.exists(collection_dir):
        print("Creating collection directory %s" % collection_dir)
        os.mkdir(collection_dir)

    # Find all of the track name directories
    track_paths = list(Path(base_dir).rglob('*.midi'))

    for track_path in tqdm(track_paths, desc='Collecting MIDI files', total=len(track_paths)):
        shutil.copy(str(track_path), os.path.join(collection_dir, str(track_path.name)))



if __name__ == "__main__":
    
    base_data_dir = './data_raw/maestro-v2.0.0'
    base_collection_dir = './data_processed/'
    full_collection_dir = os.path.join(base_collection_dir, 'maestro_tracks')

    print("Collecting MIDI files")
    collect_midis(base_data_dir, full_collection_dir)
