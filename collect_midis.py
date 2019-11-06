import os
from pathlib import Path
from tqdm import tqdm
import pypianoroll
from pypianoroll import Multitrack
from tqdm import tqdm

BASE_DIR = './data_raw/lpd_5_cleansed'
BASS_COLLECTION_DIR = './data_processed/bass_midis'
FULL_COLLECTION_DIR = './data_processed/full_midis'

def collect_midis(base_dir, collection_dir, selected_tracks=["all"]):
    """
    Collects .npz files from raw data into processed data folders as .mid
    - selected_track should be a list of track(s) 
        > options: ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings', 'all']
    """

    if not os.path.exists(collection_dir):
        print("Creating collection directory %s" % collection_dir)
        os.mkdir(collection_dir)

    selected_tracks.sort()  # to keep consistency in filename later

    # Find all of the track name directories
    track_paths = list(Path(base_dir).rglob('TR*'))

    for path in tqdm(track_paths, desc='Collecting MIDI files', total=len(track_paths)):
        for checksum in os.listdir(path):
            load_dir = os.path.join(path, checksum)
            multiroll = Multitrack(load_dir)

            # Remove all tracks but those in selected_tracks
            if "all" not in selected_tracks:

                to_remove = [idx for idx, track in enumerate(multiroll.tracks) \
                                if track.name not in selected_tracks]
                multiroll.remove_tracks(to_remove)

                # Make sure our selected tracks persist
                assert len(multiroll.tracks) == len(selected_tracks)

            # e.g. save_name = TR#########-bass-piano.mid
            name = os.path.basename(path)
            save_name = '{}-{}.mid'.format(name, "-".join(selected_tracks).lower())
            save_path = os.path.join(collection_dir, save_name)
            multiroll.write(save_path)

if __name__ == "__main__":

    # collect full midi tracks
    print("Collecting full midi files...")
    collect_midis(BASE_DIR, FULL_COLLECTION_DIR)

    # collect bass
    print("Collecting bassline midi files...")
    collect_midis(BASE_DIR, BASS_COLLECTION_DIR, selected_tracks=["Bass"])



