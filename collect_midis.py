import os
import glob
import pypianoroll
from pypianoroll import Multitrack
from tqdm import tqdm

BASE_DIR = './data_raw/lpd_5_cleansed'
BASS_COLLECTION_DIR = './data_processed/bass_midis'
FULL_COLLECTION_DIR = './data_processed/full_midis'

def collect_midis(base_dir, collection_dir, selected_tracks="all"):
    """
    Collects .npz files from raw data into processed data folders as .mid
    - selected_track should be a list of track(s) 
        > options: ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings', 'all']
    """

    if not os.path.exists(collection_dir):
        print("Creating collection directory %s" % collection_dir)
        os.mkdir(collection_dir)

    for letter_1 in tqdm(os.listdir(base_dir)):
        cur = os.path.join(base_dir, letter_1)
        for letter_2 in os.listdir(cur):
            cur = os.path.join(base_dir, letter_1, letter_2)
            for letter_3 in os.listdir(cur):
                cur = os.path.join(base_dir, letter_1, letter_2, letter_3)
                for name in os.listdir(cur):
                    cur = os.path.join(base_dir, letter_1, letter_2, letter_3, name)

                    # Each name should correspond to just one file
                    assert len(os.listdir(cur)) == 1

                    for checksum in os.listdir(cur):
                        load_dir = os.path.join(cur, checksum)
                        multiroll = Multitrack(load_dir)

                        # Remove all but selected_tracks
                        if selected_tracks is not "all":
                            # Tracks: ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
                            to_remove = [idx for idx, track in enumerate(multiroll.tracks) \
                                            if track.name not in selected_tracks]
                            multiroll.remove_tracks(to_remove)  

                        save_dir = os.path.join(collection_dir, '{}-bass.mid'.format(name))
                        multiroll.write(save_dir)

if __name__ == "__main__":

    # collect full midi tracks
    print("Collecting full midi files...")
    collect_midis(BASE_DIR, FULL_COLLECTION_DIR)

    # collect bass
    print("Collecting bassline midi files...")
    collect_midis(BASE_DIR, BASS_COLLECTION_DIR, selected_tracks=["Bass"])



