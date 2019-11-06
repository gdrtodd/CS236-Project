import os
import glob
import pypianoroll
from pypianoroll import Multitrack

base_dir = './lpd_5_cleansed'
collection_dir = './bass_midis'

for letter_1 in os.listdir(base_dir):
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

                    # tracks = multiroll.tracks
                    # to_remove = [idx for idx in range(len(tracks)) if tracks[idx].name != 'Bass' ]

                    # Remove all but the bass track
                    multiroll.remove_tracks([0,1,2,4])

                    save_dir = os.path.join(collection_dir, '{}-bass.mid'.format(name))
                    multiroll.write(save_dir)

