"""
Utility functions for the encoding / decoding of MIDI objects
into our custom tuple encoding. Uses music21 to aid in the music
file and object parsing.
"""

import music21 as m21
import numpy as np
from fractions import Fraction
from collections import Counter
import sys
import os
import subprocess

def open_file(filename):
    """
    Opens a file (e.g. a generated .mid track) using the system's
    default. Handles windows and mac separately because python os
    module does not have the same functionality for both.
    """
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


def get_vocab(upper_limit=8):
    '''
    Returns the list of all tokens in the vocabulary
    '''
    all_timings = []
    for i in np.arange(0, upper_limit, 0.25):
        all_timings.append(Fraction(i))

    for i in np.arange(0, upper_limit*3):
        if i%3 != 0:
            all_timings.append(Fraction(i, 3))

    vocab = []

    for note in range(128):
        vocab.append("on_{}".format(note))

    for timing in all_timings:
        vocab.append("dur_{}".format(str(timing)))
        vocab.append("adv_{}".format(str(timing)))

    vocab.append("off_note")
    vocab.append("<start>")
    vocab.append("<end>")

    return vocab

def get_closest_timing(timing, max_dur=16):
    '''
    Returns the closest allowed timing to a provided value,
    as a float
    '''
    all_timings = np.arange(0, max_dur, 0.125)

    closest_idx = (np.abs(np.asarray(all_timings) - timing)).argmin()

    return all_timings[closest_idx]

def get_closest_timing_idx(timing, max_dur=16):
    all_timings = np.arange(0, max_dur, 0.125)
    closest_idx = (np.abs(np.asarray(all_timings) - timing)).argmin()

    return closest_idx

def split_encoding_by_measure(encoding, beats_per_measure=4):
    assert len(encoding)%3 == 0

    # This will become a list of lists, where each list is the encoding of
    # all notes that fall within a measure
    encodings_by_measure = [[]]

    # The current offset within the measure
    measure_offset = 0

    all_timings = np.arange(0, 16, 0.125)

    triples = (encoding[i:i+3] for i in range(0, len(encoding), 3))
    for idx, triple in enumerate(triples):
        pitch, duration_idx, advance_idx = triple 

        # Add the indices of the current note to the current measure
        encodings_by_measure[-1] += [3*idx, 3*idx+1, 3*idx+2]

        # # Add the encoding of this note to the current measure
        # encodings_by_measure[-1] += [pitch, duration_idx, advance_idx]

        advance = all_timings[advance_idx]

        measure_offset += advance

        if measure_offset > beats_per_measure:
            encodings_by_measure.append([])
            measure_offset = measure_offset%beats_per_measure

    return encodings_by_measure

def encode(stream, beats_per_measure=4):
    """
    Encode a midi stream into tokens of (pitch, duration, advance).
    Also include a measure encoding to denote which measure each
    token came from.
    """
    ids_encoding = []
    measure_encoding = []

    flattened = stream.flat

    for idx, element in enumerate(flattened[:-1]):
        next_element = flattened[idx+1]
        advance = next_element.offset - element.offset

        duration_idx = get_closest_timing_idx(element.duration.quarterLength)
        advance_idx = get_closest_timing_idx(advance)

        cur_measure = int(element.offset/beats_per_measure)

        if isinstance(element, m21.note.Note):
            ids_encoding.append(int(element.pitch.midi))            # pitch
            ids_encoding.append(duration_idx)                       # duration
            ids_encoding.append(advance_idx)                        # advance

            measure_encoding += [cur_measure, cur_measure, cur_measure]
        # We encode rests as the 0th MIDI note, hopefully it doesn't get used
        # for real!
        elif isinstance(element, m21.note.Rest):
            ids_encoding.append(0)                                  # pitch
            ids_encoding.append(duration_idx)                       # duration
            ids_encoding.append(advance_idx)                        # advance

            measure_encoding += [cur_measure, cur_measure, cur_measure]

        elif isinstance(element, m21.chord.Chord):
            # We add a 3-tuple for each note in the chord. For all notes
            # except the last, the advance should be 0 (so the notes play
            # simultaneously). For the last note of the chord, the last
            # advance should be the actual advance index
            for pitch in element.pitches:
                ids_encoding.append(int(pitch.midi))                # pitch
                ids_encoding.append(duration_idx)                   # duration
                ids_encoding.append(0)                              # advance

                measure_encoding += [cur_measure, cur_measure, cur_measure]

            # Manually change the last note's advance value
            ids_encoding[-1] = advance_idx


    return ids_encoding, measure_encoding

def decode(encoding):
    """
    Decode a (pitch, duration, advance) token encoding into
    a music21 stream (which can be saved to a .mid file).
    """
    assert len(encoding)%3 == 0

    stream = m21.stream.Stream()

    # The offset value of the current note / rest / chord
    cur_offset = 0.0

    all_timings = np.arange(0, 16, 0.125)

    triples = (encoding[i:i+3] for i in range(0, len(encoding), 3))

    for pitch, duration_idx, advance_idx in triples:
        duration = m21.duration.Duration(all_timings[duration_idx])
        advance = all_timings[advance_idx]

        if pitch == 0:
            note = m21.note.Rest(duration=duration)
        else:
            note = m21.note.Note(pitch, duration=duration)

        stream.insert(cur_offset, note)
        cur_offset += advance

    return stream

if __name__ == '__main__':

    # Test encode, decode of a midi file
    test_dir = './data_processed/midis_tracks=Piano/TRAAAGR128F425B14B-piano.mid'
    stream = m21.converter.parse(test_dir)

    enc, measure_enc = encode(stream)

    triples = ([enc[i:i+3], measure_enc[i]] for i in range(0, len(enc), 3))

    all_timings = np.arange(0, 16, 0.125)
    for note_triple, measure in triples:
        pitch, dur, _ = note_triple
        print("Note: {}\tDur: {}\tMeasure: {}".format(pitch, all_timings[dur], measure))

    dec = decode(enc)

    dec.write('midi', 'test.mid')
