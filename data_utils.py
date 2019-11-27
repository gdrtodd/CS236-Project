import music21 as m21
import numpy as np
from fractions import Fraction
from collections import Counter


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

def encode(stream):
    encoding = []

    flattened = stream.flat

    for idx, element in enumerate(flattened[:-1]):
        next_element = flattened[idx+1]
        advance = next_element.offset - element.offset

        duration_idx = get_closest_timing_idx(element.duration.quarterLength)
        advance_idx = get_closest_timing_idx(advance)

        if isinstance(element, m21.note.Note):
            encoding.append(int(element.pitch.midi))            # pitch
            encoding.append(duration_idx)                       # duration
            encoding.append(advance_idx)                        # advance
        # We encode rests as the 0th MIDI note, hopefully it doesn't get used
        # for real!
        elif isinstance(element, m21.note.Rest):
            encoding.append(0)                                  # pitch
            encoding.append(duration_idx)                       # duration
            encoding.append(advance_idx)                        # advance

        elif isinstance(element, m21.chord.Chord):
            # We add a 3-tuple for each note in the chord. For all notes
            # except the last, the advance should be 0 (so the notes play
            # simultaneously). For the last note of the chord, the last
            # advance should be the actual advance index
            for pitch in element.pitches:
                encoding.append(int(pitch.midi))                # pitch
                encoding.append(duration_idx)                   # duration
                encoding.append(0)                              # advance

            # Manually change the last note's advance value
            encoding[-1] = advance_idx

    return encoding

def decode(encoding):
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
    test_dir = './data_processed/midis_tracks=Piano/TRAACQE12903CC706C-piano.mid'
    stream = m21.converter.parse(test_dir)

    enc = encode(stream)
    dec = decode(enc)

    dec.show('midi')
