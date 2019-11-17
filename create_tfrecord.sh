# Creates .tfrecord file for a collection of .mid files
# Change INPUT_DIRECTORY and SEQUENCES_TFRECORD to input / output directory
#
# Modified from source: 
# https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md

# folder containing MIDI and/or MusicXML files. can have child folders.
INPUT_DIRECTORY=data_processed/midis_tracks=Bass

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=data_processed/tfrecords/bass_notesequences.tfrecord

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive