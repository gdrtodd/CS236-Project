INPUT_DIRECTORY=../data_processed/midis_tracks=Piano

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=../data_processed/piano_notesequences.tfrecord

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive