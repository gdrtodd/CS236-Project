import pypianoroll
from pypianoroll import Multitrack

data_dir = './lpd_5_cleansed/A/A/A/TRAAAGR128F425B14B/b97c529ab9ef783a849b896816001748.npz'

multiroll = Multitrack(data_dir)

for track in multiroll.tracks:
	print("Track name: ", track.name)
	# pypianoroll.write(track, './test_{}.mid'.format(track.name))

multiroll.remove_tracks([0,1,2,4])
multiroll.write('./test.mid')