# -*- coding: utf-8 -*-

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import json
import util
import settings
from collections import defaultdict

import operator

'''
This script generates 500 recommendations for 1000 playlists in the challenge set with playlist title and first track
information only. We take a simple approach that iterates through MPD playlists one by one and create a popularity data
for the titles in challenge data set cold start scenario. For instance, for a playlist with title 'relax' in challenge
set (having the first track only), from all 'relax' playlists in MPD we create a frequency dict for songs appearing in
'relax' playlists and for all playlists that the first track appears in (no matter what title is).
We sort by decresing order and recommend top 500! Here, again the exact match of normalized titles are used.
'''


def load_cold_start_one_track_playlists(title_file_path):
    """
    Loads playlists in challenge set that has title and first track only!
    :param title_file_path:
    :return:
    """
    cstart_one_track_file = open(title_file_path, 'r')
    titles = []
    for line in cstart_one_track_file:
        line = line.strip().split(',')
        pID = line[0]
        trackID = line[1]
        name = line[2]
        titles.append((pID, trackID, name))
    cstart_one_track_file.close()
    return titles


def load_playlist_tracks(playlist):
    playlist_tracks = []
    for i, track in enumerate(playlist['tracks']):
        trackID = track['track_uri']
        playlist_tracks.append(trackID)
    return playlist_tracks


def main():
    MPD_DATA_PATH = settings.MPD_PATH
    CONVERTED_DATA_PATH = os.path.join(MPD_DATA_PATH, settings.EXP_DIR_NAME)
    rec_file = open(os.path.join(CONVERTED_DATA_PATH, 'title_one_song_popularity_recs.csv'), 'w')
    cold_start_song_playlists_file_path = os.path.join(CONVERTED_DATA_PATH, "one_song_playlists.csv")
    cold_start_one_track_playlist = load_cold_start_one_track_playlists(cold_start_song_playlists_file_path)
    path = os.path.join(MPD_DATA_PATH, settings.MPD_DATA_DIR_NAME)
    filenames = os.listdir(path)

    title_recs = dict()
    final_recs = defaultdict(list)

    for filename in sorted(filenames):
        print(filename)
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()

            slice = json.loads(js)
            for playlist in slice['playlists']:
                name = util.normalize_name(playlist['name']).encode('utf-8')
                playlist_tracks = load_playlist_tracks(playlist)
                for csp in cold_start_one_track_playlist:
                    val = 0
                    add = False
                    cID, cTrackID, cName = csp
                    if cName == name:
                        val += 1
                        add = True
                    if cTrackID in playlist_tracks:
                        val += 1
                        add = True
                    if add:
                        for trackID in playlist_tracks:
                            if trackID == cTrackID:
                                continue
                            if cID not in title_recs.keys():
                                temp_dict = dict()
                                temp_dict[trackID] = val
                                title_recs[cID] = temp_dict
                            else:
                                try:
                                    title_recs[cID][trackID] += val
                                except:
                                    title_recs[cID][trackID] = val
    for pID, candidates in title_recs.iteritems():
        temp = []
        for track in candidates.keys():
            temp.append((track, candidates[track]))
        temp.sort(key=operator.itemgetter(1), reverse=True)
        temp1 = temp[:500]

        mapped_recs = [val[0] for val in temp1]
        if len(mapped_recs) < 500:
            print("Not enough items to recommend!")
        print(len(temp1), len(mapped_recs))
        final_recs[pID] = mapped_recs

    for pID, recs in final_recs.iteritems():
        str_out = pID
        for rec in recs:
            str_out += ',' + str(rec)
        rec_file.write(str_out + '\n')
        rec_file.flush()
    rec_file.close()


if __name__ == '__main__':
    main()
