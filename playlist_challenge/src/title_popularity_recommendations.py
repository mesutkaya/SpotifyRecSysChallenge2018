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

from collections import defaultdict

import operator

import settings
import util
import os
import json

'''
This script generates 500 recommendations for 1000 playlists in the challenge set with playlist title information only.
We take a simple approach that iterates through MPD playlists one by one and create a popularity data for the titles
in challenge data set cold start scenario. For instance, for a playlist with title 'relax' in challenge set (having no known tracks)
from all 'relax' playlists in MPD we create a frequency dict for songs appearing in 'relax' playlists. We sort by decresing order
and recommend top 500!
In our approach we take a simple approach like the exact match of normalized titles. We were planning on string similarity
or semantic similarity of the titles but we did not have time to test these ideas.
'''
if __name__ == '__main__':
    MPD_PATH = settings.MPD_PATH
    MPD_DATA_PATH = os.path.join(MPD_PATH, settings.MPD_DATA_DIR_NAME)
    EXP_DATA_PATH = os.path.join(MPD_PATH, settings.EXP_DIR_NAME)

    CHALLENGE_DATA_PATH = settings.CHALLENGE_DATA_PATH

    rec_file = open(os.path.join(EXP_DATA_PATH, 'title_popularity_recs.csv'), 'w')

    print("Loading challenge dataset!")
    cold_start_titles = set()
    f = open(os.path.join(CHALLENGE_DATA_PATH, "challenge_set.json"))
    js = f.read()
    challenge_set = json.loads(js)
    f.close()
    for playlist in challenge_set['playlists']:
        if playlist['num_samples'] < 1:
            name = util.normalize_name(playlist['name']).encode('utf-8')
            cold_start_titles.add(name)
    filenames = os.listdir(MPD_DATA_PATH)

    title_recs = dict()
    final_recs = defaultdict(list)

    for filename in sorted(filenames):
        print(filename)
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((MPD_DATA_PATH, filename))
            f = open(fullpath)
            js = f.read()
            f.close()

            slice = json.loads(js)
            for playlist in slice['playlists']:
                pid = playlist['pid']
                name = util.normalize_name(playlist['name']).encode('utf-8')
                if name in cold_start_titles:
                    for i, track in enumerate(playlist['tracks']):
                        trackid = track['track_uri']
                        if name not in title_recs.keys():
                            temp_dict = dict()
                            temp_dict[trackid] = 1
                            title_recs[name] = temp_dict
                        else:
                            try:
                                title_recs[name][trackid] += 1
                            except:
                                title_recs[name][trackid] = 1
    for title, candidates in title_recs.iteritems():
        temp = []
        for track in candidates.keys():
            temp.append((track, candidates[track]))
        temp.sort(key=operator.itemgetter(1), reverse=True)
        temp1 = temp[:500]

        mapped_recs = [val[0] for val in temp1]
        if len(mapped_recs) < 500:
            print("Not enough items to recommend!")
        print(len(temp1), len(mapped_recs))
        final_recs[title] = mapped_recs

    for title, recs in final_recs.iteritems():
        str_out = title
        for rec in recs:
            str_out += ',' + str(rec)
        rec_file.write(str_out + '\n')
        rec_file.flush()
    rec_file.close()
    # Write cold start titles to a file just in case!
    cold_start_title_file = open(os.path.join(EXP_DATA_PATH, settings.COLD_START_TITLES_FILE_NAME), 'w')
    for title in cold_start_titles:
        cold_start_title_file.write(title + '\n')
    cold_start_title_file.flush()
    cold_start_title_file.close()
