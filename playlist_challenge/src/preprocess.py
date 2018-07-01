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
import sys
import json
import util
import pickle
import settings
import pandas as pd
import numpy as np


# Method below is from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


# Method below is modified from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
def filter_triplets(tp, min_uc=0, min_sc=2):
    """
    We use this method to filter the tracks appearing in only one playlist in the dataset.
    :param tp:
    :param min_uc:
    :param min_sc:
    :return:
    """
    # Only keep the triplets for items which were clicked on by at least min_sc playlists.
    if min_sc > 0:
        itemcount = get_count(tp, 'track')
        tp = tp[tp['track'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'playlist')
        tp = tp[tp['playlist'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'playlist'), get_count(tp, 'track')
    return tp, usercount, itemcount


# Method below is modified from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
def numerize(tp, profile2id, show2id):
    """
    Based on the new playlist and track id mappings numerize the training data and keep it in a new dataframe!
    :param tp:
    :param profile2id:
    :param show2id:
    :return:
    """
    uid = map(lambda x: profile2id[x], tp['playlist'])
    sid = map(lambda x: show2id[x], tp['track'])
    title = map(lambda x: x, tp['title'])
    return pd.DataFrame(data={'uid': uid, 'sid': sid, 'title': title}, columns=['uid', 'sid', 'title'])


# Method below is modified from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('playlist')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def preprocess_mpd(mpd_data_path, exp_data_path):
    """
    Since in our method we only use tracks and playlist names, from MPD we extract pID,trackID,title triplets and
    write them to another file to be used by our model!
    :param mpd_data_path:
    :param exp_data_path:
    :return:
    """
    if not os.path.exists(exp_data_path):
        os.makedirs(exp_data_path)
    mpd_track_train_file = open(os.path.join(exp_data_path, settings.MPD_TRACK_TRAIN_FILE_NAME), 'w')
    filenames = os.listdir(mpd_data_path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((mpd_data_path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            slice = json.loads(js)

            for playlist in slice['playlists']:
                pID = playlist['pid']
                title = util.normalize_name(playlist['name']).encode('utf-8')
                for i, track in enumerate(playlist['tracks']):
                    trackId = track['track_uri']
                    mpd_track_train_file.write(str(pID) + ',' + str(trackId) + ',' + title + '\n')
                mpd_track_train_file.flush()
    mpd_track_train_file.close()


def convert_mpd(exp_data_path, validation=True):
    """
    After extracting pID,trackID,title triplets from MPD and known tracks of the challenge set and writing to another file,
    now we preprocess the data and convert the IDS to integer. Here we filter tracks that appear in only one playlist.
    :param validation: if true randomly select 10000 playlist for validation set for hyper-parameter optimization!
    :param exp_data_path:
    :return:
    """
    # Load MPD track train file to pandas.
    raw_data = pd.read_csv(os.path.join(exp_data_path, settings.MPD_TRACK_TRAIN_FILE_NAME), header=None)
    # Manually add column names
    raw_data.columns = ["playlist", "track", "title"]
    print("Finished loading the MPD track train data!")
    # Below filter the tracks that appear only in one playlist in MPD, we will consider each playlist as a user in our model!
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print("After filtering, there are %d listening events from %d users and %d items (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    # For randomly selecting playlists for validation set.
    unique_uid = user_activity.index
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    unique_sid = pd.unique(raw_data['track'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pickle.dump(show2id, open(os.path.join(exp_data_path, "item_mappings.tsv"), 'wb'))
    pickle.dump(profile2id, open(os.path.join(exp_data_path, "user_mappings.tsv"), 'wb'))
    print("Finsihed dumping mappings to the file!")

    # If you want to create validation data for hyper-parameter optimization!
    if validation:
        val_data_path = os.path.join(exp_data_path, 'validation')
        if not os.path.exists(val_data_path):
            os.makedirs(val_data_path)
        n_users = unique_uid.size
        n_heldout_users = 10000

        tr_users = unique_uid[:(n_users - n_heldout_users)]
        vd_users = unique_uid[(n_users - n_heldout_users):]

        train_plays = raw_data.loc[raw_data['playlist'].isin(tr_users)]

        unique_sid = pd.unique(raw_data['track'])
        vad_plays = raw_data.loc[raw_data['playlist'].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays['track'].isin(unique_sid)]

        vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
        train_data = numerize(train_plays)
        train_data.to_csv(os.path.join(val_data_path, 'train.csv'), index=False)

        vad_data_tr = numerize(vad_plays_tr)
        vad_data_tr.to_csv(os.path.join(val_data_path, 'validation_tr.csv'), index=False)

        vad_data_te = numerize(vad_plays_te)
        vad_data_te.to_csv(os.path.join(val_data_path, 'validation_te.csv'), index=False)
        # TODO append 'validation_tr.csv' to the end of 'train.csv to train MF model!

    train_data = numerize(raw_data, profile2id, show2id)
    train_data.to_csv(os.path.join(exp_data_path, settings.MPD_CONVERTED_TRACK_TRAIN_FILE_NAME), index=False)
    print("Write the converted ids MPD train file to new file!")

    # Since Matrix Factorization model that we use from RankSYS wants files that has only user ids and item ids, below
    # we write the converted ids to index file to be used by RankSYS MF model!
    user_index = pd.unique(train_data['uid'])
    item_index = pd.unique(train_data['sid'])

    user_index_file = open(os.path.join(exp_data_path, settings.PLAYLIST_INDEX_FILE_NAME), 'w')
    item_index_file = open(os.path.join(exp_data_path, settings.TRACK_INDEX_FILE_NAME), 'w')

    for k in user_index:
        user_index_file.write(str(k) + '\n')
        user_index_file.flush()
    user_index_file.close()

    for k in item_index:
        item_index_file.write(str(k) + '\n')
        item_index_file.flush()
    item_index_file.close()


def create_mapped_challenge_train(u_mapping_path, i_mapping_path):
    """
    Assuming that by usings preprocess.py user and item mappings created, from challenge set we create train set
    for subprofile extraction and SPAD algorithm (For non-cold-start playlists)! There is no need to use the whole data set!
    Besides, write mapped piDs of the challenge set playlists to a file, because we will generate recommendations
    by using MF + SPAD for only those playlists.
    :param u_mapping_path:
    :param i_mapping_path:
    :return:
    """
    CHALLENGE_DATA_PATH = settings.CHALLENGE_DATA_PATH
    # Here write the list of unique tracks to a file maybe!
    MPD_DIR = settings.MPD_PATH
    EXP_DATA_DIR = os.path.join(MPD_DIR, settings.EXP_DIR_NAME)

    out_path = os.path.join(EXP_DATA_DIR, 'subprofile_train.csv')
    out_file = open(out_path, 'w')
    challenge_set_user_ids = set()
    # To write mapped challenge set playlist IDS to a file!
    challenge_set_user_ids_file = open(os.path.join(EXP_DATA_DIR, 'challenge_users.csv'), 'w')
    f = open(os.path.join(CHALLENGE_DATA_PATH, "challenge_set.json"))
    js = f.read()
    challenge_set = json.loads(js)

    user_mapping = pickle.load(open(u_mapping_path, 'rb'))
    item_mapping = pickle.load(open(i_mapping_path, 'rb'))
    f.close()

    for playlist in challenge_set['playlists']:
        ntracks = playlist['num_samples']
        if ntracks < 2:
            continue

        pID = playlist['pid']
        mapped_pid = user_mapping[pID]

        if 'name' in playlist.keys():
            name = util.normalize_name(playlist['name']).encode('utf-8')
        else:
            name = ""
        for track in playlist['tracks']:
            track_id = track['track_uri']
            track_mapped_id = item_mapping[track_id]
            out_file.write(str(mapped_pid) + ',' + str(track_mapped_id) + ',' + name + '\n')
            challenge_set_user_ids.add(mapped_pid)

        out_file.flush()
    for uID in challenge_set_user_ids:
        challenge_set_user_ids_file.write(str(uID) + '\n')
    challenge_set_user_ids_file.flush()
    challenge_set_user_ids_file.close()
    out_file.close()


def preprocess_challenge_set(CHALLENGE_DATA_PATH, EXP_DATA_PATH):
    """
    We will train our MF model by using known tracks in the challenge set so that for non-cold-start playlists in the
    challenge set we can learn latent factors (since we use challenge set as part of our train model this will be in
    creative track of the challenge!) we have to merge the known tracks in the challenge to the MPD train_track file!
    Besides we write the playlist with one song only to another file since our model for such playlist is different than others!
    For cold start scenarios see:
        title_popularity_recommendations.py => For playlists with title only.
        title_one_song_popularity_recommendations.py => For playlists with title + first track only.
    :param CHALLENGE_DATA_PATH:
    :param EXP_DATA_PATH:
    :return:
    """
    challenge_data_file = open(os.path.join(EXP_DATA_PATH, settings.MPD_TRACK_TRAIN_FILE_NAME), 'a')
    cold_start_song_playlists_file = open(os.path.join(EXP_DATA_PATH, "one_song_playlists.csv"), 'w')
    f = open(os.path.join(CHALLENGE_DATA_PATH, "challenge_set.json"))
    js = f.read()
    challenge_set = json.loads(js)
    f.close()

    for playlist in challenge_set['playlists']:
        if 'name' in playlist.keys():
            name = util.normalize_name(playlist['name']).encode('utf-8')
        else:
            name = ""
        nKnownTracks = playlist['num_samples']
        for track in playlist['tracks']:
            challenge_data_file.write(str(playlist['pid']) + ',' + str(track['track_uri']) + ',' + name + '\n')
            if nKnownTracks == 1:
                cold_start_song_playlists_file.write(
                    str(playlist['pid']) + ',' + str(track['track_uri']) + ',' + name + '\n')
                cold_start_song_playlists_file.flush()
        challenge_data_file.flush()
    challenge_data_file.close()
    cold_start_song_playlists_file.close()


if __name__ == '__main__':
    MPD_PATH = settings.MPD_PATH
    MPD_DATA_PATH = os.path.join(MPD_PATH, settings.MPD_DATA_DIR_NAME)
    EXP_DATA_PATH = os.path.join(MPD_PATH, settings.EXP_DIR_NAME)

    print("Getting <pID,trackID,title> triplets from MPD")
    preprocess_mpd(MPD_DATA_PATH, EXP_DATA_PATH)
    print("Getting <pID,trackID,title> triplets from MPD")
    preprocess_challenge_set(settings.CHALLENGE_DATA_PATH, EXP_DATA_PATH)

    print("Preprocessing the data!")
    convert_mpd(EXP_DATA_PATH, validation=False)
    # The below method creates pID,trackID,title triplets for the known tracks in the challenge set. They will be used
    # to extract sub-playlists (each of them representing a different sub-interest or sub-taste in that playlist.)
    create_mapped_challenge_train(os.path.join(EXP_DATA_PATH, "user_mappings.tsv"),
                                  os.path.join(EXP_DATA_PATH, "item_mappings.tsv"))
