# Automatic Playlist Continuation using SubProfile-Aware Diversification

This repository contains the source code to reproduce the results obtained by our team "teamrozik" in the Spotify RecSys Challenge 2018, Creative Track. 

## Team Info
Team Name: teamrozik
Challenge Track: creative

## Dependencies
The main python module dependencies are:
- numpy
- pandas
- operator
- json

**Note**: We have used Python 2.7.12 :: Anaconda 4.1.1 (64-bit). For Python 3.x, it is still usable with minor modification. Some of our python module codes are adapted from "[vae_cf](https://github.com/dawenl/vae_cf)", for preprocessing the data, and also we use some of the code that is released together with MPD dataset. 

The main Java module dependencies can be found in the pom.xml file. Our implementation is based on "[RankSYS Framework](https://github.com/RankSys/RankSys)". src/main/java/nn folder is from the Github repository of RankSYS, since in the maven repositories the version of RankSys is 0.4.3 but on GitHub it is 0.4.4-snapshot and our implementation is based on the Github version. The code in src/mf is also from RankSys, we only added a random seed for the initialization of the latent factors of Matrix Factorization algorithm for reproducability. 

**Note**: We have used Java(TM) SE Runtime Environment (build 1.8.0_101-b13) in our experiments.    


## Approach

We submit to the Creative Track, because we use known tracks of the Challenge Set playlists to train our model. Using challenge set is considered as external data as well. 

We split this problem into two:

### Cold-Start-APC

We use Cold-Start-APC for the 1000 playlists in the Challenge Set that have title only and the 1000 playlists that have a title and one track (their first).

For the 1000 playlists having title only, we create a title popularity recommender, which basically recommends the tracks that are in the playlists in MPD having the same 'normalized' title (we use normalization method provided by Spotify, see util.py file). Details can be found in the title_popularity_recommendations.py file. 

For the 1000 playlists having title and first track only, we create a popularity recommender, but this time we consider not only 'normalized' title but also the playlists having the track as well. Details can be found in the title_one_song_popularity_recommendations.py file. 

Note that, for some playlists it can be the case that the recommender cannot recommend 500 tracks with non-zero scores. In this case, we fill the rest of the recommendations with the most popular tracks in MPD.
 
**Note** There is no doubt that our cold-start solution is rudimentary, and there are many ways it could be improved, perhaps especially by using external data sources.
### SPAD-APC

Our approach for the remaining 8000 playlists in the Challenge Set, having at least 5 tracks each, is based on our recent publication "[Accurate and Diverse Recommendations Using Item-Based SubProfiles](https://aaai.org/ocs/index.php/FLAIRS/FLAIRS18/paper/view/17600)", aiming to generate a set of recommendations where each recommendation is relevant but the set of recommendations is diverse. Ordinarily, SPAD diversifies a set of recommendations to cover the different tastes (subprofiles) that we extract from a user's profile. In SPAD-APC and treat each playlist as if it were a user's profile and sun our subprofile detection algorithm to create subprofiles of the playlists.

Our assumption is that in the user generated playlists as well, there are sub-profiles that corresponds to user's different interests or tastes. Our goal is to cover those subprofiles of the playlists while producing the final set of recommendations. 
 
For the rest of 8000 playlists we consider each playlist as a user and tracks as items to be used in a Matrix Factorization recommender. As we explain later, by using preprocess.py we get <playlistID,trackID,title> for the playlists in MPD and 8000 playlist in the challenge set. We filter the tracks appearing in 1 playlist only(not due to algorithmic reasons but due to resource limitations) and map the playlist and trackIDs. We use "[Fast ALS-based factorization of Pilászy, Zibriczky and Tikk.](https://dl.acm.org/citation.cfm?id=1864726)". We use RankSys implementation.  

After generating the baseline recommendations, we re-rank the recommendations for those 8000 playlsits based on their detected subprofiles. 

**Note** Our approach to detecting the subprofiles has undergone refinement. It differs from early version.  

## Producing recommendations

The python codes are under playlist_challenge folder. First thing to do is to change MPD_PATH and CHALLENGE_DATA_PATH  to the correct paths in your environment.
For filtering the dataset and preprocessing run the following command:
```
python preprocess.py 
```

After running the above script, run:

```
python title_popularity_recommendations.py
```

Then run:

```
python title_one_song_popularity_recommendations.py
```

The Java codes for our SubProfile-Aware Diversification (SPAD) are under SpotifyChallenge/src/main/java/. First, from the training data, we pre-compute track-track similarities by using src/main/java/spotify_challenge/PreComputeItemSims.java. Change MPD_PATH with the path of your environment.

Then for detecting the subprofiles of the 8000 playlists in the challenge set use src/main/java/spotify_challenge/SubProfileExtraction.java. Do not forget to change MPD_PATH.

Then run src/main/java/spotify_challenge/MFRecommenderExample.java to produce 500 recommendations for 8000 playlists in the challenge set. 

Re-rank recommendations produced in the previous step by using src/main/java/spotify_challenge/SPADReRanker.java. 

Run src/main/java/spotify_challenge/PopularityRecommenderExample.java to produce 500 popular recommendations that will be used in cold start playlists. 

**Note** We optimized the hyper-parameters of the MAtrix Factorization algorithm and SPAD by using 10000 random playlists as validation set. We split data of those playlists as 80% train and 20% validation. Optimized hyper-parameters are the one we use in the codes in this repository.


## Compiling by Maven and running the code!
As explained above in all the files under spotify_challenge package change MPD_PATH. Then the classes can be compiled and run as follows: 

```
JAVA_HOME=/usr/java/default/ mvn clean compile
export MAVEN_OPTS="-Xmx24000M"
mvn exec:java -Dexec.mainClass="spotify_challenge.PreComputeItemSims"
mvn exec:java -Dexec.mainClass="spotify_challenge.SubProfileExtraction"
mvn exec:java -Dexec.mainClass="spotify_challenge.MFRecommenderExample"
mvn exec:java -Dexec.mainClass="spotify_challenge.SPADReRanker"
mvn exec:java -Dexec.mainClass="spotify_challenge.PopularityRecommenderExample"
```

**NOTE** Do not forget to replace jdk path instead of /usr/java/default above. 
## Creating submissions.

Finally in the python module run:
```
python create_submission.py
```


