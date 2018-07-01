/**
 * # Licensed under the Apache License, Version 2.0 (the "License");
 * # you may not use this file except in compliance with the License.
 * # You may obtain a copy of the License at
 * #
 * #     http://www.apache.org/licenses/LICENSE-2.0
 * #
 * # Unless required by applicable law or agreed to in writing,
 * # software distributed under the License is distributed on an
 * # "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * # KIND, either express or implied.  See the License for the
 * # specific language governing permissions and limitations
 * # under the License.
 */
package spotify_challenge;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.PopularityRecommender;
import es.uam.eps.ir.ranksys.rec.runner.RecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilterRecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilters;
import org.jooq.lambda.Unchecked;
import org.ranksys.formats.index.ItemsReader;
import org.ranksys.formats.index.UsersReader;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.Supplier;

import static org.ranksys.formats.parsing.Parsers.lp;

/**
 * Created by Mesut Kaya on 30/06/18.
 * <p>
 * By using the training data (preprocessed from MPD and the IDs are mapped to integers) in generates top-500 popular
 * recommendations. We will use them later for our cold start scenario playlists. 1000 playlists with title only and
 * 1000 playlists with title + first track!
 * <p>
 * Assuming that MPD and Challenge Set data is preprocessed by python scripts here only change MPD_PATH and run it.
 */
public class PopularityRecommenderExample {
    public static void main(String[] args) throws IOException {
        String MPD_PATH = "/run/media/messe/roziklinux/spotify_challenge/mpd.v1/"; // replace with your path!
        String DATA_PATH = MPD_PATH + "/exp_data/";
        String userPath = DATA_PATH + "/u_index.txt";
        String itemPath = DATA_PATH + "/i_index.txt";
        String trainDataPath = DATA_PATH + "/mpd_converted_track_train.csv";
        FastUserIndex<Long> userIndex = SimpleFastUserIndex.load(UsersReader.read(userPath, lp));
        FastItemIndex<Long> itemIndex = SimpleFastItemIndex.load(ItemsReader.read(itemPath, lp));
        System.out.println("Start loading the training data!");
        FastPreferenceData<Long, Long> trainData = SimpleFastPreferenceData.load(SpotifyPreferenceReader.get().read(trainDataPath, lp, lp), userIndex, itemIndex);

        System.out.println("Finished loading the training data!");

        Map<String, Supplier<Recommender<Long, Long>>> recMap = new HashMap<>();

        recMap.put(DATA_PATH + "/pop", () -> {
            return new PopularityRecommender<>(trainData);
        });
        Set<Long> targetUsers = new HashSet<>();
        targetUsers.add((long) 0);

        RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(lp, lp);
        Function<Long, IntPredicate> filter = FastFilters.notInTrain(trainData);
        int maxLength = 500;
        RecommenderRunner<Long, Long> runner = new FastFilterRecommenderRunner<>(userIndex, itemIndex, targetUsers.stream(), filter, maxLength);

        recMap.forEach(Unchecked.biConsumer((name, recommender) -> {
            System.out.println("Running " + name);
            try (RecommendationFormat.Writer<Long, Long> writer = format.getWriter(name)) {
                runner.run(recommender.get(), writer);
            }
        }));
    }
}
