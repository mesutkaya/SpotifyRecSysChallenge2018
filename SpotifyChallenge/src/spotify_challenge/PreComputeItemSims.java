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
import nn.item.neighborhood.ItemNeighborhood;
import nn.item.sim.ItemSimilarity;
import nn.item.neighborhood.ItemNeighborhoods;
import nn.item.sim.ItemSimilarities;
import org.ranksys.formats.index.ItemsReader;
import org.ranksys.formats.index.UsersReader;


import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.ranksys.formats.parsing.Parsers.lp;

/**
 * Created by messe on 18/06/18.
 *
 * By using the training data (preprocessed from MPD and the IDs are mapped to integers) it pre-computes item-item
 * similarities between tracks. Pre-computed item-item similarities will be used for Sub-profile (sub-playlists)
 * extraction by SubProfileExtraction.java code and SPADReRanker.java code to re-rank recommendations generated by
 * Matrix Factorization code.
 *
 * Assuming that MPD and Challenge Set data is preprocessed by python scripts here only change MPD_PATH and run it.
 *
 */
public class PreComputeItemSims {
    public static void main(String[] args) throws IOException {
        String MPD_PATH = "/run/media/messe/roziklinux/spotify_challenge/mpd.v1/"; // replace with your path!
        String DATA_PATH = MPD_PATH + "/exp_data/";
        String userPath = DATA_PATH + "/u_index.txt";
        String itemPath = DATA_PATH + "/i_index.txt";
        String trainDataPath = DATA_PATH + "/mpd_converted_track_train.csv";

        int[] factors = {100};

        String SIM_PREFIX = "sim_ib_";
        String subProfileFilePath = "ibsp/";
        FastUserIndex<Long> userIndex = SimpleFastUserIndex.load(UsersReader.read(userPath, lp));
        FastItemIndex<Long> itemIndex = SimpleFastItemIndex.load(ItemsReader.read(itemPath, lp));

        //String testDataPath = foldDir + "/validation.csv";

        FastPreferenceData<Long, Long> trainData = SimpleFastPreferenceData.load(SpotifyPreferenceReader.get().read(trainDataPath, lp, lp), userIndex, itemIndex);
        System.out.println("Loaded training data!");
        ItemSimilarity<Long> isim = ItemSimilarities.vectorCosine(trainData, false);
        System.out.println("Created item-item similariy object!");

        for (int k : factors) {
            String simPath = DATA_PATH + "/" + subProfileFilePath + "/";
            Path path = Paths.get(simPath);
            if (Files.notExists(path)) {
                Files.createDirectory(path);
            }
            System.out.println(simPath + "\t" + Integer.toString(k));

            ItemNeighborhood<Long> itemKNN = ItemNeighborhoods.topK(isim, k);
            //ItemNeighborhood<Long> itemKNN = new TopKItemNeighborhood<>(isim, k);

            Path simFilePath = Paths.get((simPath + SIM_PREFIX + Integer.toString(k)));
            try (BufferedWriter writer = Files.newBufferedWriter(simFilePath)) {
                itemKNN.getAllItems().forEach(iidx -> {
                    final String[] temp = {Long.toString(iidx)};

                    itemKNN.getNeighbors(iidx).forEach(vs -> {
                        //if (itemSims.containsKey(iidx)) itemSims.get(iidx).put(vs.v1, vs.v2);
                        temp[0] += "," + Long.toString(vs.v1);
                    });
                    try {
                        writer.write(temp[0]);
                        writer.newLine();
                        writer.flush();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });

            }
        }

    }
}
