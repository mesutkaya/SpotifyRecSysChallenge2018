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
package reranker;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Created by Mesut Kaya on 19/06/18.
 *
 */
public class SPADIntentModel<U, I, S> extends SPIntentModel<U, I, S> {
    /**
     * user-item preference data
     */
    protected final FastPreferenceData<U, I> totalData;


    protected final Map<I, Map<I, Boolean>> itemSims;

    protected final FastItemIndex<I> itemIndex;
    /**
     * subprofile data
     */
    protected final Map<U, Map<S, List<I>>> spData;

    /**
     * Constructor that caches user intent-aware models.
     *
     * @param targetUsers user whose intent-aware models are cached
     * @param totalData   preference data
     * @param itemSims    item neighbourhoods
     * @param spData      subprofile data
     * @param itemIndex   item id indexes
     */
    public SPADIntentModel(Stream<U> targetUsers, FastPreferenceData<U, I> totalData,
                           Map<I, Map<I, Boolean>> itemSims, Map<U, Map<S, List<I>>> spData,
                           FastItemIndex<I> itemIndex) {
        super(targetUsers);
        this.totalData = totalData;
        this.itemSims = itemSims;
        this.spData = spData;
        this.itemIndex = itemIndex;
    }


    /**
     * Constructor that does not cache user intent-aware models.
     *
     * @param totalData preference data
     * @param spData    subprofile data
     * @param itemSims
     * @param itemIndex
     */
    public SPADIntentModel(FastPreferenceData<U, I> totalData, Map<U, Map<S, List<I>>> spData,
                           Map<I, Map<I, Boolean>> itemSims, FastItemIndex<I> itemIndex) {
        super();
        this.totalData = totalData;
        this.spData = spData;
        this.itemSims = itemSims;
        this.itemIndex = itemIndex;
    }

    @Override
    protected UserIntentModel<U, I, S> get(U user) {
        return new SPUserIntentModel(user);
    }

    public class SPUserIntentModel implements UserIntentModel<U, I, S> {
        /**
         * Map subprofile to p(S|u)
         */
        protected final Object2DoubleOpenHashMap<S> pSu;

        protected U user;

        public SPUserIntentModel(U user) {
            this.user = user;
            Object2DoubleOpenHashMap<S> tmpCounts = new Object2DoubleOpenHashMap<>();
            tmpCounts.defaultReturnValue(0.0);

            int[] norm = {0};
            if (!spData.containsKey(user)) {
                pSu = new Object2DoubleOpenHashMap<>();
                return;
            }
            spData.get(user).forEach((s, v) -> {
                tmpCounts.addTo(s, v.size());
                norm[0] += v.size();
            });

            pSu = new Object2DoubleOpenHashMap<>();
            tmpCounts.object2DoubleEntrySet().forEach(e -> {
                S s = e.getKey();
                pSu.put(s, e.getDoubleValue() / norm[0]);
            });
        }

        /**
         * {@inheritDoc}
         *
         * @return set of subprofiles as intents
         */
        @Override
        public Set<S> getIntents() {
            return pSu.keySet();
        }


        /**
         * {@inheritDoc}
         * This is the indicator function in our case! Since we don't know the direct relation between a recommended item
         * and a subprofile like feature intent model we look at the neighborhoods!
         *
         * @param i target item
         * @return Stream of subprofile descriptors if i is a neighbour of a member of a subprofile.
         */
        @Override
        public Stream<S> getItemIntents(I i) {
            List<S> itemIntents = new ArrayList<>();
            if (!spData.containsKey(user)) return itemIntents.stream();
            spData.get(user).forEach((s, v) -> {
                // if i is a neighbour of a member of v(v is a subprofile).
                boolean result = v.stream().anyMatch(j -> isNeighbourHood(j, i));
                if (result) itemIntents.add(s);
            });
            return itemIntents.stream();
        }


        /**
         * {@inheritDoc}
         *
         * @param s feature as intent
         * @return probability of the feature-intent
         */
        @Override
        public double pS_u(S s) {
            return pSu.getDouble(s);
        }
    }


    public boolean isNeighbourHood(I i1, I i2) {

        //return itemSims.containsKey(i1) && itemSims.get(i1).containsKey(i2);
        return itemSims.get(i1).containsKey(i2);
    }
}
