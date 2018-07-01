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

import es.uam.eps.ir.ranksys.diversity.intentaware.ScoresAspectModel;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.List;

/**
 * Created by Mesut Kaya on 07/11/2017.
 *
 */
public class SubProfileAspectModel<U, I, S> extends SPAspectModel<U, I, S> {

    /**
     * Constructor taking an intent model and scores data.
     *
     * @param intentModel intent model
     */
    public SubProfileAspectModel(SPIntentModel<U, I, S> intentModel) {
        super(intentModel);
    }


    @Override
    protected UserAspectModel get(U u) {
        return new ScoresUserAspectModel(u);
    }

    /**
     * User aspect model for {@link ScoresAspectModel}.
     */
    public class ScoresUserAspectModel extends UserAspectModel {

        /**
         * Constructor.
         *
         * @param user user
         */
        public ScoresUserAspectModel(U user) {
            super(user);
        }

        @Override
        public ItemAspectModel<I, S> getItemAspectModel(List<Tuple2od<I>> items) {
            Object2DoubleOpenHashMap<S> probNorm = new Object2DoubleOpenHashMap<>();
            items.forEach(iv -> {
                getItemIntents(iv.v1).forEach(s -> {
                    probNorm.addTo(s, iv.v2);
                });
            });

            return (iv, s) -> iv.v2 / probNorm.getDouble(s);
        }
    }
}
