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

import es.uam.eps.ir.ranksys.core.model.UserModel;
import es.uam.eps.ir.ranksys.diversity.intentaware.IntentModel;

import java.util.Set;
import java.util.stream.Stream;

/**
 * @author Mesut Kaya on 06/11/2017.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 * @param <S> type of the intent
 */
public abstract class SPIntentModel<U, I, S> extends UserModel<U> {
    /**
     * Constructor that caches user intent-aware models.
     *
     * @param targetUsers user whose intent-aware models are cached
     */
    public SPIntentModel(Stream<U> targetUsers) {
        super(targetUsers);
    }

    /**
     * Constructor that does not cache user intent-aware models.
     */
    public SPIntentModel() {
        super();
    }

    @Override
    protected abstract UserIntentModel<U, I, S> get(U user);

    @SuppressWarnings("unchecked")
    @Override
    public UserIntentModel<U, I, S> getModel(U user) {
        return (UserIntentModel<U, I, S>) super.getModel(user);
    }

    /**
     * User intent-aware model for {@link IntentModel}.
     * @param <U> user type
     * @param <I> item type
     * @param <S> feature type
     */
    public interface UserIntentModel<U, I, S> extends Model<U> {

        /**
         * Returns the intents considered in the intent model.
         *
         * @return the intents considered in the intent model
         */
        public abstract Set<S> getIntents();

        /**
         * Returns the intents associated with an item.
         *
         * @param i item
         * @return the intents associated with the item
         */
        public abstract Stream<S> getItemIntents(I i);

        /**
         * Returns the probability of an intent in the model.
         *
         * @param s intent
         * @return probability of an intent in the model
         */
        public abstract double pS_u(S s);
    }
}
