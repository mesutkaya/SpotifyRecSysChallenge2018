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
import es.uam.eps.ir.ranksys.diversity.intentaware.AspectModel;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/**
 * @author Mesut Kaya on 07/11/2017.
 *
 * @param <U> user type
 * @param <I> item type
 * @param <S> aspect type
 */
public abstract class SPAspectModel<U, I, S> extends UserModel<U> {
    /**
     * SP Intent Model
     */
    protected SPIntentModel<U, I, S> intentModel;

    /**
     * Constructor taking the intent model
     *
     * @param spIntentModel intent model
     */
    public SPAspectModel(SPIntentModel<U, I, S> spIntentModel) {
        this.intentModel = spIntentModel;
    }

    @SuppressWarnings("unchecked")
    @Override
    public UserAspectModel getModel(U user) {
        return (UserAspectModel) super.getModel(user);
    }

    /**
     * User aspect model for {@link SPAspectModel}.
     */
    public abstract class UserAspectModel implements SPIntentModel.UserIntentModel<U, I, S> {
        private final SPIntentModel.UserIntentModel<U, I, S> uim;

        /**
         * Constructor taking user intent model.
         *
         * @param user user
         */
        public UserAspectModel(U user) {
            this.uim = intentModel.getModel(user);
        }

        /**
         * Returns an item aspect model from a list of scored items.
         *
         * @param items list of items with scores
         */
        public abstract ItemAspectModel<I, S> getItemAspectModel(List<Tuple2od<I>> items);

        @Override
        public Set<S> getIntents() {
            return uim.getIntents();
        }

        @Override
        public Stream<S> getItemIntents(I i) {
            return uim.getItemIntents(i);
        }

        @Override
        public double pS_u(S s) {
            return uim.pS_u(s);
        }
    }

    /**
     * Item aspect model for {@link AspectModel}.
     *
     * @param <I> item type
     * @param <S> aspect type
     */
    public interface ItemAspectModel<I, S> {
        /**
         * Returns probability of an item given an aspect
         *
         * @param iv item-value pair
         * @param s aspect
         * @return probability of an item given an aspect
         */
        public double pi_S(Tuple2od<I> iv, S s);
    }

}
