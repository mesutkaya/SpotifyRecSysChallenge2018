/**
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
 */
package reranker;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.LambdaReranker;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

/**
 * Created by Mesut Kaya on 07/11/2017.
 *
 */
public class AlphaSPAD<U,I,S> extends LambdaReranker<U, I> {
    private final double alpha;
    protected final SPAspectModel<U,I,S> aspectModel;

    public AlphaSPAD(SPAspectModel<U,I,S> aspectModel, double alpha, double lambda, int cutoff, boolean norm){
        super(lambda, cutoff, norm);
        this.aspectModel = aspectModel;
        this.alpha = alpha;
    }

    @Override
    protected LambdaUserReranker getUserReranker(Recommendation<U, I> recommendation, int maxLength) {
        return new UserAlphaSPAD(recommendation, maxLength);
    }

    protected class UserAlphaSPAD extends LambdaUserReranker {
        private final SPAspectModel<U,I,S>.UserAspectModel uam;
        private final SPAspectModel.ItemAspectModel<I,S> iam;
        private final Object2DoubleOpenHashMap<S> redundancy;

        public UserAlphaSPAD(Recommendation<U, I> recommendation, int maxLength){
            super(recommendation, maxLength);
            this.uam = aspectModel.getModel(recommendation.getUser());
            this.iam = uam.getItemAspectModel(recommendation.getItems());
            this.redundancy = new Object2DoubleOpenHashMap<>();
            this.redundancy.defaultReturnValue(1.0);
        }

        @Override
        protected double nov(Tuple2od<I> iv) {
            return uam.getItemIntents(iv.v1)
                    .mapToDouble(s -> {
                        return uam.pS_u(s) * iam.pi_S(iv, s) * redundancy.getDouble(s);
                    })
                    .sum();
        }

        @Override
        protected void update(Tuple2od<I> biv) {
            uam.getItemIntents(biv.v1).sequential()
                    .forEach(s -> {
                        redundancy.put(s, redundancy.getDouble(s) * (1 - alpha * iam.pi_S(biv, s)));
                    });
        }
    }
}
