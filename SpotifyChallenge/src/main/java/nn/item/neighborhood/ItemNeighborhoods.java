package nn.item.neighborhood;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;

import nn.item.sim.ItemSimilarity;
import nn.neighborhood.CachedNeighborhood;
import nn.neighborhood.ThresholdNeighborhood;
import nn.neighborhood.TopKNeighborhood;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.stream.Stream;

import static org.ranksys.core.util.tuples.Tuples.tuple;

public class ItemNeighborhoods {

    public static <I> nn.item.neighborhood.ItemNeighborhood<I> topK(ItemSimilarity<I> similarity, int k) {
        return new ItemNeighborhood<>(similarity, new TopKNeighborhood(similarity.similarity(), k));
    }

    public static <I> nn.item.neighborhood.ItemNeighborhood<I> threshold(ItemSimilarity<I> similarity, double threshold) {
        return new ItemNeighborhood<>(similarity, new ThresholdNeighborhood(similarity.similarity(), threshold));
    }

    public static <I> nn.item.neighborhood.ItemNeighborhood<I> cached(nn.item.neighborhood.ItemNeighborhood<I> neighborhood) {
        return new ItemNeighborhood<>(neighborhood, new CachedNeighborhood(neighborhood.numItems(), neighborhood.neighborhood()));
    }

    public static <I> nn.item.neighborhood.ItemNeighborhood<I> cached(FastItemIndex<I> items, Stream<Tuple2<I, Stream<Tuple2od<I>>>> neighborhoods) {
        return new ItemNeighborhood<>(items, new CachedNeighborhood(items.numItems(), neighborhoods
                .map(t -> tuple(items.item2iidx(t.v1), t.v2.map(items::item2iidx)))));
    }

}
