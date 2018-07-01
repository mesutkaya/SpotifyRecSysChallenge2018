/* 
 * Copyright (C) 2015 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package nn.item.neighborhood;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import nn.neighborhood.Neighborhood;
import org.ranksys.core.util.tuples.Tuple2id;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.stream.Stream;

import static java.util.stream.StreamSupport.stream;

/**
 * Item neighborhood. Wraps a generic neighborhood and a fast item index.
 *
 * @author Saúl Vargas (saul.vargas@uam.es)
 * 
 * @param <I> type of the items
 */
public class ItemNeighborhood<I> implements FastItemIndex<I> {

    /**
     * Fast item index.
     */
    protected final FastItemIndex<I> iIndex;

    /**
     * Generic neighborhood.
     */
    protected final Neighborhood neighborhood;

    /**
     * Constructor.
     *
     * @param iIndex fast item index
     * @param neighborhood generic fast neighborhood
     */
    public ItemNeighborhood(FastItemIndex<I> iIndex, Neighborhood neighborhood) {
        this.iIndex = iIndex;
        this.neighborhood = neighborhood;
    }

    public Neighborhood neighborhood() {
        return neighborhood;
    }

    @Override
    public int numItems() {
        return iIndex.numItems();
    }

    @Override
    public I iidx2item(int iidx) {
        return iIndex.iidx2item(iidx);
    }

    @Override
    public int item2iidx(I i) {
        return iIndex.item2iidx(i);
    }

    /**
     * Returns the neighborhood of a user/index.
     *
     * @param idx user/index whose neighborhood is calculated
     * @return stream of user/item-similarity pairs.
     */
    public Stream<Tuple2id> getNeighbors(int idx) {
        return neighborhood.getNeighbors(idx);
    }

    /**
     * Returns a stream of item neighbors
     *
     * @param i item whose neighborhood is returned
     * @return a stream of item-score pairs
     */
    public Stream<Tuple2od<I>> getNeighbors(I i) {
        return stream(getNeighbors(item2iidx(i)).spliterator(), false)
                .map(this::iidx2item);
    }
}
