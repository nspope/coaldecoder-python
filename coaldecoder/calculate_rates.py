#
# MIT License
#
# Copyright (c) 2021-2024 Nathaniel S. Pope
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Helper functions to extract trio coalescence rates from tree sequences.
"""

import numpy as np
import tskit
import numba

from numba import types
from math import exp, log, sqrt, lgamma, comb
from itertools import combinations_with_replacement as combn_replace


_void = types.void
_b = types.bool
_f = types.float64
_i = types.int32
_f1r = types.Array(_f, 1, 'C', readonly=True)
_f1w = types.Array(_f, 1, 'C', readonly=False)
_f2r = types.Array(_f, 2, 'C', readonly=True)
_f2w = types.Array(_f, 2, 'C', readonly=False)
_f3r = types.Array(_f, 3, 'C', readonly=True)
_f3w = types.Array(_f, 3, 'C', readonly=False)
_f4r = types.Array(_f, 4, 'C', readonly=True)
_f4w = types.Array(_f, 4, 'C', readonly=False)
_i1r = types.Array(_i, 1, 'C', readonly=True)
_i1w = types.Array(_i, 1, 'C', readonly=False)
_i2r = types.Array(_i, 2, 'C', readonly=True)
_i2w = types.Array(_i, 2, 'C', readonly=False)


@numba.njit(_void(_i1r, _i1r, _f1r, _f1r, _i1r, _i1r, _f, _i1r))
def _assert_complete_binary_trees(
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    insert_index,
    remove_index,
    sequence_length,
    sample_set_map,
):
    """
    Check that every tree is binary, rooted, and contains all samples with non-null values
    in `sample_set_map`
    """

    assert edges_parent.size == edges_child.size == edges_left.size == edges_right.size
    assert insert_index.size == remove_index.size == edges_parent.size

    num_nodes = max(edges_parent.max(), edges_child.max()) + 1
    num_edges = edges_parent.size
    num_sample_sets = sample_set_map.max() + 1
    max_children = 2

    insert_position = edges_left[insert_index]
    remove_position = edges_right[remove_index]
    nodes_parent = np.full(num_nodes, tskit.NULL)
    nodes_children = np.zeros(num_nodes, dtype=np.int32)
    sample_counts = np.zeros((num_nodes, num_sample_sets))
    sample_set_sizes = np.zeros(num_sample_sets)
    for n, i in enumerate(sample_set_map):  # initial state
        if i == tskit.NULL: continue
        sample_counts[n, i] = 1
        sample_set_sizes[i] += 1

    position = 0.0
    a, b = 0, 0
    edges = 0
    root = tskit.NULL
    while position < sequence_length:

        while b < num_edges and remove_position[b] == position: # edges out
            e = remove_index[b]
            p = edges_parent[e]
            c = edges_child[e]
            nodes_parent[c] = tskit.NULL
            nodes_children[p] -= 1
            update = sample_counts[c]
            while p != tskit.NULL:
                sample_counts[p] -= update
                p = nodes_parent[p]
            edges -= 1
            b += 1

        while a < num_edges and insert_position[a] == position: # edges in
            e = insert_index[a]
            p = edges_parent[e]
            c = edges_child[e]
            nodes_parent[c] = p
            nodes_children[p] += 1
            if nodes_children[p] > max_children:
                raise ValueError("Each node must have at most two children")
            update = sample_counts[c]
            while p != tskit.NULL:
                sample_counts[p] += update
                if nodes_parent[p] == tskit.NULL:
                    root = p
                p = nodes_parent[p]
            edges += 1
            a += 1

        if edges > 0 and not np.all(sample_counts[root] == sample_set_sizes):
            raise ValueError("Each tree must have a single root subtending all samples")

        position = sequence_length
        if b < num_edges: position = min(position, remove_position[b])
        if a < num_edges: position = min(position, insert_position[a])

    return


@numba.njit(_f4w(_i1r, _i1r, _f1r, _f1r, _i1r, _i1r, _f, _i1r, _i, _i1r, _i2r, _f1r, _b, _b))
def _trio_coalescence_counts(
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    insert_index,
    remove_index,
    sequence_length,
    sample_set_map, 
    num_bins,
    node_bin_map,
    indexes, 
    windows, 
    span_normalise, 
    trio_normalise,
):
    """
    Low-level trio counting. Assumes trees are binary; if this is not the case
    then the total number of first coalescence events in each tree will not
    sum to the correct number.
    """

    def _comb(n, k):
        return round(exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)))
    
    def _pair_coalescences(i, j, update, outgr):
        pairs = update[i] * outgr[j]
        if i != j: 
            pairs += update[j] * outgr[i]
        return pairs
    
    def _trio_first_coalescences(i, j, k, m, update, outgr, above, pairs):
        trios = update[i] * outgr[j] * (update[k] + above[k]) - pairs[m] * update[k]
        if i != j: 
            trios += update[j] * outgr[i] * (update[k] + above[k])
        return trios
    
    def _trio_second_coalescences(i, j, k, update, outgr, below):
        trios = update[k] * outgr[i] * outgr[j]
        trios += outgr[k] * (update[j] * below[i] + update[i] * below[j] - update[i] * update[j])
        if i == j:
            trios = (trios - outgr[k] * update[i] - update[k] * outgr[i]) / 2
        return trios

    assert edges_child.size == edges_parent.size == edges_left.size == edges_right.size
    assert edges_child.size == insert_index.size == remove_index.size
    assert sample_set_map.size <= node_bin_map.size
    assert indexes.shape[1] == 3
    assert np.all(np.diff(windows) > 0) and np.all(windows >= 0)
    assert windows[0] == 0.0 and windows[-1] == sequence_length

    num_windows = windows.size - 1
    num_sample_sets = sample_set_map.max() + 1
    num_indexes = indexes.shape[0]
    num_nodes = node_bin_map.size
    num_edges = edges_parent.size

    insert_position = edges_left[insert_index]
    remove_position = edges_right[remove_index]
    samples = np.flatnonzero(sample_set_map != tskit.NULL)

    visited = np.full(num_nodes, False)
    nodes_parent = np.full(num_nodes, tskit.NULL)
    sample_counts = np.zeros((num_nodes, num_sample_sets))
    pair_counts = np.zeros((num_nodes, num_indexes))
    coalescing_trios = np.zeros((num_bins, num_indexes, 2))
    sample_set_sizes = np.zeros(num_sample_sets, dtype=np.int32)
    total_trios = np.zeros(num_indexes)
    output = np.zeros((num_windows, num_bins, num_indexes, 2))

    for n, i in enumerate(sample_set_map):  # initial state
        if i == tskit.NULL: continue
        sample_counts[n, i] = 1
        sample_set_sizes[i] += 1

    for m, index_set in enumerate(indexes):
        trio_size = np.bincount(index_set, minlength=num_sample_sets).astype(np.float64)
        for i, n in enumerate(sample_set_sizes):
            trio_size[i] = _comb(n, trio_size[i])
        total_trios[m] = np.prod(trio_size)

    position = 0.0
    w, a, b = 0, 0, 0
    while position < sequence_length:

        #remainder = sequence_length - position
        remainder = 1 - position / sequence_length

        while b < num_edges and remove_position[b] == position: # edges out
            e = remove_index[b]
            p = edges_parent[e]
            c = edges_child[e]
            nodes_parent[c] = tskit.NULL
            update = sample_counts[c]
            while p != tskit.NULL:  # downdate trios
                u = node_bin_map[p]
                if u != tskit.NULL:
                    outgr = sample_counts[p] - sample_counts[c]
                    above = sample_set_sizes - sample_counts[p]
                    below = sample_counts[c]
                    pairs = pair_counts[p]
                    for m, index_set in enumerate(indexes):
                        i, j, k = index_set
                        coalescing_trios[u, m, 0] -= \
                            remainder * _trio_first_coalescences(i, j, k, m, update, outgr, above, pairs)
                        coalescing_trios[u, m, 1] -= \
                            remainder * _trio_second_coalescences(i, j, k, update, outgr, below)
                c, p = p, nodes_parent[p]
            c, p = edges_child[e], edges_parent[e]
            while p != tskit.NULL:  # downdate pairs
                outgr = sample_counts[p] - sample_counts[c]
                for m, index_set in enumerate(indexes):
                    i, j, k = index_set
                    pair_counts[p, m] -= _pair_coalescences(i, j, update, outgr)
                c, p = p, nodes_parent[p]
            c, p = edges_child[e], edges_parent[e]
            while p != tskit.NULL:  # downdate singles
                sample_counts[p] -= update
                p = nodes_parent[p]
            b += 1

        while a < num_edges and insert_position[a] == position: # edges in
            e = insert_index[a]
            p = edges_parent[e]
            c = edges_child[e]
            nodes_parent[c] = p
            update = sample_counts[c]
            while p != tskit.NULL:  # update singles
                sample_counts[p] += update
                p = nodes_parent[p]
            c, p = edges_child[e], edges_parent[e]
            while p != tskit.NULL:  # update pairs
                outgr = sample_counts[p] - sample_counts[c]
                for m, index_set in enumerate(indexes):
                    i, j, k = index_set
                    pair_counts[p, m] += _pair_coalescences(i, j, update, outgr)
                c, p = p, nodes_parent[p]
            c, p = edges_child[e], edges_parent[e]
            while p != tskit.NULL:  # update trios
                u = node_bin_map[p]
                if u != tskit.NULL:
                    outgr = sample_counts[p] - sample_counts[c]
                    above = sample_set_sizes - sample_counts[p]
                    below = sample_counts[c]
                    pairs = pair_counts[p]
                    for m, index_set in enumerate(indexes):
                        i, j, k = index_set
                        coalescing_trios[u, m, 0] += \
                            remainder * _trio_first_coalescences(i, j, k, m, update, outgr, above, pairs)
                        coalescing_trios[u, m, 1] += \
                            remainder * _trio_second_coalescences(i, j, k, update, outgr, below)
                c, p = p, nodes_parent[p]
            a += 1

        position = sequence_length
        if b < num_edges: position = min(position, remove_position[b])
        if a < num_edges: position = min(position, insert_position[a])

        while w < num_windows and windows[w + 1] <= position:  # flush window
            #remainder = (sequence_length - windows[w + 1]) / 2
            remainder = (1 - windows[w + 1] / sequence_length) / 2
            output[w] = coalescing_trios[:]
            coalescing_trios[:] = 0.0
            for c in samples:  # split current tree across windows
                p = nodes_parent[c]
                while not visited[c] and p != tskit.NULL:
                    u = node_bin_map[p]
                    if u != tskit.NULL:
                        outgr = sample_counts[p] - sample_counts[c]
                        above = sample_set_sizes - sample_counts[p]
                        below = sample_counts[c]
                        pairs = pair_counts[p]
                        for m, index_set in enumerate(indexes):
                            i, j, k = index_set
                            first = above[k] * pairs[m]
                            second = outgr[k] * below[i] * below[j] + outgr[i] * outgr[j] * below[k]
                            if i == j:
                                second = (second - outgr[k] * below[i] - outgr[i] * below[k]) / 2
                            output[w, u, m, 0] -= remainder * first
                            output[w, u, m, 1] -= remainder * second
                            coalescing_trios[u, m, 0] += remainder * first
                            coalescing_trios[u, m, 1] += remainder * second
                    visited[c] = True
                    p, c = nodes_parent[p], p
            for c in samples:
                p = nodes_parent[c]
                while visited[c] and p != tskit.NULL:
                    visited[c] = False
                    p, c = nodes_parent[p], p
            if span_normalise:
                #output[w] /= (windows[w + 1] - windows[w])
                output[w] *= sequence_length / (windows[w + 1] - windows[w])
            if trio_normalise:
                output[w] /= total_trios[np.newaxis, :, np.newaxis]
            w += 1

    return output


def trio_coalescence_counts(
    ts, 
    sample_sets, 
    indexes, 
    windows=None, 
    time_windows="nodes", 
    check_binary=True, 
    span_normalise=False, 
    trio_normalise=False,
):

    if windows is None:
        windows = np.array([0.0, ts.sequence_length])
        drop_left_dimension = True

    if isinstance(time_windows, str) and time_windows == "nodes":
        node_bin_map = np.arange(ts.num_nodes, dtype=np.int32)
        num_bins = ts.num_nodes
    else:
        if not (isinstance(time_windows, np.ndarray) and time_windows.size > 1):
            raise ValueError("Time windows must be an array of breakpoints")
        if not np.all(np.diff(time_windows) > 0):
            raise ValueError("Time windows must be strictly increasing")
        if ts.time_units == tskit.TIME_UNITS_UNCALIBRATED:
            raise ValueError("Time windows require calibrated node times")
        node_bin_map = np.searchsorted(time_windows, ts.nodes_time, side="right") - 1
        #nodes_oob = np.logical_or(node_bin_map < 0, node_bin_map >= time_windows.size - 1)
        #node_bin_map[nodes_oob] = tskit.NULL
        node_bin_map[node_bin_map == time_windows.size - 1] = tskit.NULL
        node_bin_map = node_bin_map.astype(np.int32)
        num_bins = time_windows.size - 1

    sample_set_map = np.full(ts.num_samples, tskit.NULL, dtype=np.int32)
    for i, s in enumerate(sample_sets):
        sample_set_map[s] = i
    indexes = indexes.astype(np.int32)

    if check_binary:
        _assert_complete_binary_trees(
            ts.edges_parent, 
            ts.edges_child, 
            ts.edges_left, 
            ts.edges_right, 
            ts.indexes_edge_insertion_order,
            ts.indexes_edge_removal_order,
            ts.sequence_length,
            sample_set_map,
        )

    return _trio_coalescence_counts(
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
        ts.sequence_length,
        sample_set_map, 
        num_bins,
        node_bin_map,
        indexes, 
        windows, 
        span_normalise, 
        trio_normalise,
    )


class TrioCoalescenceRates:
    """
    Class that calculates rates of first and second trio coalescences in a tree
    sequence and optionally block-bootstraps these counts
    """

    def __init__ (self, ts, sample_sets, time_breaks, sample_set_names=None, bootstrap_blocks=None, check_binary=True):

        self.prefix = "[TrioCoalescenceRates] "

        if isinstance(ts, str):
            ts = tskit.load(ts)
        else:
            assert isinstance(ts, tskit.TreeSequence), "Input is not tree sequence"
        self.sequence_length = ts.sequence_length

        if bootstrap_blocks is None:
            bootstrap_blocks = np.array([0, ts.sequence_length])
        self.block_span = np.diff(bootstrap_blocks) / self.sequence_length
        self.num_blocks = bootstrap_blocks.size - 1

        self.sample_sets = sample_sets
        self.sample_set_sizes = np.array([len(s) for s in sample_sets])
        self.num_sample_sets = self.sample_set_sizes.size
        if sample_set_names is None:
            sample_set_names = [str(i) for i in np.arange(self.num_sample_sets)]
        self.sample_set_names = sample_set_names

        self.time_breaks = time_breaks
        self.num_time_windows = self.time_breaks.size - 1

        # calculate weights per block
        # num_weights = 2 * P * (combn(P, 2) + P)
        self.num_weights = 2*int(self.num_sample_sets*self.num_sample_sets*(self.num_sample_sets+1)/2)

        trio_indexes = []
        total_trios = []
        for a, b in combn_replace(range(self.num_sample_sets), 2):
            for c in range(self.num_sample_sets):
                trio_indexes.append([a, b, c])
                trio_size = np.bincount([a, b, c], minlength=self.num_sample_sets)
                for i, n in enumerate(self.sample_set_sizes):
                    trio_size[i] = comb(n, trio_size[i])
                total_trios.append(np.prod(trio_size))
        trio_indexes = np.array(trio_indexes)
        total_trios = np.array(total_trios)[:, np.newaxis] * self.block_span[np.newaxis, :]
        trio_counts = trio_coalescence_counts(
            ts, 
            sample_sets, 
            trio_indexes, 
            windows=bootstrap_blocks, 
            time_windows=time_breaks, 
            check_binary=check_binary,
            span_normalise=False,
            trio_normalise=False,
        )
        trio_counts = np.concatenate([trio_counts[..., 0], trio_counts[..., 1]], axis=2)
        total_trios = np.concatenate([total_trios, total_trios], axis=0)
        self.y = trio_counts.transpose(2, 1, 0)
        self.n = total_trios[:, np.newaxis, :] - self._share_denominator_across_initial_states(self.y.cumsum(axis=1))
        self.total_trios = total_trios #DEBUG

        #self.n = self._share_denominator_across_initial_states(block_n)
        #self.y = block_y

    def _share_denominator_across_initial_states (self, n):
        """
        Sum denominator of rate for emissions with the same initial state, then
        map back to original array
        """
        denominator_labels = self.configurations()
        unique_denominator_labels = np.sort(np.unique(denominator_labels))
        denominator_labels_idx = np.searchsorted(unique_denominator_labels, denominator_labels, side='right') - 1
        n_pool = np.zeros((len(unique_denominator_labels), n.shape[1], n.shape[2]))
        for i, j in enumerate(denominator_labels_idx):
            n_pool[j, :, :] += n[i, :, :]
        n_share = np.zeros(n.shape)
        for i, j in enumerate(denominator_labels_idx):
            n_share[i, :, :] = n_pool[j, :, :]
        return n_share

    def labels (self):
        trio_labels = []
        for t in ['t1::', 't2::']:
            for a, b in combn_replace(range(self.num_sample_sets), 2):
                pair_label = t + '((' + self.sample_set_names[a] + ',' + self.sample_set_names[b] + '),'
                for c in range(self.num_sample_sets):
                    trio_labels += [pair_label + self.sample_set_names[c] + ")"]
        return trio_labels

    def configurations (self):
        configurations = []
        for t in ['t1::', 't2::']:
            for a, b in combn_replace(range(self.num_sample_sets), 2):
                pair = [a, b]
                for c in range(self.num_sample_sets):
                    u = np.sort(pair + [c])
                    configurations.append(t + "{" + 
                            self.sample_set_names[u[0]] + "," + 
                            self.sample_set_names[u[1]] + "," + 
                            self.sample_set_names[u[2]] + "}")
        return configurations

    def epochs (self):
        return ["[" + str(self.time_breaks[i]) + "," + str(self.time_breaks[i+1]) + ")" 
                for i in range(len(self.time_breaks)-1)]

    def join (self, rhs):
        assert isinstance (rhs, TrioCoalescenceRates)
        assert all([i == j for i, j in zip(self.epochs(), rhs.epochs())])
        assert all([i == j for i, j in zip(self.labels(), rhs.labels())])
        total_length = self.sequence_length + rhs.sequence_length
        lhs_weight = self.sequence_length / total_length
        rhs_weight = rhs.sequence_length / total_length
        self.y = np.concatenate((self.y * lhs_weight, rhs.y * rhs_weight), axis=2)
        self.n = np.concatenate((self.n * lhs_weight, rhs.n * rhs_weight), axis=2)
        self.block_span = np.concatenate((self.block_span * lhs_weight, rhs.block_span * rhs_weight))
        self.sequence_length += rhs.sequence_length

    # TODO: these can be streamlined with fancy indexing
    def numerator (self, block_weights=None):
        if block_weights is None:
            block_weights = self.block_span
            block_weights /= np.sum(block_weights)
        y = np.zeros((self.num_weights, self.num_time_windows))
        for block, weight in enumerate(block_weights):
            y[:, :] += weight * self.y[:, :, block]
        return y

    def denominator (self, block_weights=None):
        if block_weights is None:
            block_weights = self.block_span
            block_weights /= np.sum(block_weights)
        n = np.zeros((self.num_weights, self.num_time_windows))
        for block, weight in enumerate(block_weights):
            n[:, :] += weight * self.n[:, :, block]
        return n

    def rates (self, block_weights=None):
        rates = self.numerator(block_weights) / self.denominator(block_weights)
        for i in range(self.num_time_windows):
            rates[:, i] /= (self.time_breaks[i + 1] - self.time_breaks[i])
        return rates
    #/TODO

    def block_bootstrap (self, num_replicates, random_seed=None):
        num_replicates = int(num_replicates)
        assert num_replicates > 0

        if random_seed is not None:
            random_seed = int(random_seed)
            assert random_seed >= 0

        num_blocks = self.block_span.size
        rng = np.random.default_rng(random_seed)
        for rep in range(num_replicates):
            # TODO can't remember the derivation here; it seems unnecessarily complicated
            # (and maybe wrong)
            block_multiplier = rng.multinomial(num_blocks, np.full(num_blocks, 1.0 / num_blocks))
            block_weights = self.block_span * block_multiplier
            block_weights /= np.sum(block_weights)
            rates = self.rates(block_weights / self.block_span)
            yield rates

    def std_dev (self, num_replicates, random_seed=None):
        num_replicates = int(num_replicates)
        assert num_replicates > 1
        n = 0.0
        mean = np.zeros((self.num_weights, self.num_time_windows))
        sumsq = np.zeros((self.num_weights, self.num_time_windows))
        for x in self.block_bootstrap(num_replicates, random_seed):
            n += 1.0
            delta = x - mean
            mean += delta / n
            delta *= x - mean
            sumsq += delta
        return np.sqrt(sumsq / (n - 1.0))

    def bootstrapped_rates (self, num_replicates, random_seed=None):
        num_replicates = int(num_replicates)
        assert num_replicates > 0
        rates = np.zeros((self.num_weights, self.num_time_windows, num_replicates))
        for i, x in enumerate(self.block_bootstrap(num_replicates, random_seed)):
            rates[:,:,i] = x
        return rates

