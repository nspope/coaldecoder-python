//
// MIT License
//
// Copyright (c) 2021-2024 Nathaniel S. Pope
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <vector>
#include <string>
#include <pybind11/stl.h>

#include "array_conversion.hpp"
#include "matrix_exponential.hpp"
#include "transition_rate_matrix.hpp"
#include "admixture_matrix.hpp"

struct CoalescentEpoch
{
  const std::string prefix = "[CoalescentEpoch] ";
  const bool check_valid = true;
  const bool use_marginal_statistics = true; //see note in cnstr

  //TODO this should depend on dimension of state vectors?
  const double left_stochastic_tol = 1e-12; 

  const TrioAdmixtureProportions admix_matrix;
  const TrioTransitionRates rate_matrix;

  const arma::umat emission_mapping;
  const arma::umat state_mapping;
  const arma::umat initial_mapping;

  const arma::mat A; //admixture proportions
  const arma::mat M; //demographic parameters
  const double t; //duration of epoch

  arma::vec y_hat;
  arma::vec residual;
  arma::vec gradient;
  arma::vec uncoalesced; //uncoalesced lineages at start
  arma::mat states; //states immediately after admixture
  double loglikelihood;

  CoalescentEpoch (
    arma::mat& _states,
    const arma::mat& _migration_matrix,
    const arma::mat& _admixture_matrix,
    const double _time_step,
    // mappings
    const arma::umat& _emission_mapping,
    const arma::umat& _state_mapping,
    const arma::umat& _initial_mapping,
    const bool _check_valid = true
  ) 
    : check_valid (_check_valid)
    , admix_matrix (_admixture_matrix, _check_valid)
    , rate_matrix (_migration_matrix, _check_valid)
    , emission_mapping (_emission_mapping)
    , state_mapping (_state_mapping)
    , initial_mapping (_initial_mapping)
    , A (_admixture_matrix)
    , M (_migration_matrix)
    , t (_time_step)
  {
    /* 
     *  Update states and calculate likelihood, gradient of emissions within epoch.
     */

    if (_states.n_rows != arma::accu(rate_matrix.S) || 
        _states.n_cols != initial_mapping.n_cols)
    {
      throw std::invalid_argument(prefix + "State probability vectors have the wrong dimension");
    }

    if (check_valid)
    {
      if (t <= 0.0)
      {
        throw std::invalid_argument(prefix + "Epoch duration must be positive");
      }

      if (arma::any(arma::abs(arma::sum(_states, 0) - 1.0) > left_stochastic_tol))
      {
        throw std::invalid_argument(prefix + "State probability vector does not sum to one");
      }
      if (arma::any(arma::vectorise(_states) < 0.0))
      {
        throw std::invalid_argument(prefix + "Negative state probabilities");
      }
    }

    unsigned num_emission = emission_mapping.n_cols;
    unsigned num_initial = initial_mapping.n_cols;

    // admixture
    states = _states;
    _states = admix_matrix.X * _states;

    // transition probabilities
    SparseMatrixExponentialMultiply transition (arma::trans(rate_matrix.X), _states, t);
    _states = transition.result;

    // fitted rates
    arma::vec transitory_start = arma::ones(num_initial);
    arma::vec coalesced_start = arma::zeros(num_emission);
    arma::vec coalesced_end = arma::zeros(num_emission);
    for (unsigned i = 0; i < state_mapping.n_cols; ++i)
    {
      arma::uword col = state_mapping.at(0, i);
      arma::uword row = state_mapping.at(1, i);
      arma::uword j = state_mapping.at(2, i);
      transitory_start.at(col) += -states.at(row, col);
      coalesced_start.at(j) += states.at(row, col);
      coalesced_end.at(j) += _states.at(row, col);
    }

    arma::vec subtransitory_start = arma::zeros(num_initial);
    uncoalesced = arma::zeros(num_emission);
    int offset = num_emission / 2;
    for (int j = 0; j < num_emission; ++j)
    {
      arma::uword col = emission_mapping.at(0, j);
      arma::uword lin = emission_mapping.at(1, j);

      // sum mass over transitory 2-lineage states in a column
      subtransitory_start[col] += lin == 2 ? coalesced_start[j] : 0.0;

      // two-lineage states can transition to single-lineage states
      uncoalesced[j] = lin == 2 ? transitory_start[col] : transitory_start[col] + subtransitory_start[col]; 

      // if the input probabilities are marginal, add 1-lineage to 2-lineage.
      // this is because the state vector contains:
      //    p(t1 < x & t2 > x) ===> 2-lineage statistics
      //    p(t1 < x & t2 < x) ===> 1-lineage statistics
      // and we want:
      //    p(t1 < x) = p(t1 < x & t2 < x) + p(t1 < x & t2 > x)
      // which will ensure that the statistics are strictly positive.
      if (lin == 1 && use_marginal_statistics)
      {
        coalesced_start[j - offset] += coalesced_start[j];
        coalesced_end[j - offset] += coalesced_end[j];
      }
    }

    y_hat = (coalesced_end - coalesced_start) / uncoalesced / t;
  }

  arma::cube reverse_differentiate (arma::mat& _states, const arma::mat& _gradient)
  {
    if (_states.n_rows != arma::accu(rate_matrix.S) || _states.n_cols != initial_mapping.n_cols)
    {
      throw std::invalid_argument(prefix + "State gradient vectors have the wrong dimension");
    }

    if (_gradient.n_rows != y_hat.n_rows)
    {
      throw std::invalid_argument(prefix + "Rate gradient vectors have the wrong dimension");
    }

    unsigned num_emission = emission_mapping.n_cols;
    unsigned num_initial = initial_mapping.n_cols;

    // Gradient wrt fitted rates
    arma::vec d_y_hat = _gradient * 1.0 / t;

    // Gradient wrt state vectors
    arma::mat d_states = arma::zeros(arma::size(states));
    arma::vec d_uncoalesced = -t * y_hat % d_y_hat/uncoalesced;
    arma::vec d_coalesced_start = -d_y_hat/uncoalesced;
    arma::vec d_coalesced_end = d_y_hat/uncoalesced;
    arma::vec d_transitory_start = arma::zeros(num_initial);
    arma::vec d_subtransitory_start = arma::zeros(num_initial);
    int offset = num_emission / 2;
    for (int j = num_emission - 1; j >= 0; --j)
    {
      arma::uword col = emission_mapping.at(0, j);
      arma::uword lin = emission_mapping.at(1, j);
      if (lin == 1 && use_marginal_statistics)
      {
        d_coalesced_start[j] += d_coalesced_start[j - offset];
        d_coalesced_end[j] += d_coalesced_end[j - offset];
      }
      d_transitory_start[col] += d_uncoalesced[j];
      if (lin == 1) 
      {
        d_subtransitory_start[col] += d_uncoalesced[j];
      } else if (lin == 2) {
        d_coalesced_start[j] += d_subtransitory_start[col];
      }
    }
    for (int i = 0; i < state_mapping.n_cols; ++i)
    {
      arma::uword col = state_mapping.at(0, i);
      arma::uword row = state_mapping.at(1, i);
      arma::uword j = state_mapping.at(2, i);
      d_states.at(row, col) += -d_transitory_start.at(col);
      d_states.at(row, col) += d_coalesced_start.at(j);
      _states.at(row, col) += d_coalesced_end.at(j);
    }

    // Gradient wrt starting state vectors, trio rate matrix
    // (this adds gradient contribution to existing d_states)
    SparseMatrixExponentialMultiply transition (arma::trans(rate_matrix.X), admix_matrix.X * states, t);
    arma::mat _updated_states;
    arma::sp_mat d_rate_matrix = transition.reverse_differentiate(_updated_states, _states); 
    _states = _updated_states;

    // Gradient wrt pre-admixture state vectors, trio admixture matrix
    arma::sp_mat d_admix_matrix (arma::size(admix_matrix.X));
    for (arma::sp_mat::const_iterator it = admix_matrix.X.begin(); it != admix_matrix.X.end(); ++it)
    {
      double val = arma::dot(_states.row(it.row()), states.row(it.col()));
      d_admix_matrix.at(it.row(), it.col()) = val;
    }
    _states = arma::trans(admix_matrix.X) * _states;
    _states = _states + d_states;

    // Gradient wrt demographic parameter, admixture matrices
    arma::cube d_parameters (rate_matrix.P, rate_matrix.P, 2);
    d_parameters.slice(0) = rate_matrix.reverse_differentiate(arma::trans(d_rate_matrix));
    d_parameters.slice(1) = admix_matrix.reverse_differentiate(d_admix_matrix);

    return d_parameters;
  }
};


struct TrioCoalescenceRateModel
{
    const std::string prefix = "[TrioCoalescenceRateModel] ";
    const bool check_valid = true;
  
    const unsigned num_populations;
    const TrioTransitionRates rate_matrix;
    const arma::umat emission_mapping;
    const arma::umat state_mapping;
    const arma::umat initial_mapping;
    std::vector<CoalescentEpoch> epochs;
  
    TrioCoalescenceRateModel (const unsigned num_populations)
        : num_populations (num_populations)
        , rate_matrix (arma::ones(num_populations, num_populations))
        , emission_mapping (rate_matrix.emission_to_initial())
        , state_mapping (rate_matrix.states_to_emission())
        , initial_mapping (rate_matrix.initial_to_states())
    {}
  
    std::vector<std::string> labels (std::vector<std::string> population_names) const
    {
        if (population_names.size() != num_populations) {
            throw std::invalid_argument(prefix + "Population name vector has the wrong dimension");
        }
  
        return rate_matrix.emission_states(population_names);
    }
  
    arma::mat forward (const arma::cube& demographic_parameters, const arma::cube& admixture_coefficients, const arma::vec& time_step)
    {
        if (demographic_parameters.n_slices != time_step.n_rows)
        {
            throw std::invalid_argument(prefix + "Demographic parameter array has the wrong dimensions");
        }
  
        if (admixture_coefficients.n_slices != time_step.n_rows)
        {
            throw std::invalid_argument(prefix + "Admixture parameter array has the wrong dimensions");
        }
  
        epochs.clear();
        arma::mat state = initial_state_vectors();
        arma::mat expected_rates (emission_mapping.n_cols, time_step.n_rows);
        for (unsigned i = 0; i < time_step.n_rows; ++i)
        {
            epochs.emplace_back(
                state, demographic_parameters.slice(i), admixture_coefficients.slice(i), time_step.at(i), 
                emission_mapping, state_mapping, initial_mapping, check_valid
            );
            expected_rates.col(i) = epochs[i].y_hat;
        }
  
        return expected_rates;
    }
  
    std::tuple<arma::cube, arma::cube, arma::vec> backward (const arma::mat& d_expected_rates)
    {
        if (d_expected_rates.n_cols != epochs.size())
        {
            throw std::invalid_argument(prefix + "Gradient dimension does not match internal state size");
        }
  
        arma::cube d_demographic_parameters (num_populations, num_populations, epochs.size());
        arma::cube d_admixture_coefficients (num_populations, num_populations, epochs.size());
  
        arma::mat d_states = arma::zeros(arma::size(epochs.back().states));
        for (int i = epochs.size() - 1; i >= 0; --i)
        {
            arma::cube gradient = epochs[i].reverse_differentiate(d_states, d_expected_rates.col(i));
            d_demographic_parameters.slice(i) = gradient.slice(0);
            d_admixture_coefficients.slice(i) = gradient.slice(1);
        }
        arma::vec d_time_step = arma::zeros(epochs.size()); // placeholder
  
        return std::make_tuple(d_demographic_parameters, d_admixture_coefficients, d_time_step);
    }
  
    arma::mat initial_state_vectors (void)
    {
        arma::mat states = arma::zeros<arma::mat>(
            arma::accu(rate_matrix.S), 
            initial_mapping.n_cols
        );
        for (unsigned i = 0; i < initial_mapping.n_cols; ++i)
        {
            arma::uword col = initial_mapping.at(0, i);
            arma::uword row = initial_mapping.at(1, i);
            states.at(row, col) = 1.0;
        }
        return states;
    }
  
};


struct PairCoalescenceRateModel
{
    /* This workaround is a bit hacky. It would be better to use a
     * pair-specific rate matrix to save computation. */

    TrioCoalescenceRateModel model;
    arma::uvec trio_to_pair;
    arma::ivec pair_to_trio;

    PairCoalescenceRateModel (const unsigned num_populations)
        : model (num_populations + 1)
    {
        /* map trios to pairs and vis versa */
        std::vector<std::string> names (num_populations, "+");
        names.emplace_back("-");
        auto trio_names = model.labels(names);
        std::string pair = "t1::((+,+),-)";
        std::vector<arma::uword> trio_indices;
        std::vector<arma::sword> pair_indices (trio_names.size(), -1);
        for (unsigned i = 0; i < trio_names.size(); ++i) {
            if (trio_names[i] == pair) {
                pair_indices[i] = trio_indices.size();
                trio_indices.push_back(i);
            }
        }
        trio_to_pair = arma::uvec(trio_indices);
        pair_to_trio = arma::ivec(pair_indices);
    }

    arma::mat trio_to_pair_rates(const arma::mat& inp) {
        arma::mat out = arma::zeros(trio_to_pair.n_rows, inp.n_cols);
        for (unsigned i = 0; i < trio_to_pair.n_rows; ++i) {
            int j = trio_to_pair[i];
            out.row(i) = inp.row(j);
        }
        return out;
    }

    arma::mat pair_to_trio_rates(const arma::mat& inp) {
        arma::mat out = arma::zeros(pair_to_trio.n_rows, inp.n_cols);
        for (unsigned i = 0; i < pair_to_trio.n_rows; ++i) {
            int j = pair_to_trio[i];
            if (j != -1) out.row(i) = inp.row(j);
        }
        return out;
    }

    arma::cube insert_dummy_population(const arma::cube& inp, double fill) {
        arma::cube out = arma::zeros(inp.n_rows + 1, inp.n_cols + 1, inp.n_slices);
        for (unsigned k = 0; k < inp.n_slices; ++k) {
            for (unsigned i = 0; i < inp.n_rows; ++i) {
                for (unsigned j = 0; j < inp.n_cols; ++j) {
                  out.at(i, j, k) = inp.at(i, j, k);
                }
            }
            out.at(out.n_rows - 1, out.n_cols - 1, k) = fill;
        }
        return out;
    }

    arma::cube remove_dummy_population(const arma::cube& inp) {
        arma::cube out = arma::zeros(inp.n_rows - 1, inp.n_cols - 1, inp.n_slices);
        for (unsigned k = 0; k < out.n_slices; ++k) {
            for (unsigned i = 0; i < out.n_rows; ++i) {
                for (unsigned j = 0; j < out.n_cols; ++j) {
                    out.at(i, j, k) = inp.at(i, j, k);
                }
            }
        }
        return out;
    }

    std::vector<std::string> labels (std::vector<std::string> population_names) {
        population_names.emplace_back("-");
        std::vector<std::string> trio_labels = model.labels(population_names);
        std::vector<std::string> pair_labels;
        for (auto i : trio_to_pair) { pair_labels.push_back(trio_labels[i]); }
        return pair_labels;
    }

    arma::mat forward(const arma::cube& demographic_parameters, const arma::cube& admixture_coefficients, const arma::vec& time_step) {
        auto trio_rates = model.forward(
            insert_dummy_population(demographic_parameters, arma::datum::inf),
            insert_dummy_population(admixture_coefficients, 1),
            time_step
        );
        return trio_to_pair_rates(trio_rates);
    }

    std::tuple<arma::cube, arma::cube, arma::vec> backward(const arma::mat& d_expected_rates) {
        auto [d_demographic_parameters, d_admixture_coefficients, d_time_step] = 
            model.backward(pair_to_trio_rates(d_expected_rates));
        return std::make_tuple(
            remove_dummy_population(d_demographic_parameters),
            remove_dummy_population(d_admixture_coefficients),
            d_time_step
        );
    }
};


/* ---------------- API ---------------- */

PYBIND11_MODULE(_coaldecoder, m) {
  py::class_<TrioCoalescenceRateModel>(m, "TrioCoalescenceRateModel")
    .def(py::init<unsigned>())
    .def("labels", &TrioCoalescenceRateModel::labels)
    .def("forward", 
        [] (TrioCoalescenceRateModel& self, pyarr<double> demographic_parameters, pyarr<double> admixture_coefficients, pyarr<double> time_step) 
        {
            auto expected_rates = self.forward(
                to_cube<double>(demographic_parameters),
                to_cube<double>(admixture_coefficients),
                to_vec<double>(time_step)
            );
            return from_mat<double>(expected_rates);
        }
    )
    .def("backward", 
        [] (TrioCoalescenceRateModel& self, pyarr<double> d_expected_rates)
        {
            auto [d_demographic_parameters, d_admixture_coefficients, d_time_step] = self.backward(to_mat<double>(d_expected_rates));
            return py::make_tuple(
                from_cube<double>(d_demographic_parameters),
                from_cube<double>(d_admixture_coefficients),
                py::none()
            );
        }
    );
  py::class_<PairCoalescenceRateModel>(m, "PairCoalescenceRateModel")
    .def(py::init<unsigned>())
    .def("labels", &PairCoalescenceRateModel::labels)
    .def("forward", 
        [] (PairCoalescenceRateModel& self, pyarr<double> demographic_parameters, pyarr<double> admixture_coefficients, pyarr<double> time_step) 
        {
            auto expected_rates = self.forward(
                to_cube<double>(demographic_parameters),
                to_cube<double>(admixture_coefficients),
                to_vec<double>(time_step)
            );
            return from_mat<double>(expected_rates);
        }
    )
    .def("backward", 
        [] (PairCoalescenceRateModel& self, pyarr<double> d_expected_rates)
        {
            auto [d_demographic_parameters, d_admixture_coefficients, d_time_step] = self.backward(to_mat<double>(d_expected_rates));
            return py::make_tuple(
                from_cube<double>(d_demographic_parameters),
                from_cube<double>(d_admixture_coefficients),
                py::none()
            );
        }
    );
}

