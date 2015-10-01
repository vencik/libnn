#ifndef libnn__ml__backpropagation_hxx
#define libnn__ml__backpropagation_hxx

/**
 *  Backward propagation of errors for neural networks
 *
 *  See https://en.wikipedia.org/wiki/Backpropagation
 *
 *  \date    2015/09/06
 *  \author  Vaclav Krpec  <vencik@razdva.cz>
 *
 *
 *  LEGAL NOTICE
 *
 *  Copyright (c) 2015, Vaclav Krpec
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 *  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "libnn/topo/nn.hxx"
#include "libnn/ml/computation.hxx"

#include <vector>
#include <list>
#include <stdexcept>
#include <algorithm>
#include <cassert>


namespace libnn {
namespace ml {

/**
 *  \brief  Backpropagation algorithm
 *
 *  Implements the backpropagation algorithm.
 *  Supports batch mode (i.e. update after processing of a batch of training
 *  patterns).
 *  Note that the \c Act_fn functor must provide method
 *   d(const Base_t & x) const
 *
 *  that computes 1st derivation of the activation function in \c x.
 *
 *  \tparam  Base_t   Base numeric type
 *  \tparam  Act_fn   Activation function
 */
template <typename Base_t, class Act_fn>
class backpropagation {
    private:

    /** Neural network type */
    typedef topo::nn<Base_t, Act_fn> nn_t;

    /**
     *  \brief  Neural network forward synapses mapping
     *
     *  For each neuron, this list contains all synapsis (and their neurons
     *  indices) that connect to the neuron.
     */
    typedef std::vector<
        std::list<
            std::pair<const typename nn_t::neuron::dendrite &, size_t> > >
        forward_map_t;

    /**
     *  \brief  Forward phase result (for a neuron)
     *
     *  Activation function value and its argument.
     */
    struct forward_result {
        Base_t net;      /**< Sum of weighed inputs                 */
        Base_t phi_net;  /**< Activation function value of phi(net) */

        /** Default constructor */
        forward_result(): net(0), phi_net(0) {}

    };  // end of struct forward_result

    /**
     *  \brief  Forward phase
     *
     *  Computation of neurons' activation function and its argument.
     */
    class forward: public computation<Base_t, Act_fn, forward_result> {
        private:

        /**< Ancestor type */
        typedef computation<Base_t, Act_fn, forward_result> computation_t;

        /**< Neural network type */
        typedef typename computation_t::nn_t nn_t;

        /**
         *  \brief  Compute forward result for a neuron
         *
         *  \param  n  Neuron
         *
         *  \return Activation function value and its argument
         */
        forward_result f(const typename nn_t::neuron & n) {
            forward_result res;

            n.for_each_dendrite(
            [&res, this](const typename nn_t::neuron::dendrite & dend) {
                res.net += dend.weight * this->fx(dend.source.index()).phi_net;
            });

            res.phi_net = n.act_fn(res.net);

            return res;
        }

        public:

        /**
         *  \brief  Constructor
         *
         *  \param  network  Neural network
         */
        forward(const nn_t & network): computation_t(network) {}

        /**
         *  \brief  Execute the forward phase
         *
         *  \tparam Input  Input container type (iterable)
         *  \param  input  Input
         *
         *  \return Output vector
         */
        template <class Input>
        std::vector<Base_t> operator () (const Input & input) {
            this->reset();  // make sure all is clean

            // Set input layer
            forward_result in_res;

            auto in_iter = input.begin();
            this->network().for_each_input(
            [&in_res, &in_iter, this](const typename nn_t::neuron & n) {
                in_res.phi_net = *(in_iter++);

                this->fx(n.index(), in_res);
            });

            // Compute output layer (and therefore all on paths)
            std::vector<Base_t> output;
            output.reserve(this->network().output_size());

            this->network().for_each_output(
            [&output, this](const typename nn_t::neuron & n) {
                output.push_back(this->fx(n.index()).phi_net);
            });

            return output;
        }

    };  // end of class forward

    /**
     *  \brief  Backward phase result (for a neuron)
     *
     *  Delta (backward error propagation).
     */
    struct backward_result {
        Base_t delta;  /**< Backward error propagation */

        /** Default constructor */
        backward_result(): delta(0) {}

    };  // end of struct backward_result

    /**
     *  \brief  Backward phase
     *
     *  Computation of error propagation.
     */
    class backward: public computation<Base_t, Act_fn, backward_result> {
        private:

        /**< Ancestor type */
        typedef computation<Base_t, Act_fn, backward_result> computation_t;

        /**< Neural network type */
        typedef typename computation_t::nn_t nn_t;

        /**< Neuron type */
        typedef typename nn_t::neuron neuron_t;

        /**< Dendrite type */
        typedef typename neuron_t::dendrite dendrite_t;

        const forward_map_t & m_fmap;     /**< Forward mapping       */
        const forward       & m_forward;  /**< Forward stage results */

        /**
         *  \brief  Compute backward result for non-output neuron
         *
         *  \param  n  Neuron
         *
         *  \return Delta
         */
        backward_result f(const typename nn_t::neuron & n) {
            backward_result res;

            if (nn_t::neuron::OUTPUT == n.type())
                throw std::logic_error(
                    "libnn::ml::backpropagation: "
                    "unexpected output layer neuron for error propagation");

            assert(n.index() < m_fmap.size());

            const auto & fw_neurons = m_fmap[n.index()];
            std::for_each(fw_neurons.begin(), fw_neurons.end(),
            [&res, this](const std::pair<const dendrite_t &, size_t> & dend_n) {
                const dendrite_t & fw_dend    = dend_n.first;
                const size_t       fw_n_index = dend_n.second;

                res.delta += this->fx(fw_n_index).delta * fw_dend.weight;
            });

            res.delta *= n.act_fn().d(m_forward.fx(n.index()).net);

            return res;
        }

        public:

        /**
         *  \brief  Constructor
         *
         *  \param  network  Neural network
         *  \param  fmap     Forward mapping
         *  \param  forvard  Forward stage results
         */
        backward(
            const nn_t          & network,
            const forward_map_t & fmap,
            const forward       & forvard)
        :
            computation_t(network),
            m_fmap(fmap),
            m_forward(forvard)
        {}

        /**
         *  \brief  Execute the backward phase
         *
         *  \param  error  Error
         */
        void operator () (const std::vector<Base_t> & error) {
            this->reset();  // make sure all is clean

            // Set output layer delta
            backward_result out_res;

            auto err_iter = error.begin();
            this->network().for_each_output(
            [&out_res, &err_iter, this](const neuron_t & n) {
                const Base_t dact_net = n.act_fn().d(
                    m_forward.fx(n.index()).net);

                out_res.delta = *(err_iter++) * dact_net;

                this->fx(n.index(), out_res);
            });

            // Compute input layer (and therefore all on paths)
            this->network().for_each_input(
            [this](const neuron_t & n) {
                this->fx(n.index());
            });
        }

    };  // end of class backward

    /** Computation slot */
    struct comp_slot {
        forward  fw;  /**< Forward  stage */
        backward bw;  /**< Backward stage */

        /**
         *  \brief  Constructor
         *
         *  \param  network  Trained neural network
         *  \param  fmap     The network forward map
         */
        comp_slot(
            const nn_t & network,
            const forward_map_t & fmap)
        :
            fw(network),
            bw(network, fmap, fw)
        {}

    };  // end of struct comp_slot

    typedef std::list<comp_slot> slots_t;  /**< Computation slot list */

    nn_t &              m_network;    /**< Trained neural network         */
    const forward_map_t m_fmap;       /**< The neural network forward map */
    slots_t             m_slots;      /**< Computation slots              */

    /**
     *  \brief  Create NN forward synapses mapping
     *
     *  See \ref forward_map_t.
     *
     *  \param  nn  Neural network
     *
     *  \return NN forward synapses mapping
     */
    static forward_map_t create_fmap(const nn_t & nn) {
        forward_map_t fmap(nn.slot_cnt());

        nn.for_each_neuron(
        [&fmap](const typename nn_t::neuron & n) {
            const size_t n_index = n.index();

            n.for_each_dendrite(
            [&fmap, n_index](const typename nn_t::neuron::dendrite & dend) {
                fmap[dend.source.index()].emplace_back(dend, n_index);
            });
        });

        return fmap;
    }

    /**
     *  \brief  Make \c n forward/backward computation slots available
     *
     *  \param  n  Number of slots
     */
    void assert_slots(size_t n) {
        for (size_t i = m_slots.size(); i < n; ++i)
            m_slots.emplace_back(m_network, m_fmap);
    }

    /**
     *  \brief  Backward error propagation: computation
     *
     *  Computes forward error on an \c input and its bacward propagation.
     *
     *  \tparam Input   Input container type (iterable)
     *  \tparam Output  Output container type (iterable)
     *  \param  input   Input
     *  \param  output  Output (desired)
     *  \param  slot    Computation slot
     *
     *  \return Error norm squared
     */
    template <class Input, class Output>
    Base_t compute(
        const Input  & input,
        const Output & output,
        comp_slot    & slot)
    {
        Base_t error_norm2 = 0;

        // Compute forward stage (activation func. and its argument)
        auto error = slot.fw(input);

        // Compute error (actual output minus desired output)
        if (output.size() != error.size())
            throw std::logic_error(
                "libnn::ml::backpropagation: "
                "invalid output target supplied");

        auto out_iter = output.begin();
        std::for_each(error.begin(), error.end(),
        [&error_norm2, &out_iter](Base_t & err) {
            err -= *(out_iter++);

            error_norm2 += err * err;
        });

        // Compute backward stage (delta distribution)
        slot.bw(error);

        return error_norm2;
    }

    /**
     *  \brief  Backward error propagation: network update
     *
     *  Updates the network by previously computed backward error propagation
     *  (parametrisd by the learning factor).
     *
     *  \param  alpha   Learning factor
     *  \param  slot    Computation slot
     */
    void update(
        const Base_t    & alpha,
        const comp_slot & slot)
    {
        m_network.for_each_neuron(
        [&slot, &alpha, this](
            typename nn_t::neuron & n)
        {
            const auto & bw_res = slot.bw.fx(n.index());

            n.for_each_dendrite(
            [&slot, &alpha, &bw_res, this](
                typename nn_t::neuron::dendrite & dend)
            {
                const auto & fw_res = slot.fw.fx(dend.source.index());

                dend.weight -= alpha * bw_res.delta * fw_res.phi_net;
            });
        });
    }

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  nn  Neural network
     */
    backpropagation(nn_t & nn):
        m_network(nn),
        m_fmap(create_fmap(m_network))
    {}

    /**
     *  \brief  Run backpropagation on a single input/output pair
     *
     *  Implements on-line and stochastic training modes.
     *  The \c Criterion functor takes one argument of error norm squared
     *  (square of output difference norm) and returns learning factor
     *  for the backward error propagation.
     *  Update is done immediately after the computation,
     *  if the \c criterion returns non-zero learning factor.
     *  This mechanism allows for adaptive learning as well as fixed
     *  learning factor (the \c criterion defines entirely the learning
     *  progress and/or stop condition).
     *
     *  \tparam Input      Input container type (iterable)
     *  \tparam Output     Output container type (iterable)
     *  \tparam Criterion  Update criterion type
     *  \param  input      Input
     *  \param  output     Output (desired)
     *  \param  criterion  Update criterion
     *
     *  \return Error norm squared
     */
    template <class Input, class Output, class Criterion>
    Base_t operator () (
        const Input  & input,
        const Output & output,
        Criterion    & criterion)
    {
        assert_slots(1);

        Base_t error_norm2 = compute(input, output, m_slots.front());
        const Base_t alpha = criterion(error_norm2);
        if (0 != alpha) update(alpha, m_slots.front());

        return error_norm2;
    }

    /**
     *  \brief  Run backpropagation on a training set
     *
     *  Implements batch training mode.
     *  Update is done based on average error propagation
     *  over the whole set of training samples.
     *  See on-line/stochastic overload for explanation of \c criterion
     *  parameter.
     *  Just note that the criterion takes average of the samples error
     *  norms squared and its return value is divided by the set size
     *  before it's applied as the learning factor per each sample.
     *
     *  \tparam TSet       Training set (iterable container of
     *                     \c std::pair containing [input, output] samples)
     *  \tparam Criterion  Update criterion type
     *  \param  set        Training set
     *  \param  criterion  Update criterion
     *
     *  \return Error norm squared average
     */
    template <class TSet, class Criterion>
    Base_t operator () (
        const TSet   & set,
        Criterion    & criterion)
    {
        size_t set_size = set.size();

        assert_slots(set_size);

        // Compute batch
        Base_t error_norm2_avg = 0;
        auto iter = set.begin();
        for (auto slot = m_slots.begin(); slot != m_slots.end(); ++slot, ++iter)
            error_norm2_avg += compute(iter->first, iter->second, *slot);

        error_norm2_avg /= set_size;

        // Get learning factor
        const Base_t alpha = criterion(error_norm2_avg);

        // Update batch
        if (0 != alpha) {
            const Base_t alpha4sample = alpha / set_size;
            for (auto slot = m_slots.begin(); slot != m_slots.end(); ++slot)
                update(alpha4sample, *slot);
        }

        return error_norm2_avg;
    }

};  // end of template class backpropagation

}}  // end of namespace libnn::ml

#endif  // end of #ifndef libnn__ml__backpropagation_hxx
