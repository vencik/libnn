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

        /**
         *  \brief  Constructor (activation function hard fixation)
         *
         *  \param  phi  Activation function value
         */
        forward_result(const Base_t & phi): net(0), phi_net(phi) {}

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
         *  \brief  Hard-fix neurons' activation function values
         *
         *  \param  n    Neuron index
         *  \param  phi  Activation function value
         */
        void fix(size_t n, const Base_t & phi) {
            this->const_fx(n, forward_result(phi));
        }

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

        /**
         *  \brief  Constructor (backward error propagation hard fixation)
         *
         *  \param  d  Backward error propagation hard fixation
         */
        backward_result(const Base_t & d): delta(d) {}

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
         *  \brief  Hard-fix neurons' backward error propagation
         *
         *  \param  n      Neuron index
         *  \param  delta  Backward error propagation
         */
        void fix(size_t n, const Base_t & delta) {
            this->const_fx(n, backward_result(delta));
        }

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

    /** Hard fixations list */
    typedef std::vector<std::pair<size_t, Base_t> > fixes_t;

    nn_t &              m_network;    /**< Trained neural network         */
    const forward_map_t m_fmap;       /**< The neural network forward map */
    fixes_t             m_fixes;      /**< Hard fixations list            */
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
     *  Fixes activation functions for neurons specified in \c m_fixes.
     *  Since these neurons synapses are irrelevant, it also fixes
     *  their backward propagation error (aka delta) values to 0
     *  (meaning that the potential synapses won't be altered).
     *  Note that such neurons have no reason to have synapses...
     *
     *  \param  n  Number of slots
     */
    void assert_slots(size_t n) {
        for (size_t i = m_slots.size(); i < n; ++i) {
            m_slots.emplace_back(m_network, m_fmap);

            // Fix activation function values & backward error propagations
            auto & slot = m_slots.back();

            std::for_each(m_fixes.begin(), m_fixes.end(),
            [&slot](const std::pair<size_t, Base_t> & fix) {
                slot.fw.fix(fix.first, fix.second);
                slot.bw.fix(fix.first, 0);
            });
        }
    }

    /**
     *  \brief  Backward error propagation: computation
     *
     *  Computes forward error on an \c input and its backward propagation.
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
     *  \brief  Constructor (with hard fixations)
     *
     *  Allows to set hard fixations of neurons activation function values.
     *  The \c fixes argument shall contain \c std::pair<size_t,Base_t>
     *  specifications.
     *  Respective neurons' activation functions shall be set as constants.
     *  NOTE: respective neurons' net values (sums of weighed inputs)
     *  shall be 0 (so it's not logical for them to have any synapses).
     *
     *  \tparam Fixes  Container type of hard fixations (iterable)
     *  \param  nn     Neural network
     *  \param  fixes  Container of hard fixations
     */
    template <typename Fixes>
    backpropagation(nn_t & nn, const Fixes & fixes):
        m_network(nn),
        m_fmap(create_fmap(m_network))
    {
        // Set hard fixations
        m_fixes.reserve(fixes.size());

        std::for_each(fixes.begin(), fixes.end(),
        [this](const std::pair<size_t, Base_t> & fix) {
            m_fixes.push_back(fix);
        });
    }

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


/**
 *  \brief  Fixed learning factor backpropagation criterion
 *
 *  Learning criterion with constant learning factor.
 *  Allows for specification of max. allowed error norm squared.
 *
 *  \tparam  Base_t  Base numeric type
 */
template <typename Base_t>
class const_learning_factor {
    private:

    const Base_t m_alpha;   /**< Learning factor                 */
    const Base_t m_sigma;   /**< Max. allowed error norm squared */
    bool         m_update;  /**< Last call did return non-zero   */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  sigma  Max. allowed error norm squared
     *  \param  alpha  Learning factor
     */
    const_learning_factor(const Base_t & sigma, const Base_t & alpha = 0):
        m_alpha  ( alpha ),
        m_sigma  ( sigma ),
        m_update ( false )
    {}

    /** Check whether update was done last time */
    bool update() const { return m_update; }

    /**
     *  \brief  Criterion
     *
     *  \param  err_norm2  Error norm squared
     *
     *  \return Learning factor
     */
    Base_t operator () (const Base_t & err_norm2) {
        m_update = err_norm2 > m_sigma;
        return m_update ? m_alpha : 0;
    }

};  // end of template class const_learning_factor


/**
 *  \brief  Adaptive learning factor backpropagation criterion
 *
 *  This class implements a simple yet affective adaptation mechanism
 *  for setting the learning factor so that the backpropagation algorithm
 *  maintains fast convergency:
 *  * The learning converges if consecutive training steps result in smaller
 *    error norm
 *  * There's a counter that is incremented on convergency, decremented
 *    otherwise
 *  * If the counter reaches a certain maximum (i.e. the learning process
 *    overally converges well), the learning factor is increased (to speed
 *    the convergency up)
 *  * If the counter reaches a certain negative minimum (i.e. the learning
 *    process seem to diverge), the learning factor is decreased (in attempt
 *    to achieve convergency)
 *
 *  The functor provides a way to find out whether an update was done the last
 *  time it was called.
 *  For batch training, this means that once there was no update, there won't
 *  ever be one (for the same sample set).
 *
 *  TODO: How to detect stagnation?
 *
 *  Allows for specification of max. allowed error norm squared.
 *
 *  \tparam  Base_t  Base numeric type
 */
template <typename Base_t>
class adaptive_learning_factor {
    private:

    Base_t       m_alpha;       /**< Current learning factor          */
    const Base_t m_sigma;       /**< Max. allowed error norm squared  */
    bool         m_update;      /**< Last call did return non-zero    */
    Base_t       m_last_en2;    /**< Last error norm squared          */
    int          m_conv_cnt;    /**< Convergency/divergency counter   */
    const int    m_conv_cmax;   /**< Convergency counter maximum      */
    const int    m_conv_cmin;   /**< Convergency counter minimum      */
    const Base_t m_alpha_incf;  /**< Learning factor inc. coefficient */
    const Base_t m_alpha_decf;  /**< Learning factor dec. coefficient */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  sigma       Max. allowed error norm squared
     *  \param  alpha       Initial learning factor
     *  \param  conv_cmax   Convergency counter maximum
     *  \param  conv_cmin   Convergency counter minimum
     *  \param  alpha_incf  Learning factor incrementation coefficient
     *  \param  alpha_decf  Learning factor decrementation coefficient
     */
    adaptive_learning_factor(
        const Base_t & sigma      = 0,
        const Base_t & alpha      = 0.01,
        int            conv_cmax  = 5,
        int            conv_cmin  = -2,
        const Base_t & alpha_incf = 1.15,
        const Base_t & alpha_decf = 0.3)
    :
        m_alpha ( alpha ),
        m_sigma ( sigma ),

        m_update   ( false ),
        m_last_en2 ( 0 ),
        m_conv_cnt ( 0 ),

        m_conv_cmax  ( conv_cmax  ),
        m_conv_cmin  ( conv_cmin  ),
        m_alpha_incf ( alpha_incf ),
        m_alpha_decf ( alpha_decf )
    {}

    /** Check whether update was done last time */
    bool update() const { return m_update; }

    /**
     *  \brief  Criterion
     *
     *  \param  err_norm2  Error norm squared
     *
     *  \return Learning factor
     */
    Base_t operator () (const Base_t & err_norm2) {
        m_update = err_norm2 > m_sigma;
        if (!m_update) return 0;  // no need for training

        const bool convergency = err_norm2 < m_last_en2;

        // Convergency
        if (convergency) {
            ++m_conv_cnt;

            // Converges significantly
            if (m_conv_cnt >= m_conv_cmax) {
                m_conv_cnt  = 0;
                m_alpha    *= m_alpha_incf;  // try to speed things up
            }
        }

        // Divergency (or stagnation)
        else {
            --m_conv_cnt;

            // Seems to diverge dangerously
            if (m_conv_cnt <= m_conv_cmin) {
                m_conv_cnt  = 0;
                m_alpha    *= m_alpha_decf;  // try smaller steps
            }
        }

        m_last_en2 = err_norm2;  // store |error|^2

        return m_alpha;
    }

};  // end of template class adaptive_learning_factor

}}  // end of namespace libnn::ml

#endif  // end of #ifndef libnn__ml__backpropagation_hxx
