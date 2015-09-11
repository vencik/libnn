#ifndef libnn__ml__computation_hxx
#define libnn__ml__computation_hxx

/**
 *  Neuron model
 *
 *  \date    2015/09/09
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

#include <vector>
#include <algorithm>
#include <stdexcept>


namespace libnn {
namespace ml {

/**
 *  \brief  Computation of activation functions over a neural network
 *
 *  \tparam  Base_t  Base numeric type
 *  \tparam  Act_fn  Activation function
 */
template <typename Base_t, class Act_fn>
class computation {
    private:

    /** Neural network type */
    typedef topo::nn<Base_t, Act_fn> nn_t;

    /** Activation function value type */
    typedef misc::fixable<Base_t> phi_t;

    /** Neurons' activation functions values */
    typedef std::vector<phi_t> phi_eval_t;

    const nn_t & m_network;   /**< Neural network                     */
    phi_eval_t   m_phi_eval;  /**< Neurons' act. functions evaluation */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  network  Neural network
     */
    computation(const nn_t & network):
        m_network(network),
        m_phi_eval(m_network.size())
    {}

    /**
     *  \brief  Reset state
     */
    void reset() {
        std::for_each(m_phi_eval.begin(), m_phi_eval.end(),
        [](phi_t & f) {
            f.reset();
        });
    }

    /**
     *  \biref  Evaluate activation function for a neuron
     *
     *  Note that the function may need to evaluate inputs, recursively.
     *  Activation function values for each neuron is, however cached
     *  in the \c state object.
     *  Feel free to call this repreatedly without loss of efficiency.
     *
     *  \param  index  Neuron index
     *
     *  \return Activation function value for neuron with \c index
     */
    const Base_t & phi(size_t index) {
        if (!(index < m_phi_eval.size()))
            throw std::range_error(
                "libnn::ml::computation::phi: "
                "neuron index out of range");

        phi_t & f = m_phi_eval[index];

        if (f.fixed()) return f;

        f.fix();  // fix in advance in case there's a cycle

        const typename nn_t::neuron & n = m_network.get_neuron(index);

        Base_t net = 0;
        n.for_each_dendrite(
        [&net, this](const typename nn_t::neuron::dendrite & dend) {
            net += dend.weight * phi(dend.source.index());
        });

        return f.set(n.act_fn(net), true);  // override early fixation
    }

    /**
     *  \brief  Evaluate activation functions
     *
     *  The function computes output layer activation function values
     *  (and therefore act. function values of all neurons required
     *  to do so).
     */
    void compute_phi() {
        m_network.for_each_output([this](const typename nn_t::neuron & n) {
            phi(n.index());
        });
    }

    /**
     *  \brief  Set input
     *
     *  \tparam Input           Input container type (iterable)
     *  \param  input           Input
     *  \param  override_fixed  Override fixed value (optional)
     */
    template <class Input>
    void set_input(const Input & input, bool override_fixed = false) {
        if (input.size() != m_network.input_size())
            throw std::range_error(
                "libnn::ml::computation::set_input: "
                "input dimension doesn't fit network input layer");

        auto in_iter = input.begin();
        m_network.for_each_input(
        [this, &in_iter, override_fixed](const typename nn_t::neuron & n) {
            m_phi_eval[n.index()].fix(*(in_iter++), override_fixed);
        });
    }

    /**
     *  \brief  Get output
     *
     *  \return Output (as vector)
     */
    std::vector<Base_t> get_output() {
        std::vector<Base_t> output;
        output.reserve(m_network.output_size());

        m_network.for_each_output(
        [this, &output](const typename nn_t::neuron & n) {
            output.push_back(m_phi_eval[n.index()]);
        });

        return output;
    }

    /**
     *  \brief  Computation
     *
     *  \tparam Input  Input container type (iterable)
     *  \param  input  Input
     *
     *  \return Output (as vector)
     */
    template <class Input>
    std::vector<Base_t> operator () (const Input & input) {
        set_input(input);
        compute_phi();
        return get_output();
    }

};  // end of class computation

}}  // end of namespace libnn::ml

#endif  // end of #ifndef libnn__ml__computation_hxx
