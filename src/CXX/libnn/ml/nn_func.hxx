#ifndef libnn__ml__nn_func_hxx
#define libnn__ml__nn_func_hxx

/**
 *  Neuron model
 *
 *  \date    2015/09/14
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


namespace libnn {
namespace ml {

/**
 *  \brief  Computation of network function
 *
 *  The \c Act_fn functor shall compute activation function of neuron.
 *
 *  Sets the network input layer and computes the output layer
 *  by evaluation of activation functions of neurons on the paths.
 *
 *  \tparam  Base_t  Base numeric type
 *  \tparam  Act_fn  Activation function
 */
template <typename Base_t, class Act_fn>
class nn_func: public computation<Base_t, Act_fn, Base_t> {
    private:

    /**< Ancestor type */
    typedef computation<Base_t, Act_fn, Base_t> computation_t;

    /**< Neural network type */
    typedef typename computation_t::nn_t nn_t;

    /**
     *  \brief  Compute activation function for a neuron
     *
     *  \param  n  Neuron
     *
     *  \return Activation function value
     */
    Base_t f(const typename nn_t::neuron & n) {
        Base_t net = 0;

        n.for_each_dendrite(
        [&net, this](const typename nn_t::neuron::dendrite & dend) {
            net += dend.weight * this->fx(dend.source.index());
        });

        return n.act_fn(net);
    }

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  network  Neural network
     */
    nn_func(const nn_t & network): computation_t(network) {}

    /**
     *  \brief  Compute network function
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
        auto in_iter = input.begin();
        this->network().for_each_input(
        [this, &in_iter](const typename nn_t::neuron & n) {
            this->fx(n.index(), *(in_iter++));
        });

        // Compute output layer
        std::vector<Base_t> output;
        output.reserve(this->network().output_size());

        this->network().for_each_output(
        [this, &output](const typename nn_t::neuron & n) {
            output.push_back(this->fx(n.index()));
        });

        return output;
    }

};  // end of template class nn_func

}}  // end of namespace libnn::ml

#endif  // end of #ifndef libnn__ml__nn_func_hxx
