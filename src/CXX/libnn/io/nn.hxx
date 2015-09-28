#ifndef libnn__io__nn_hxx
#define libnn__io__nn_hxx

/**
 *  Neural network (de)serialisation
 *
 *  \date    2015/09/25
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

#include <iostream>
#include <regex>
#include <string>
#include <cassert>


using namespace libnn::topo;

namespace libnn {
namespace io {

/**
 *  \brief  Serialise neural network topology
 *
 *  \tparam Base_t   Base numeric type
 *  \tparam Act_fn   Activation function
 *  \param  out      Output stream
 *  \param  network  Neural network
 *  \param  indent   Indentation prefix
 */
template <typename Base_t, class Act_fn>
std::ostream & serialise(
    std::ostream & out,
    const nn<Base_t, Act_fn> & network,
    const std::string & indent = "")
{
    static const char * const type_str[] = {
        /* INNER  */  "INNER",
        /* INPUT  */  "INPUT",
        /* OUTPUT */  "OUTPUT",
    };

    out << indent << "NNTopology" << std::endl;

    // Serialise neurons
    network.for_each_neuron(
    [&](const typename nn<Base_t, Act_fn>::neuron & n) {
        assert((int)n.type() < sizeof(type_str) / sizeof(type_str[0]));
        const auto n_type_str = type_str[(int)n.type()];

        out
            << indent << "    Neuron " << n.index()       << std::endl
            << indent << "        type  = " << n_type_str << std::endl
            << indent << "        f     = " << n.act_fn() << std::endl
            << indent << "    NeuronEnd"                  << std::endl;
    });

    // Serialise synapses
    network.for_each_neuron(
    [&](const typename nn<Base_t, Act_fn>::neuron & n) {
        n.for_each_dendrite(
        [&](const typename nn<Base_t, Act_fn>::neuron::dendrite & d) {
            out
                << indent << "    Synapsis "
                << indent << d.source.index() << " -> " << n.index()
                << indent << " weight = " << d.weight
                << indent << std::endl;
        });
    });

    out << indent << "NNTopologyEnd" << std::endl;

    return out;
}

}}  // end of namespace libnn::io


// (De)serialisation operators
/** \cond */
template <typename Base_t, class Act_fn>
std::ostream & operator << (
    std::ostream & out,
    const nn<Base_t, Act_fn> & network)
{
    return libnn::io::serialise(out, network);
}
/** \endcond */

#endif  // end of #ifndef libnn__io__nn_hxx
