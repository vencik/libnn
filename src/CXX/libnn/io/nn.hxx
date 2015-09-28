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
#include <sstream>
#include <cassert>


using namespace libnn::topo;

namespace libnn {
namespace io {

namespace impl {

/**
 *  \brief  Lexical conversion from string
 *
 *  Converts \c str to type \c T.
 *  Throws an exception on error.
 *
 *  \param  str  String
 *
 *  \return \c str as type \c T
 */
template <typename T>
const T lexical_cast(const std::string & str) {
    T retval;
    std::stringstream ss(str);
    if ((ss >> retval).fail())
        throw std::runtime_error(
            "libnn::io::impl::lexical_cast: "
            "failed to convert value");

    return retval;
}


/**
 *  \brief  Read meaningful line
 *
 *  Removes comments and trailing spaces and skips empty lines.
 *  On EOF, an empty line is provided.
 *
 *  \param  in    Input stream
 *  \param  line  Resulting line
 */
void getline(std::istream & in, std::string & line) {
    while (!in.eof()) {
        std::getline(in, line);

        // Remove comments
        size_t pos = line.find_first_of('#');
        if (std::string::npos != pos)
            line.erase(pos);

        // Remove trailing spaces
        pos = line.find_last_not_of(" \t");
        if (std::string::npos != pos) {
            line.erase(pos + 1);
            return;
        }

        // Skip empty line
    }

    line.clear();  // provides empty line on EOF
}

}  // end of namespace impl


/**
 *  \brief  Serialise neural network topology
 *
 *  \tparam Base_t   Base numeric type
 *  \tparam Act_fn   Activation function
 *  \param  out      Output stream
 *  \param  network  Neural network
 *  \param  indent   Indentation prefix
 *
 *  \return \c out
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
            << indent << "    Neuron " << n.index()      << std::endl
            << indent << "        type = " << n_type_str << std::endl
            << indent << "        f    = " << n.act_fn() << std::endl
            << indent << "    NeuronEnd"                 << std::endl;
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


/**
 *  \brief  Deserialise neural network topology
 *
 *  \tparam Base_t   Base numeric type
 *  \tparam Act_fn   Activation function
 *  \param  in       Input stream
 *  \param  network  Neural network
 *
 *  \return \c in
 */
template <typename Base_t, class Act_fn>
std::istream & deserialise(
    std::istream & in,
    nn<Base_t, Act_fn> & network)
{
    network.clear();  // discard existing network topology

    std::smatch bref;  // back-references
    std::string line;  // input line

    // Topology section begin
    impl::getline(in, line);
    if (!std::regex_match(line, bref, std::regex(
        "^[ \\t]*NNTopology$")))
    {
        throw std::runtime_error(
            "libnn::io::deserialise: "
            "topology section expected");
    }

    // Neurons
    for (;;) {
        // Neuron definition begin
        impl::getline(in, line);
        if (!std::regex_match(line, bref, std::regex(
            "^[ \\t]*Neuron[ \\t]+(\\d+)$")))
        {
            break;  // neurons parsed
        }

        size_t index = impl::lexical_cast<size_t>(bref[1]);

        // Neuron type
        impl::getline(in, line);
        if (!std::regex_match(line, bref, std::regex(
            "^[ \\t]*type[ \\t]*=[ \\t]*(.*)$")))
        {
            throw std::runtime_error(
                "libnn::io::deserialise: "
                "neuron type expected");
        }

        typename topo::nn<Base_t, Act_fn>::neuron::type_t type;
        if ("INPUT" == bref[1])
            type = topo::nn<Base_t, Act_fn>::neuron::INPUT;
        else if ("INNER" == bref[1])
            type = topo::nn<Base_t, Act_fn>::neuron::INNER;
        else if ("OUTPUT" == bref[1])
            type = topo::nn<Base_t, Act_fn>::neuron::OUTPUT;
        else
            throw std::runtime_error(
                "libnn::io::deserialise: "
                "neuron type unknown");

        // Neuron activation function
        impl::getline(in, line);
        if (!std::regex_match(line, bref, std::regex(
            "^[ \\t]*f[ \\t]*=[ \\t]*(.*)$")))
        {
            throw std::runtime_error(
                "libnn::io::deserialise: "
                "activation function specification expected");
        }

        Act_fn f = impl::lexical_cast<Act_fn>(bref[1]);

        // Neuron definition end
        impl::getline(in, line);
        if (!std::regex_match(line, bref, std::regex(
            "^[ \\t]*NeuronEnd$")))
        {
            throw std::runtime_error(
                "libnn::io::deserialise: "
                "neuron section end expected");
        }

        // Add neuron
        network.set_neuron(index, type, f);
    }

    // Synapses
    for (;;) {
        if (!std::regex_match(line, bref, std::regex(
            "^[ \\t]*Synapsis[ \\t]+(\\d+)[ \\t]*->[ \\t]*(\\d+)[ \\t]+"
            "weight[ \\t]*=[ \\t]*(.+)$")))
        {
            break;  // synapses parsed
        }

        size_t from_index   = impl::lexical_cast<size_t>(bref[1]);
        size_t to_index     = impl::lexical_cast<size_t>(bref[2]);
        const Base_t weight = impl::lexical_cast<Base_t>(bref[3]);

        // Add synapsis
        typename topo::nn<Base_t, Act_fn>::neuron & from_neuron =
            network.get_neuron(from_index);
        typename topo::nn<Base_t, Act_fn>::neuron & to_neuron =
            network.get_neuron(to_index);

        to_neuron.set_dendrite(from_neuron, weight);

        impl::getline(in, line);  // get another line
    }

    // Topology section end
    if (!std::regex_match(line, bref, std::regex(
        "^[ \\t]*NNTopologyEnd[ \\t]*$")))
    {
        throw std::runtime_error(
            "libnn::io::deserialise: "
            "topology section end expected");
    }

    return in;
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

template <typename Base_t, class Act_fn>
std::istream & operator >> (
    std::istream & in,
    nn<Base_t, Act_fn> & network)
{
    return libnn::io::deserialise(in, network);
}
/** \endcond */

#endif  // end of #ifndef libnn__io__nn_hxx
