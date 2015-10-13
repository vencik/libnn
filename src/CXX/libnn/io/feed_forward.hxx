#ifndef libnn__io__feed_forward_hxx
#define libnn__io__feed_forward_hxx

/**
 *  Feed-forward neural network (de)serialisation
 *
 *  \date    2015/10/07
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

#include "libnn/io/nn.hxx"
#include "libnn/model/feed_forward.hxx"

#include <iostream>
#include <regex>
#include <string>
#include <sstream>
#include <cassert>


using namespace libnn::model;

namespace libnn {
namespace io {

/**
 *  \brief  Serialise feed-forward neural network
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
    const feed_forward<Base_t, Act_fn> & network,
    const std::string & indent = "")
{
    out << indent << "FFNN" << std::endl;

    out << indent << "    features = 0x"
        << std::hex << network.features() << std::dec
        << std::endl;

    serialise(out, network.topology(), indent + "    ");

    out << indent << "FFNNEnd" << std::endl;

    return out;
}


/**
 *  \brief  Deserialise feed-forward neural network
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
    feed_forward<Base_t, Act_fn> & network)
{
    network.topology().clear();  // discard existing network topology

    std::smatch bref;  // back-references
    std::string line;  // input line

    // Feed-forward NN section begin
    impl::getline(in, line);
    if (!std::regex_match(line, bref, std::regex(
        "^[ \\t]*FFNN$")))
    {
        throw std::runtime_error(
            "libnn::io::deserialise: "
            "FFNN section expected");
    }

    // Features
    impl::getline(in, line);
    if (!std::regex_match(line, bref, std::regex(
        "^[ \\t]*features[ \\t]*=[ \\t]*([xa-f\\d]+)$")))
    {
        throw std::runtime_error(
            "libnn::io::deserialise: "
            "features expected");
    }

    network.features(impl::lexical_cast<int>(bref[1]));

    // Topology
    deserialise(in, network.topology());

    // Feed-forward NN section end
    impl::getline(in, line);
    if (!std::regex_match(line, bref, std::regex(
        "^[ \\t]*FFNNEnd[ \\t]*$")))
    {
        throw std::runtime_error(
            "libnn::io::deserialise: "
            "FFNN section end expected");
    }

    return in;
}

}}  // end of namespace libnn::io


// (De)serialisation operators
/** \cond */
template <typename Base_t, class Act_fn>
std::ostream & operator << (
    std::ostream & out,
    const feed_forward<Base_t, Act_fn> & network)
{
    return libnn::io::serialise(out, network);
}

template <typename Base_t, class Act_fn>
std::istream & operator >> (
    std::istream & in,
    feed_forward<Base_t, Act_fn> & network)
{
    return libnn::io::deserialise(in, network);
}
/** \endcond */

#endif  // end of #ifndef libnn__io__feed_forward_hxx
