#ifndef libnn__io__sigmoid_hxx
#define libnn__io__sigmoid_hxx

/**
 *  Sigmoid functions (de)serialisation
 *
 *  \date    2015/10/14
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

#include "libnn/math/sigmoid.hxx"

#include <iostream>
#include <stdexcept>
#include <algorithm>


// (De)serialisation operators
/** \cond */
template <typename Base_t, class X0, class L, class K>
std::ostream & operator << (
    std::ostream & out,
    const libnn::math::logistic_fn<Base_t, X0, L, K> & fn)
{
    return out << "logistic(" << X0() << ',' << L() << ',' << K() << ')';
}

template <typename Base_t, class X0, class L, class K>
std::istream & operator >> (
    std::istream & in,
    libnn::math::logistic_fn<Base_t, X0, L, K> & fn)
{
    static const std::string logistic_str("logistic");

    // Read "logistic"
    std::for_each(logistic_str.begin(), logistic_str.end(),
    [&in](char ch) {
        if (in.get() != ch)
            throw std::runtime_error(
                "libnn::io: deserialisation of logistic_fn: "
                "expected the function identifier");
    });

    // Read '('
    if (in.get() != '(')
        throw std::runtime_error(
            "libnn::io: deserialisation of logistic_fn: "
            "expected left parenthesis");

    // Read parameters
    Base_t x0, l, k;  // midpoint, asymptote, steepness

    if ((in >> x0).fail())
        throw std::runtime_error(
            "libnn::io: deserialisation of logistic_fn: "
            "midpoint deserialisation failed");
    if (in.get() != ',')
        throw std::runtime_error(
            "libnn::io: deserialisation of logistic_fn: "
            "comma expected between midpoint and asymptote");
    if ((in >> l).fail())
        throw std::runtime_error(
            "libnn::io: deserialisation of logistic_fn: "
            "asymptote deserialisation failed");
    if (in.get() != ',')
        throw std::runtime_error(
            "libnn::io: deserialisation of logistic_fn: "
            "comma expected between asymptote and steepness");
    if ((in >> k).fail())
        throw std::runtime_error(
            "libnn::io: deserialisation of logistic_fn: "
            "steepness deserialisation failed");

    // Read ')'
    if (in.get() != ')')
        throw std::runtime_error(
            "libnn::io: deserialisation of logistic_fn: "
            "expected right parenthesis");

    // Check parametetrs
    if (x0 != X0() || l != L() || k != K())
        throw std::runtime_error(
            "libnn::io: deserialisation of logistic_fn: "
            "incompatible parameters");

    return in;
}
/** \endcond */

#endif  // end of #ifndef libnn__io__sigmoid_hxx
