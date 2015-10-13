#ifndef libnn__math__sigmoid_hxx
#define libnn__math__sigmoid_hxx

/**
 *  Sigmoid functions
 *
 *  Sigmoid (i.e. S-shaped) functions have much use in neural networks.
 *  Activation functions are commonly constructed using a sigmoid function.
 *  The file offers various sigmoids.
 *
 *  See https://en.wikipedia.org/wiki/Sigmoid_function
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

#include <libnn/math/common.hxx>
#include <libnn/math/util.hxx>


namespace libnn {
namespace math {

/**
 *  \brief  Signum function
 *
 *  Note that the functor requires that the \c Base_t type value
 *  is comparable with literal 0 using < and == operators.
 *
 *  See https://en.wikipedia.org/wiki/Sign_function
 */
template <typename Base_t>
class sign_fn {
    public:

    /** Returns signum function value for argument \c x */
    Base_t operator () (const Base_t & x) const {
        if (x < 0) return -1;
        if (x == 0) return  0;
        return 1;
    }

};  // end of template class sign_fn


/**
 *  \brief  Logistic function
 *
 *  With the default parameters, this template provides
 *  the Standard (logistic) sigmoid function.
 *
 *  See https://en.wikipedia.org/wiki/Logistic_function
 *
 *  \tparam  Base_t  Base numeric type
 *  \tparam  X0      Midpoint
 *  \tparam  L       Maximum value
 *  \tparam  K       Steepness
 */
template <
    typename Base_t,
    class    X0 = int_parameter<Base_t, 0>,
    class    L  = int_parameter<Base_t, 1>,
    class    K  = int_parameter<Base_t, 1> >
class logistic_fn {
    public:

    /** Returns logistic function value for argument \c x */
    Base_t operator () (const Base_t & x) const {
        return L() / (1 + exp(-K() * (x - X0())));
    }

    /** Returns logistic function derivation value for \c x */
    Base_t d(const Base_t & x) const {
        Base_t f_x = (*this)(x);
        return K() * (1 - f_x / L()) * f_x;
    }

};  // end of template class logistic_fn


/**
 *  \brief  Error function
 *
 *  See https://en.wikipedia.org/wiki/Error_function
 *
 *  \tparam  Base_t  Base numeric type
 */
template <typename Base_t>
class error_fn {
    public:

    /** Returns error function value for argument \c x */
    Base_t operator () (const Base_t & x) const { return erf(x); }

};  // end of template class error_fn


/**
 *  \brief  Arctangent
 *
 *  See https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
 *
 *  \tparam  Base_t  Base numeric type
 */
template <typename Base_t>
class arctangent_fn {
    public:

    /** Returns arctangent value for argument \c x */
    Base_t operator () (const Base_t & x) const { return atan(x); }

};  // end of template class arctangent_fn


/**
 *  \brief  Hyperbolic tangent
 *
 *  See https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
 *
 *  \tparam  Base_t  Base numeric type
 */
template <typename Base_t>
class hyperbolic_tangent_fn {
    public:

    /** Returns hyperbolic tangent value for argument \c x */
    Base_t operator () (const Base_t & x) const {
        return 2 / (1 + exp(-2 * x)) - 1;
    }

};  // end of template class hyperbolic_tangent_fn

}}  // end of namespace libnn::math

#endif  // end of #ifndef libnn__math__sigmoid_hxx
