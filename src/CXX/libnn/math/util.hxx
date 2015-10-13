#ifndef libnn__math__util_hxx
#define libnn__math__util_hxx

/**
 *  Utilities
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

#include <stdexcept>
#include <cmath>
#include <climits>
#include <cassert>


namespace libnn {
namespace math {

/**
 *  \brief  Integer template parameter wrapper
 *
 *  Usefull for fixation of non-type template parameters.
 *  Unfortunately, the template parameter must be an integer,
 *  since C++11 doesn't allow floating point template parameters.
 *  Should you need a floating point, see \ref fraction_parameter
 *  for rational number parameters or you'll need to create
 *  a custom wrapper.
 *
 *  \tparam  Base_t  Base numeric type
 *  \tparam  Value   Constant parameter value
 */
template <typename Base_t, int Value>
class int_parameter {
    public:

    /** Returns \c Value (typecast to \c Base_t) */
    operator Base_t () const { return (Base_t)Value; }

};  // end of template class parameter


/**
 *  \brief  Rational template parameter wrapper
 *
 *  Usefull for fixation of non-type template parameters.
 *  Unfortunately, the template parameters must be integers,
 *  since C++11 doesn't allow floating point template parameters.
 *  The parameter evaluates to (typecast) fraction of the constants.
 *
 *  \tparam  Base_t       Base numeric type
 *  \tparam  Numerator    The fraction value numerator
 *  \tparam  Denominator  The fraction value denominator
 */
template <typename Base_t, int Numerator, unsigned Denominator>
class fraction_parameter {
    public:

    /** Returns \c Numerator / Denominator (typecast to \c Base_t) */
    operator Base_t () const {
        return (Base_t)Numerator / (Base_t)Denominator;
    }

};  // end of template class parameter


/**
 *  \brief  Random number generator of X ~ U(min, max)
 *
 *  Provides random variable with uniform distribution evaluation.
 *
 *  \tparam  Base_t       Base numeric type
 *  \tparam  Granularity  Default precision granularity quotient
 */
template <
    typename Base_t,
    class    Granularity = int_parameter<Base_t, 1000000000> >
class rng_uniform {
    private:

    const Base_t m_min;   /**< Minimal value         */
    const Base_t m_max;   /**< Maximal value         */
    const Base_t m_gran;  /**< Precision granularity */

    public:

    /** Default constructor; standard uniform distribution */
    rng_uniform(): m_min(0), m_max(1), m_gran(Granularity()) {}

    /**
     *  \brief  Constructor
     *
     *  \param  min   Minimal value
     *  \param  max   Maximal value
     *  \param  gran  Precision granularity quotient
     */
    rng_uniform(
        const Base_t & min,
        const Base_t & max,
        const Base_t & gran = Granularity())
    :
        m_min  ( min  ),
        m_max  ( max  ),
        m_gran ( gran )
    {
        if (!(m_min <= m_max))
            throw std::range_error(
                "libnn::math::random: "
                "invalid range specified");
    }

    /**
     *  \brief  Returns random value within [min, max]
     */
    Base_t operator () () const {
        const Base_t i((Base_t)::rand() / (Base_t)INT_MAX);
        const Base_t s((Base_t)((long long)(i * m_gran)) / m_gran);
        const Base_t x((m_max - m_min) * s + m_min);

        assert(m_min <= x && x <= m_max);

        return x;
    }

};  // end of template class rng_uniform

}}  // end of namespace libnn::math

#endif  // end of #ifndef libnn__math__util_hxx
