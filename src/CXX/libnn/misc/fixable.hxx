#ifndef libnn__misc__fixable_hxx
#define libnn__misc__fixable_hxx

/**
 *  Fixable value
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

#include <algorithm>
#include <stdexcept>


namespace libnn {
namespace misc {

/**
 *  \brief  Fixable value
 *
 *  Container for a value that may be fixed.
 *  The value fixation status may be checked.
 *  Fixation is only done explicitly (constructors don't fix value).
 *  Fixation may be overriden and reset.
 *
 *  \tparam  T  Value type
 */
template <typename T>
class fixable {
    private:

    T    m_val;    /**< Current value                 */
    bool m_fixed;  /**< \v true iff \c m_val is fixed */

    public:

    /** Constructor */
    fixable(): m_fixed ( false ) {}

    /** Constructor (with initialiser) */
    fixable(const T & val): m_val(val), m_fixed(false) {}

    /** Value is fixed */
    bool fixed() const { return m_fixed; }

    /** Value getter */
    const T & get() const { return m_val; }

    /** Value getter (operator) */
    operator const T & () const { return get(); }

    /**
     *  \brief  Value setter
     *
     *  The function will set the value.
     *  If the optional \c override_fixed parameter is \c true
     *  it will do so even though the value is marked as fixed.
     *  Otherwise, it will throw an exception.
     *
     *  \param  val             Value
     *  \param  override_fixed  Override fixation (optional)
     *
     *  \return Value
     */
    const T & set(const T & val, bool override_fixed = false) {
        if (fixed() && !override_fixed)
            throw std::logic_error(
                "libnn::misc::fixable: attempt to set fixed value");

        return m_val = val;
    }

    /**
     *  \brief  Value setter
     *
     *  Restriction of \ref set.
     *
     *  \param  val  Value
     *
     *  \return Value
     */
    const T & operator = (const T & val) { return set(val); }

    /** Fix value */
    void fix() { m_fixed = true; }

    /**
     *  \brief  Set & fix value
     *
     *  See \ref set for parameters explanation.
     *
     *  \param  val             Value
     *  \param  override_fixed  Override fixation (optional)
     */
    void fix(const T & val, bool override_fixed = false) {
        set(val, override_fixed);
        fix();
    }

    /**
     *  \brief  Reset value
     *
     *  The function resets value and removes its fixation mark.
     *
     *  \param  val  Initial value (optional)
     */
    void reset(const T & val = T()) {
        m_val   = val;
        m_fixed = false;
    }

};  // end of template class fixable

}}  // end of namespace libnn::misc

#endif  // end of #ifndef libnn__misc__fixable_hxx
