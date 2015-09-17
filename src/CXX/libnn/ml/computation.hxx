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
 *  \brief  Computation of a function over a neural network
 *
 *  The \c computation result is evaluated on each node and stored
 *  in \c misc::fixable wrapper (so that it is evaluated only once,
 *  and also stops recursion in case of a cycle).
 *
 *  \tparam  Base_t  Base numeric type
 *  \tparam  Act_fn  Neuron activation function
 *  \tparam  Fx      Function return value
 */
template <typename Base_t, class Act_fn, typename Fx>
class computation {
    public:

    /** Neural network type */
    typedef topo::nn<Base_t, Act_fn> nn_t;

    private:

    /** Fixed function value type */
    typedef misc::fixable<Fx> fx_t;

    /** Neuron function values */
    typedef std::vector<fx_t> results_t;

    const nn_t & m_network;  /**< Neural network             */
    results_t    m_results;  /**< Function results           */
    bool         m_reset;    /**< Function results are reset */

    protected:

    /** Check if neuron index is within bounds */
    void check_index(size_t index) const {
        if (!(index < m_results.size()))
            throw std::range_error(
                "libnn::ml::computation: "
                "neuron index out of range");
    }

    /**
     *  \brief  Set function evaluation for a neuron
     *
     *  Set & fix \c Fn evaluation for a neuron, directly.
     *
     *  \param  index           Neuron index
     *  \paran  value           Function value
     *  \param  override_fixed  Override fixed value (optional)
     */
    void fx(size_t index, const Fx & value, bool override_fixed = false) {
        check_index(index);
        m_results[index].fix(value, override_fixed);
        m_reset = false;
    }

    /**
     *  \brief  Function (purely virtual)
     *
     *  \param  n  Neuron
     *
     *  \return Function evaluation for \c n
     */
    virtual Fx f(const typename nn_t::neuron & n) = 0;

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  network  Neural network
     */
    computation(const nn_t & network):
        m_network(network),
        m_results(m_network.slot_cnt()),
        m_reset(true)
    {}

    /** Network getter */
    const nn_t & network() const { return m_network; }

    /**
     *  \brief  Reset functions return values
     *
     *  Note that if reset is not necessary (i.e. the values are already
     *  reset, no operation is done.
     */
    void reset() {
        if (m_reset) return;

        std::for_each(m_results.begin(), m_results.end(),
        [](fx_t & value) {
            value.reset();
        });

        m_reset = true;
    }

    /**
     *  \brief  Function evaluation for a neuron (const getter)
     *
     *  If the function evaluation is fixed, it's returned.
     *  Otherwise, an exception is thrown (logical error, since the instance
     *  is constant).
     *
     *  \param  index  Neuron index
     *
     *  \return Function value for neuron with \c index
     */
    const Fx & fx(size_t index) const {
        check_index(index);

        const fx_t & value = m_results[index];

        if (value.fixed()) return value;

        throw std::logic_error(
            "libnn::ml::computation: "
            "function value not fixed for const instance");
    }

    /**
     *  \brief  Evaluate function for a neuron
     *
     *  Note that should the function need to evaluate inputs recursively,
     *  the values are stored (and fixed) in advance.
     *  This eliminates repeated computations and also infinite recursion
     *  in case of a cycle.
     *
     *  Feel free to call this repreatedly without loss of efficiency.
     *
     *  \param  index  Neuron index
     *
     *  \return Function value for neuron with \c index
     */
    const Fx & fx(size_t index) {
        check_index(index);

        fx_t & value = m_results[index];

        if (value.fixed()) return value;

        value.fix();  // fix in advance in case there's a cycle
        m_reset = false;

        const typename nn_t::neuron & n = m_network.get_neuron(index);

        return value.set(f(n), true);  // override early fixation
    }

};  // end of template class computation

}}  // end of namespace libnn::ml

#endif  // end of #ifndef libnn__ml__computation_hxx
