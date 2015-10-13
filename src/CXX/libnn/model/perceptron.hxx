#ifndef libnn__model__perceptron_hxx
#define libnn__model__perceptron_hxx

/**
 *  Perceptron feed-forward neural network
 *
 *  \date    2015/10/03
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

#include "libnn/model/feed_forward.hxx"
#include "libnn/math/sigmoid.hxx"


namespace libnn {
namespace model {

/**
 *  \brief  Perceptron network
 *
 *  Classic perceptron model with logistic sigmoid activation function.
 *  All common feed-forward features are available (namely using bias).
 *  Default template parameters provide the standard sigmoid activation
 *  function.
 *
 *  \tparam  Base_t         Base numeric type
 *  \tparam  X0             Logistic function midpoint
 *  \tparam  L              Logistic function maximum
 *  \tparam  K              Logistic function steepness
 *  \tparam  RandWeightMin  Random weight minimum
 *  \tparam  RandWeightMax  Random weight maximum
 */
template <
    typename Base_t,
    class    X0 = math::int_parameter<Base_t, 0>,
    class    L  = math::int_parameter<Base_t, 1>,
    class    K  = math::int_parameter<Base_t, 1>,
    class    RandWeightMin = math::fraction_parameter<Base_t, 1, 100000>,
    class    RandWeightMax = math::fraction_parameter<Base_t, 1, 1000> >
class perceptron:
    public feed_forward<
        Base_t,
        math::logistic_fn<Base_t, X0, L, K>,
        RandWeightMin, RandWeightMax>
{
    private:

    /** Superclass type */
    typedef
        feed_forward<Base_t, math::logistic_fn<Base_t, X0, L, K> >
        feed_forward_t;

    public:

    /**
     *  \brief  Constructor
     *
     *  \tparam WInit        Weight initialiser functor type
     *  \param  layers_spec  Number of neurons per each layer
     *  \param  w_init       Weight initialiser functor
     *  \param  features     Feature bits sum
     */
    template <class WInit>
    perceptron(
        const std::vector<size_t> & layers_spec,
        WInit                       w_init,
        int                         features = DEFAULT)
    :
        feed_forward_t(w_init, layers_spec, features)
    {}

    /**
     *  \brief  2-layer network constructor
     *
     *  Constructs 2-layer perceptron network (i.e. with no hidden layer).
     *  Initialises synapsis weights by small non-zero random numbers.
     *
     *  \param  input_d   Input dimension
     *  \param  output_d  Output dimension
     *  \param  features  Feature bits sum
     */
    perceptron(
        size_t input_d,
        size_t output_d,
        int    features = DEFAULT)
    :
        feed_forward_t(input_d, output_d, features)
    {}

    /**
     *  \brief  3-layer network constructor
     *
     *  Constructs 3-layer perceptron network (i.e. with 1 hidden layer).
     *  Initialises synapsis weights by small non-zero random numbers.
     *
     *  \param  input_d     Input dimension
     *  \param  hidden_cnt  Hidden layer size
     *  \param  output_d    Output dimension
     *  \param  features    Feature bits sum
     */
    perceptron(
        size_t input_d,
        size_t hidden_cnt,
        size_t output_d,
        int    features = DEFAULT)
    :
        feed_forward_t(input_d, hidden_cnt, output_d, features)
    {}

};  // end of template class perceptron

}}  // end of namespace libnn::model

#endif  // end of #ifndef libnn__model__perceptron_hxx
