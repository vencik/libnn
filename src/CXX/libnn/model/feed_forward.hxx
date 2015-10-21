#ifndef libnn__model__feed_forward_hxx
#define libnn__model__feed_forward_hxx

/**
 *  Feed-Forward Neural Network
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

#include "libnn/topo/nn.hxx"
#include "libnn/ml/nn_func.hxx"
#include "libnn/ml/backpropagation.hxx"
#include "libnn/math/util.hxx"

#include <stdexcept>
#include <algorithm>
#include <cstdarg>


namespace libnn {
namespace model {

/**
 *  \brief  Feed-forward neural network
 *
 *  N-layer feed-forward NN with optional bias and lateral synapses.
 *  The topology is acyclic (lateral synapses only connect to previous
 *  neurons in a layer).
 *
 *  \tparam  Base_t         Base numeric type
 *  \tparam  Act_fn         Activation function
 *  \tparam  RandWeightMin  Random weight minimum
 *  \tparam  RandWeightMax  Random weight maximum
 */
template <
    typename Base_t,
    class    Act_fn,
    class    RandWeightMin = math::fraction_parameter<Base_t, 1, 100000>,
    class    RandWeightMax = math::fraction_parameter<Base_t, 1, 1000> >
class feed_forward {
    public:

    typedef Act_fn                   act_fn_t;  /**< Activation fn type */
    typedef topo::nn<Base_t, Act_fn> topo_t;    /**< Topology type      */

    /** Feature bits */
    enum {
        NONE         = 0x0,  /**< No extra features                     */
        BIAS         = 0x1,  /**< Use bias                              */
        LATERAL_PREV = 0x2,  /**< Synapses to previous neurons in layer */

        /** All supported lateral synapses */
        LATERAL = LATERAL_PREV,

        /** Default features */
        DEFAULT = NONE
    };  // end of enum

    /** Network function */
    class func: public ml::nn_func<Base_t, Act_fn> {
        friend class feed_forward;

        private:

        /**
         *  \brief  Constructor (only available via the network method)
         *
         *  \param  topo      Neural network topology
         *  \param  features  Feaure bits sum
         */
        func(const topo_t & topo, int features):
            ml::nn_func<Base_t, Act_fn>(topo)
        {
            if (BIAS & features) this->const_fx(0, 1);  // set bias source
        }

    };  // end of class func

    /** Network training */
    class train: public ml::backpropagation<Base_t, Act_fn> {
        friend class feed_forward;

        private:

        /**
         *  \brief  Create fixation specifications
         *
         *  \param  features  Feature bits sum
         *
         *  \return Vector containing bias fixation specifications
         */
        static std::vector<std::pair<size_t, Base_t> > fixations(int features) {
            std::vector<std::pair<size_t, Base_t> > fixes;
            if (BIAS & features) fixes.emplace_back(0, 1);
            return fixes;
        }

        /**
         *  \brief  Constructor (only available via the network method)
         *
         *  \param  topo      Neural network topology
         *  \param  features  Feaure bits sum
         */
        train(topo_t & topo, int features):
            ml::backpropagation<Base_t, Act_fn>(topo, fixations(features))
        {}

    };  // end of class train

    typedef func  function_t;  /**< Network function alias */
    typedef train training_t;  /**< Network training alias */

    private:

    int    m_features;  /**< Feature bits sum */
    topo_t m_topo;      /**< Implementation   */

    /**
     *  \brief  Create network topology
     *
     *  \tparam WInit        Weight initialiser functor type
     *  \param  layers_spec  Number of neurons per each layer
     *  \param  w_init       Weight initialiser functor
     */
    template <class WInit>
    void create_topo(const std::vector<size_t> & layers_spec, WInit & w_init) {
        if (layers_spec.size() < 2)
            throw std::logic_error(
                "libnn::model::feed_forward: "
                "invalid topology: not enough layers");

        typename topo_t::neuron * bias = NULL;

        // Create bias source
        if (BIAS & m_features) bias = &m_topo.add_neuron();

        std::vector<typename topo_t::neuron *> prev_layer;

        // Create input layer
        prev_layer.reserve(layers_spec[0]);
        for (size_t i = 0; i < layers_spec[0]; ++i)
            prev_layer.push_back(&m_topo.add_neuron(topo_t::neuron::INPUT));

        // Create hidden and output layers
        for (size_t i = 1; i < layers_spec.size(); ++i) {
            // Neuron type for this layer
            typename topo_t::neuron::type_t type
                = i < layers_spec.size() - 1
                ? topo_t::neuron::INNER
                : topo_t::neuron::OUTPUT;

            std::vector<typename topo_t::neuron *> layer;

            layer.reserve(layers_spec[i]);
            for (size_t j = 0; j < layers_spec[i]; ++j) {
                typename topo_t::neuron & n = m_topo.add_neuron(type);

                // Create bias synapsis
                if (bias) n.set_dendrite(*bias, w_init());

                // Create lateral synapses (to previous neurons in layer)
                if (LATERAL_PREV & m_features)
                    std::for_each(layer.begin(), layer.end(),
                    [&n,&w_init](typename topo_t::neuron * n_sibling) {
                        n.set_dendrite(*n_sibling, w_init());
                    });

                // Create synapses to previous layer
                std::for_each(prev_layer.begin(), prev_layer.end(),
                [&n,&w_init](typename topo_t::neuron * n_prev) {
                    n.set_dendrite(*n_prev, w_init());
                });

                layer.push_back(&n);
            }

            prev_layer = layer;
        }
    }

    /**
     *  \brief  Create vector from arguments
     *
     *  \param  cnt  Argument count
     *  \param  ...  Arguments
     *
     *  \return \c std::vector of arguments
     */
    template <typename T>
    static std::vector<T> vector(size_t cnt, ...) {
        va_list args;

        std::vector<T> v; v.reserve(cnt);

        va_start(args, cnt);

        for (size_t i = 0; i < cnt; ++i)
            v.push_back(va_arg(args, T));

        va_end(args);

        return v;
    }

    /** Create default RNG for synapsis weight initialisation */
    static math::rng_uniform<Base_t> default_rng() {
        return math::rng_uniform<Base_t>(RandWeightMin(), RandWeightMax());
    }

    public:

    /** Default constructor */
    feed_forward(): m_features(DEFAULT) {}

    /**
     *  \brief  Constructor
     *
     *  Construct feed-forward neural network, initialising the synapses
     *  weights using the \c w_init functor.
     *  Note that at least 2 layers must be specified (input and output).
     *
     *  \tparam WInit        Weight initialiser functor type
     *  \param  layers_spec  Number of neurons per each layer
     *  \param  w_init       Weight initialiser functor
     *  \param  features     Feature bits sum
     */
    template <class WInit>
    feed_forward(
        const std::vector<size_t> & layers_spec,
        WInit                     & w_init,
        int                         features)
    :
        m_features(features)
    {
        create_topo(layers_spec, w_init);
    }

    /**
     *  \brief  2-layer network constructor
     *
     *  Constructs 2-layer feed-forward neural network (i.e. with no hidden
     *  layer).
     *  Initialises synapsis weights by small non-zero random numbers.
     *
     *  \param  input_d   Input dimension
     *  \param  output_d  Output dimension
     *  \param  features  Feature bits sum
     */
    feed_forward(
        size_t input_d,
        size_t output_d,
        int    features)
    :
        m_features(features)
    {
        auto rng = default_rng();
        create_topo(vector<size_t>(2, input_d, output_d), rng);
    }

    /**
     *  \brief  3-layer network constructor
     *
     *  Constructs 3-layer feed-forward neural network (i.e. with 1 hidden
     *  layer).
     *  Initialises synapsis weights by small non-zero random numbers.
     *
     *  \param  input_d     Input dimension
     *  \param  hidden_cnt  Hidden layer size
     *  \param  output_d    Output dimension
     *  \param  features    Feature bits sum
     */
    feed_forward(
        size_t input_d,
        size_t hidden_cnt,
        size_t output_d,
        int    features)
    :
        m_features(features)
    {
        auto rng = default_rng();
        create_topo(vector<size_t>(3, input_d, hidden_cnt, output_d), rng);
    }

    /** Feature bits sum getter */
    int features() const { return m_features; }

    /**
     *  \brief  Features setter
     *
     *  NOTE that setting features is ONLY POSSIBLE if topology
     *  is not yet created.
     *  If the function is called on a network with existing topology,
     *  an exception is thrown.
     *
     *  \param  feature_bits  Feature bits sum
     */
    void features(int feature_bits) {
        if (m_topo.size())
            throw std::logic_error(
                "libnn::model::feed_forward: "
                "Can't set features for an existing topology");

        m_features = feature_bits;
    }

    /** Network topology getter */
    topo_t & topology() { return m_topo; }

    /** Network topology getter (const) */
    const topo_t & topology() const { return m_topo; }

    /**
     *  \brief  Create the network function computation
     */
    function_t function() const { return func(m_topo, m_features); }

    /**
     *  \brief  Create training algorithm for the network
     */
    training_t training() { return train(m_topo, m_features); }

};  // end of template class feed_forward

}}  // end of namespace libnn::model

#endif  // end of #ifndef libnn__model__feed_forward_hxx
