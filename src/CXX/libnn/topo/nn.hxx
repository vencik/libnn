#ifndef libnn__topo__nn_hxx
#define libnn__topo__nn_hxx

/**
 *  Neuron model
 *
 *  \date    2015/09/04
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


#include <list>
#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>


namespace libnn {
namespace topo {

/**
 *  \brief  Neural network
 *
 *  \tparam  Base_t  Base numeric type
 *  \tparam  Act_fn  Activation function
 */
template <typename Base_t, class Act_fn>
class nn {
    public:

    /**
     *  \brief  Neural network state
     *
     *  Computational state of a neural network instance.
     */
    class state {
        public:

        /** Neuron state */
        class neuron_state {
            private:

            Base_t m_f;        /**< Current activation function value */
            bool   m_f_set;    /**< \c true iff \c m_f is set         */
            bool   m_f_fixed;  /**< \v true iff \c m_f is fixed       */

            public:

            /** Constructor */
            neuron_state():
                m_f_set   ( false ),
                m_f_fixed ( false )
            {}

            /** Activation function value is set */
            bool f_set() const { return m_f_set; }

            /** Activation function value is fixed */
            bool f_fixed() const { return m_f_fixed; }

            /**
             *  \brief  Activation function value getter
             *
             *  The function will return the act. function value
             *  (provided that it was set).
             *
             *  \return Activation function value
             */
            const Base_t & f() const {
                if (!m_f_set)
                    throw std::logic_error(
                        "libnn::topo::nn::state::neuron_state: "
                        "activation function undefined");

                return m_f;
            }

            /**
             *  \brief  Activation function value setter
             *
             *  The function will set the activation function value.
             *  If the optional \c override_fixed parameter is \c false (default),
             *  it will do so even though the value is marked as fixed.
             *  Otherwise, it will throw an exception.
             *
             *  \param  x               Activation function value
             *  \param  override_fixed  Override fixed value (optional)
             *
             *  \return Activation function value
             */
            const Base_t & f(const Base_t & x, bool override_fixed = false) {
                if (m_f_fixed && !override_fixed)
                    throw std::logic_error(
                        "libnn::topo::nn::state::neuron_state: "
                        "attempt to set fixed activation function value");

                m_f_set = true;
                return m_f = x;
            }

            /**
             *  \brief  Fix activation function value
             *
             *  Fixes activation function value.
             */
            void f_fix() { m_f_fixed = true; }

            /**
             *  \brief  Set & fix activation function value
             *
             *  See \ref f for parameters explanation.
             *
             *  \param  x               Activation function value
             *  \param  override_fixed  Override fixed value (optional)
             */
            void f_fix(const Base_t & x, bool override_fixed = false) {
                f(x, override_fixed);
                f_fix();
            }

            /**
             *  \brief  Reset activation function value
             *
             *  The function unsets act. function value and removes its
             *  fixation mark.
             */
            void f_reset() {
                m_f_set = m_f_fixed = false;
            }

        };  // end of class neuron_state

        private:

        /** Neuron states list */
        typedef std::vector<neuron_state> neuron_states_t;

        const nn &      m_network;        /**< Neural network */
        neuron_states_t m_neuron_states;  /**< Neuron states  */

        /**
         *  \brief  Neuron status access
         *
         *  \param  index  Neuron index
         */
        neuron_state & n_state(size_t index) {
            if (!(index < m_neuron_states.size()))
                throw std::range_error(
                    "libnn::topo::nn::state: "
                    "neuron index out of range");

            return m_neuron_states[index];
        }

        public:

        /**
         *  \brief  Constructor
         *
         *  \param  network  Neural network
         */
        state(const nn & network):
            m_network(network),
            m_neuron_states(m_network.size())
        {}

        /**
         *  \brief  Neuron status access
         *
         *  \param  index  Neuron index
         */
        neuron_state & operator [] (size_t index) { return n_state(index); }

        /**
         *  \brief  Reset state
         */
        void reset() {
            m_network.for_each_neuron([this](const neuron & n) {
                n_state(n.index()).f_reset();
            });
        }

        /**
         *  \brief  Compute
         *
         *  The function computes output layer activation function values
         *  (and therefore act. function values of all neurons required
         *  to do so).
         */
        void compute() {
            m_network.for_each_output([this](const neuron & n) {
                n.compute_f(*this);
            });
        }

        /**
         *  \brief  Set input
         *
         *  \tparam Input           Input container type (iterable)
         *  \param  input           Input
         *  \param  override_fixed  Override fixed value (optional)
         */
        template <class Input>
        void set_input(const Input & input, bool override_fixed = false) {
            if (input.size() != m_network.input_size())
                throw std::range_error(
                    "libnn::topo::nn::state: "
                    "input dimension doesn't fit network input layer");

            auto in_iter = input.begin();
            m_network.for_each_input(
            [this, &in_iter, override_fixed](const neuron & n) {
                n_state(n.index()).f_fix(*in_iter, override_fixed);
                ++in_iter;
            });
        }

        /**
         *  \brief  Get output
         *
         *  \return Output (as vector)
         */
        std::vector<Base_t> get_output() {
            std::vector<Base_t> output;
            output.reserve(m_network.output_size());

            m_network.for_each_output(
            [this, &output](const neuron & n) {
                output.push_back(n_state(n.index()).f());
            });

            return output;
        }

    };  // end of class state

    /**
     *  \brief  Neuron
     *
     *  Neural cell model.
     */
    class neuron {
        public:

        /** Neuron type */
        enum type_t {
            INTERNAL = 0,  /**< Internal neuron     */
            INPUT,         /**< Input layer neuron  */
            OUTPUT,        /**< Output layer neuron */
        };  // end of enum type_t

        /**
         *  \brief  Dendrite
         *
         *  Neuron's input connection (aka synapsis) to another neuron.
         */
        struct dendrite {
            Base_t   weight;  /**< Weight        */
            neuron & source;  /**< Source neuron */

            /**
             *  \brief  Constructor (default weight)
             *
             *  \param  src  Input neuron
             */
            dendrite(neuron & src): source(src) {}

            /**
             *  \brief  Constructor
             *
             *  \param  src  Input neuron
             *  \param  w    Weight of the synapse
             */
            dendrite(neuron & src, const Base_t & w):
                weight(w),
                source(src)
            {}

        };  // end of struct dendrite

        typedef std::list<dendrite> dendrites_t;  /**< List of dendrites */

        private:

        size_t      m_index;      /**< Index       */
        type_t      m_type;       /**< Neuron type */
        dendrites_t m_dendrites;  /**< Dendrites   */

        /** Index setter */
        size_t index(size_t new_index) {
            return m_index = new_index;
        }

        /** Type setter */
        type_t type(type_t new_type) {
            return m_type = new_type;
        }

        /**
         *  \brief  Add another dendrite
         *
         *  Note that the function DOES NOT CHECK for prior existence
         *  of other dendrites to the neuron.
         *
         *  \param  n  Source neuron
         *  \param  w  Synapsis weight
         *
         *  \return New dendrite
         */
        dendrite & add_dendrite(neuron & n, Base_t w = Base_t()) {
            m_dendrites.emplace_back(n, w);
            return m_dendrites.back();
        }

        /**
         *  \brief  Remove dendrite
         *
         *  \param  d_iter  Dendrite iterator
         *
         *  \return Iterator to next dendrite (or end iterator)
         */
        typename dendrites_t::iterator
        remove_dendrite(typename dendrites_t::iterator d_iter) {
            return m_dendrites.erase(d_iter);
        }

        /**
         *  \brief  Get dendrite iterator
         *
         *  \param  n  Source neuron
         *
         *  \return Dendrite to \c n or \c m_dendrites.end() if it doesn't exist
         */
        typename dendrites_t::iterator get_dendrite_iter(const neuron & n) {
            auto d_iter = m_dendrites.begin();

            while (d_iter != m_dendrites.end() && &(d_iter->source) != &n)
                ++d_iter;

            return d_iter;
        }

        public:

        /**
         *  \brief  Constructor
         *
         *  \param  index  Neuron index
         *  \param  type   Neuron type
         */
        neuron(size_t index, type_t type = INTERNAL):
            m_index(index),
            m_type(type)
        {}

        /** Index getter */
        size_t index() const { return m_index; }

        /** Type setter */
        type_t type() const { return m_type; }

        /** Dendrite count getter */
        size_t dendrite_cnt() const { return m_dendrites.size(); }

        /**
         *  \brief  Get dendrite (i.e. synapsis to another neuron)
         *
         *  \param  n  Source neuron
         *
         *  \return Dendrite to \c n or \c NULL if it doesn't exist
         */
        dendrite * get_dendrite(const neuron & n) {
            auto d_iter = get_dendrite_iter(n);

            return m_dendrites.end() == d_iter ? NULL : &*d_iter;
        }

        /**
         *  \brief  Set dendrite (i.e. synapsis to another neuron)
         *
         *  If no such dendrite already exists, it is added.
         *
         *  \param  n  Source neuron
         *  \param  w  Synapsis weight
         *
         *  \return Dendrite to \c n
         */
        dendrite & set_dendrite(neuron & n, Base_t w = Base_t()) {
            auto d_iter = get_dendrite_iter(n);

            if (m_dendrites.end() == d_iter) return add_dendrite(n, w);

            d_iter->weight = w;
            return *d_iter;
        }

        /**
         *  \brief  Unset dendrite (i.e. remove synapsis to another neuron)
         *
         *  If such a synapsis exists, it is removed.
         *
         *  \param  n  Source neuron
         */
        void unset_dendrite(const neuron & n) {
            auto d_iter = get_dendrite_iter(n);

            if (m_dendrites.end() == d_iter) return;

            remove_dendrite(d_iter);
        }

        /**
         *  \brief  Minimise dendrite count
         *
         *  The function removes all synapses that are equal to 0.
         *  Note that the equality is checked by the \n == operator
         *  of \c Base_t type; it is actually possible that the weight
         *  is not strictly 0, but just very close to it.
         */
        void minimise_dendrites() {
            Base_t w = 0;
            auto   d_iter  = m_dendrites.begin();
            while (d_iter != m_dendrites.end()) {
                if (d_iter->weight == 0) {
                    w += d_iter->weight;
                    d_iter = remove_dendrite(d_iter);
                }
                else
                    ++d_iter;
            }
        }

        /**
         *  \brief  Compute activation function
         *
         *  If the neuron activation function is already computed,
         *  the function simply returns its value.
         *  Otherwise, it performs the computation (based on input
         *  neurons activation).
         *  If it detects a cycle it will:
         *  a) throw an exception if the cycle is unresolved
         *     (i.e. dependency can't be computed)
         *  b) re-sets the value
         *     (to enable recurent networks)
         *
         *  \param  network_state  Neural network state
         *
         *  \return The neuron's activation function value
         */
        const Base_t & compute_f(state & network_state) const {
            typename state::neuron_state & this_state = network_state[m_index];

            if (this_state.f_fixed()) return this_state.f();

            this_state.f_fix();  // won't do this again

            Base_t arg = 0;
            std::for_each(m_dendrites.begin(), m_dendrites.end(),
            [&arg, &network_state](const dendrite & dend) {
                arg += dend.weight * dend.source.compute_f(network_state);
            });

            return this_state.f(Act_fn()(arg), true);  // override fixed value
        }

    };  // end of class neuron

    private:

    typedef std::unique_ptr<neuron> neuron_ptr;  /**< Neuron pointer */
    typedef std::vector<neuron_ptr> neurons_t;   /**< Neurons list   */
    typedef std::list<size_t>       indices_t;   /**< Indices list   */

    size_t    m_size;     /**< Number of neurons */
    neurons_t m_neurons;  /**< Neurons           */
    indices_t m_inputs;   /**< Input layer       */
    indices_t m_outputs;  /**< Output layer      */

    /**
     *  \brief  Iterate over valid neuron pointers
     *
     *  \tparam Fn  Action type
     *  \param  fn  Action
     */
    template <class Fn>
    void for_each_neuron_ptr(Fn fn) {
        std::for_each(m_neurons.begin(), m_neurons.end(),
        [fn](neuron_ptr & n_ptr) {
            if (n_ptr) fn(n_ptr);
        });
    }

    /**
     *  \brief  Resolve I/O layer
     *
     *  The function return I/O layer definition for I/O neuron type,
     *  or \c NULL if the type is different.
     *
     *  \param  type  Neuron type
     *
     *  \return I/O layer definition pointer or \c NULL if N/A
     */
    indices_t * io_resolve(typename neuron::type_t type) {
        switch (type) {
            case neuron::INTERNAL: break;

            case neuron::INPUT:  return &m_inputs;
            case neuron::OUTPUT: return &m_outputs;
        }

        return NULL;
    }

    /**
     *  \brief  Add I/O layer neuron entry
     *
     *  If the \c n neuron belongs to input or output layer of the network,
     *  the function adds its index to according layer definition.
     *
     *  \param  n  Neuron
     */
    void io_add(const neuron & n) {
        auto layer = io_resolve(n.type());
        if (NULL != layer) layer->push_back(n.index());
    }

    /**
     *  \brief  Remove I/O layer neuron entry
     *
     *  If the \c n neuron belongs to input or output layer of the network,
     *  the function removes its index from according layer definition.
     *
     *  \param  n  Neuron
     */
    void io_remove(const neuron & n) {
        auto layer = io_resolve(n.type());
        if (NULL != layer) layer->remove(n.index());
    }

    public:

    /**
     *  \brief  Constructor (empty network)
     */
    nn() {}

    /**
     *  \brief  Network size (i.e. number of neurons) getter
     */
    size_t size() const { return m_size; }

    /**
     *  \brief  Input dimension (i.e. number of input layer neurons) getter
     */
    size_t input_size() const { return m_inputs.size(); }

    /**
     *  \brief  Output dimension (i.e. number of output layer neurons) getter
     */
    size_t output_size() const { return m_outputs.size(); }

    /**
     *  \brief  Iterate over neurons
     *
     *  \tparam Fn  Action type
     *  \param  fn  Action
     */
    template <class Fn>
    void for_each_neuron(Fn fn) {
        for_each_neuron_ptr([fn](neuron_ptr & n_ptr) {
            fn(*n_ptr);
        });
    }

    /**
     *  \brief  Iterate over neurons of const model
     *
     *  \tparam Fn  Action type
     *  \param  fn  Action
     */
    template <class Fn>
    void for_each_neuron(Fn fn) const {
        for_each_neuron_ptr([fn](neuron_ptr & n_ptr) {
            fn(*n_ptr);
        });
    }

    /**
     *  \brief  Iterate over input layer neurons
     *
     *  \tparam Fn  Action type
     *  \param  fn  Action
     */
    template <class Fn>
    void for_each_input(Fn fn) {
        std::for_each(m_inputs.begin(), m_inputs.end(),
        [this, fn](size_t index) {
            fn(*m_neurons[index]);
        });
    }

    /**
     *  \brief  Iterate over input layer neurons of const model
     *
     *  \tparam Fn  Action type
     *  \param  fn  Action
     */
    template <class Fn>
    void for_each_input(Fn fn) const {
        std::for_each(m_inputs.begin(), m_inputs.end(),
        [this, fn](size_t index) {
            fn(*m_neurons[index]);
        });
    }

    /**
     *  \brief  Iterate over output layer neurons
     *
     *  \tparam Fn  Action type
     *  \param  fn  Action
     */
    template <class Fn>
    void for_each_output(Fn fn) {
        std::for_each(m_outputs.begin(), m_outputs.end(),
        [this, fn](size_t index) {
            fn(*m_neurons[index]);
        });
    }

    /**
     *  \brief  Iterate over output layer neurons of const model
     *
     *  \tparam Fn  Action type
     *  \param  fn  Action
     */
    template <class Fn>
    void for_each_output(Fn fn) const {
        std::for_each(m_outputs.begin(), m_outputs.end(),
        [this, fn](size_t index) {
            fn(*m_neurons[index]);
        });
    }

    /**
     *  \brief  Add neuron
     *
     *  Note that this invalidates any existing \ref state instances.
     *  However, if state for the neuron is added to the NN state too,
     *  it would bee righted, again.
     *
     *  \param  type  Neuron type
     *
     *  \return Added neuron
     */
    neuron & add_neuron(typename neuron::type_t type = neuron::INTERNAL) {
        size_t index = m_neurons.size();
        auto n = new neuron(index, type);
        m_neurons.emplace_back(n);
        ++m_size;

        io_add(*n);  // add I/O layer entry

        return *n;
    }

    /**
     *  \brief  Remove neuron
     *
     *  Note that this technically doesn't invalidates any existing
     *  \ref state instances, since the removed neuron state will
     *  simply not be used (although it's still there).
     *
     *  \param  n  Neuron to remove
     */
    void remove_neuron(neuron & n) {
        io_remove(n);  // remove I/O layer entry

        // Remove all synapses to the neuron
        for_each_neuron([&n](neuron & n_other) {
            n_other.unset_dendrite(n);
        });

        // Destroy the neuron
        m_neurons[n.index()].reset();
        --m_size;
    }

    /**
     *  \brief  Reindex network
     *
     *  Reassigns neurons indices so that there are no gaps.
     *  Note that this invalidates any existing \ref state!
     */
    void reindex() {
        neurons_t neurons;
        neurons.reserve(m_size);

        // Clear I/O layer definitions
        m_inputs.clear();
        m_outputs.clear();

        for_each_neuron_ptr([&neurons](neuron_ptr & n_ptr) {
            size_t index = neurons.size();
            n_ptr->index(index);
            neurons.push_back(n_ptr);

            io_add(*n_ptr);  // resolve I/O layer
        });

        m_neurons.swap(neurons);
    }

    /**
     *  \brief  Prune network
     *
     *  The function removes useless dendrites (i.e. dendrites with 0 weight)
     *  from network's neurons.
     *  \see neuron::minimise_dendrites.
     *  This should be generally harmless, since such synapses don't affect
     *  the activation function value.
     */
    void prune() {
        for_each_neuron([](neuron & n) {
            n.minimise_dendrites;
        });
    }

    /**
     *  \brief  Minimise network
     *
     *  First, the function prunes the network (see \ref prune).
     *  Then it removes all internal neurons with no synapses.
     *  Note that this step MAY in fact alter the network functionality
     *  in case the activation function is non-zero for 0 argument.
     *  In such case, you probably don't want to do this.
     *  Input and output neurons are kept in any case to keep the network
     *  interface intact.
     *  If any neuron was removed, the above step is repeated (since
     *  by removal of a neuron, more of them might have lost all synapses).
     *  Finally, it reindexes the network (see \ref reindex).
     */
    void minimise() {
        prune();

        size_t rm_cnt;
        do {
            rm_cnt = 0;
            for_each_neuron([this, &rm_cnt](neuron & n) {
                if (n.type() == neuron::INTERNAL &&
                    0 == n.dendrite_cnt())
                {
                    remove_neuron(n);
                    ++rm_cnt;
                }
            });
        } while (rm_cnt);

        reindex();
    }

};  // end of template class nn

}}  // end of namespace libnn::topo

#endif  // end of #ifndef libnn__topo__nn_hxx
