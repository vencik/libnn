/**
 *  Feed-forward neural network unit test
 *
 *  \date    2015/10/10
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


#include "config.hxx"

#include <libnn/model/feed_forward.hxx>
#include <libnn/io/feed_forward.hxx>
#include <libnn/math/util.hxx>

#include <iostream>
#include <exception>
#include <stdexcept>
#include <list>


/** Identity activation functor */
template <typename Base_t>
class identity {
    public:

    /** Identity function */
    Base_t operator () (const Base_t & x) const { return x; }

    /** Identity derivation (i.e. 1) */
    Base_t d(const Base_t & x) const { return 1; }

};  // end of template class identity

/** Simple linear feed-forward neural network model */
typedef libnn::model::feed_forward<double, identity<double> > nn_t;

/** Adaptive learning criterion */
typedef libnn::ml::adaptive_learning_factor<double>
    adaptive_learning_factor_t;


/** Identity activation functor serialisation */
template <typename Base_t>
std::ostream & operator << (
    std::ostream & out,
    const identity<Base_t> & id)
{
    return out << "identity";
}


/**
 *  \brief  Feed-forward test
 *
 *  \param  loops  Training loop count
 *  \param  alpha  Initial learning factor
 *  \param  sigma  Acceptable error
 *
 *  \return Count of errors
 */
static int test_ff(size_t loops, double alpha, double sigma) {
    std::cout << "Feed-forward NN test BEGIN" << std::endl;

    int error_cnt = 0;

    const double min = -100, max = 100;
    libnn::math::rng_uniform<double> rng(min, max);  // X ~ U(min, max)

    //
    // Create the network:
    // * input layer of 4 neurons
    // * hidden layer of 6 neurons
    // * output layer of 3 neurons
    // * using bias
    // * using lateral synapses
    //

    nn_t nn(4, 6, 3, nn_t::BIAS | nn_t::LATERAL);

    //
    // Train the network
    //

    std::cout << "Initial learning factor: " << alpha << std::endl;
    std::cout << "Acceptable error: " << sigma << std::endl;

    nn_t::training_t training = nn.training();

    adaptive_learning_factor_t criterion(sigma, alpha);

    // f([x, y, z, c]) = [2x + y + 2c - 1, 4x + z - 3c - 5, 3y + c - x + 10]
    auto f = [](const std::vector<double> & i) -> std::vector<double> {
        std::vector<double> o; o.reserve(3);
        o.push_back(2*i[0] + i[1] + 2*i[3] - 1);
        o.push_back(4*i[0] + i[2] - 3*i[3] - 5);
        o.push_back(3*i[1] + i[3] -   i[0] + 10);
        return o;
    };

    // Generate training samples
    std::list<std::pair<
        const std::vector<double>,
        const std::vector<double> > > set;

    std::cout << "Training samples:" << std::endl;

    for (size_t i = 0; i < 100; ++i) {
        std::vector<double> input; input.reserve(4);
        input.push_back(rng());
        input.push_back(rng());
        input.push_back(rng());
        input.push_back(rng());

        const std::vector<double> output = f(input);

        std::cout
            << "f["
            << input[0] << ',' << input[1] << ','
            << input[2] << ',' << input[3]
            << "], == ["
            << output[0] << ',' << output[1] << ',' << output[2]
            << "]" << std::endl;

        set.emplace_back(input, output);
    }

    // Train
    double en2 = 0;

    double en2_order = -1;
    for (size_t i = 0; i < loops; ++i) {
        en2 = training(set, criterion);

        // Print each order-magnitude improvement
        if (en2 / en2_order <= 0.1) {
            std::cout
                << "Loop " << i + 1 << ": |err|^2 == " << en2
                << std::endl;

            en2_order = en2;
        }

        // We use batch training; if there was not update once,
        // there'll never be one after that
        if (!criterion.update()) break;
    }

    // We can learn this
    if (!(en2 <= sigma)) {
        std::cout << "Failed to learn" << std::endl;

        ++error_cnt;
    }

    // Test
    nn_t::function_t function = nn.function();

    std::cout << "Test samples:" << std::endl;

    for (size_t i = 0; i < 10; ++i) {
        std::vector<double> input; input.reserve(4);
        input.push_back(rng());
        input.push_back(rng());
        input.push_back(rng());
        input.push_back(rng());

        const std::vector<double> output = f(input);

        const auto nn_output = function(input);
        const double err_0  = nn_output[0] - output[0];
        const double err_1  = nn_output[1] - output[1];
        const double err_2  = nn_output[2] - output[2];
        const double err_n2 = err_0 * err_0 + err_1 * err_1 + err_2 * err_2;

        std::cout
            << "x = ["
            << input[0] << ',' << input[1] << ','
            << input[2] << ',' << input[3]
            << ']' << std::endl
            << "f(x) == ["
            << output[0] << ',' << output[1] << ',' << output[2]
            << ']' << std::endl
            << "net_f(x) == ["
            << nn_output[0] << ',' << nn_output[1] << ',' << nn_output[2]
            << ']' << std::endl
            << "|err|^2 == " << err_n2
            << std::endl;

        if (!(err_n2 <= sigma * 10)) {
            std::cout << "Failed to generalise" << std::endl;

            ++error_cnt;
        }
    }

    std::cout << "Network:" << std::endl << nn;

    std::cout << "Feed-forward NN test END" << std::endl;

    return error_cnt;
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    size_t loops = 100;
    if (1 < argc) loops = ::atoi(argv[1]);

    double alpha = 0.0001;  // initial learning factor
    if (2 < argc) alpha = ::atof(argv[2]);

    double sigma = 1e-20;  // acceptable error
    if (3 < argc) sigma = ::atof(argv[3]);

    do {  // pragmatic do ... while (0) loop allowing for breaks
        exit_code = test_ff(loops, alpha, sigma);
        if (0 != exit_code) break;

    } while (0);  // end of pragmatic loop

    std::cerr
        << "Exit code: " << exit_code
        << std::endl;

    return exit_code;
}

/** Unit test exception-safe wrapper */
int main(int argc, char * const argv[]) {
    int exit_code = 128;

    try {
        exit_code = main_impl(argc, argv);
    }
    catch (const std::exception & x) {
        std::cerr
            << "Standard exception caught: "
            << x.what()
            << std::endl;
    }
    catch (...) {
        std::cerr
            << "Unhandled non-standard exception caught"
            << std::endl;
    }

    return exit_code;
}
