/**
 *  Perceptron feed-forward neural network unit test
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


#include "config.hxx"

#include <libnn/model/perceptron.hxx>
#include <libnn/io/perceptron.hxx>
#include <libnn/math/util.hxx>

#include <iostream>
#include <exception>
#include <stdexcept>
#include <list>
#include <cstdlib>

extern "C" {
#include <time.h>
}


/** Perceptron feed-forward neural network model */
typedef libnn::model::perceptron<
    double,
    libnn::math::int_parameter<double, 0>,
    libnn::math::int_parameter<double, 1>,
    libnn::math::int_parameter<double, 15> >
    nn_t;


/**
 *  \brief  Perceptron neural network test
 *
 *  \param  loops       Training loop count
 *  \param  alpha       Initial learning factor
 *  \param  sigma       Acceptable error
 *  \param  learn_rate  Acceptable learn rate
 *  \param  verbose     Verbose output
 *
 *  \return Count of errors
 */
static int test_perceptron(
    size_t loops,
    double alpha,
    double sigma,
    float  learn_rate,
    bool   verbose)
{
    std::cout << "Perceptron NN test BEGIN" << std::endl;

    int error_cnt = 0;

    //
    // Create the network
    //

    const size_t input_d = 2, output_d = 1;
    nn_t nn(input_d, 2, output_d, nn_t::BIAS);

    //
    // Train the network
    //

    std::cout << "Initial learning factor: " << alpha << std::endl;
    std::cout << "Acceptable error: " << sigma << std::endl;
    std::cout << "Acceptable learn rate: " << learn_rate << std::endl;

    nn_t::training_t training = nn.training();

    libnn::ml::adaptive_learning_factor<double> criterion(sigma, alpha);

    // f([x, y]) == (x - y)^2 < 0.01
    auto f = [output_d](const std::vector<double> & x) -> std::vector<double> {
        std::vector<double> fx; fx.reserve(output_d);
        fx.push_back((x[0] - x[1]) * (x[0] - x[1]) < 0.01 ? 1 : 0);
        return fx;
    };

    // Vector normalisation
    auto normal = [](const std::vector<double> & x) -> std::vector<double> {
        double sum_xi2 = 0;
        std::for_each(x.begin(), x.end(),
        [&sum_xi2](double xi) {
            sum_xi2 += xi * xi;
        });
        std::vector<double> nx(x);
        std::for_each(nx.begin(), nx.end(),
        [sum_xi2](double & nxi) {
            nxi /= sum_xi2;
        });
        return nx;
    };

    auto rng = libnn::math::rng_uniform<double>(-10, 10);

    // Generate training samples
    std::list<std::pair<
        const std::vector<double>,
        const std::vector<double> > > set;

    if (verbose)
        std::cout << "Training samples:" << std::endl;

    for (size_t i = 0; i < 100; ++i) {
        std::vector<double> input; input.reserve(input_d);
        for (size_t i = 0; i < input_d; ++i)
            input.push_back(rng());

        input = normal(input);  // normalise

        const std::vector<double> output = f(input);

        if (verbose) {
            std::cout << "f[";
            for (size_t i = 0; i < input_d - 1; ++i)
                std::cout << input[i] << ',';
            std::cout
                << input[input_d - 1] << "] == [";
            for (size_t i = 0; i < output_d - 1; ++i)
                std::cout << output[i] << ',';
            std::cout
                << output[output_d - 1] << "]"
                << std::endl;
        }

        set.emplace_back(input, output);
    }

    // Train
    double en2 = 0;

    double en2_order = -1;
    for (size_t i = 0; i < loops; ++i) {
        en2 = training(set, criterion);

        // Print each order-magnitude improvement or regression
        double en2_fraction = en2 / en2_order;
        if (verbose || (en2_fraction <= 0.1 || en2_fraction >= 10)) {
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

    std::cout
        << "Test samples"
        << (verbose ? "" : " (only failed)")
        << ':' << std::endl;

    size_t fail_cnt = 0;
    const size_t test_cnt = 500;
    for (size_t i = 0; i < test_cnt; ++i) {
        std::vector<double> input; input.reserve(input_d);
        for (size_t i = 0; i < input_d; ++i)
            input.push_back(rng());

        input = normal(input);  // normalise

        const std::vector<double> output = f(input);

        const auto nn_output = function(input);
        double err_n2  = 0;
        double err_rn2 = 0;
        for (size_t i = 0; i < output_d; ++i) {
            const double err_i  = nn_output[i] - output[i];
            const double err_ri = (nn_output[i] < 0.5 ? 0 : 1) - output[i];
            err_n2  += err_i  * err_i;
            err_rn2 += err_ri * err_ri;
        }

        const bool failed = !(err_rn2 <= sigma * 10);

        if (verbose || failed) {
            std::cout << "x == [";
            for (size_t i = 0; i < input_d - 1; ++i)
                std::cout << input[i] << ',';
            std::cout
                << input[input_d - 1] << ']'
                << std::endl
                << "f(x) == [";
            for (size_t i = 0; i < output_d - 1; ++i)
                std::cout << output[i] << ',';
            std::cout
                << output[output_d - 1] << ']'
                << std::endl
                << "net_f(x) == [";
            for (size_t i = 0; i < output_d - 1; ++i)
                std::cout << nn_output[i] << ',';
            std::cout
                << nn_output[output_d - 1] << ']'
                << std::endl
                << "|err|^2 == " << err_n2
                << std::endl
                << "Rounded output |err|^2 == " << err_rn2
                << std::endl;
        }

        if (failed) {
            std::cout << "Failed to generalise" << std::endl;

            ++fail_cnt;
        }
    }

    const float success_rate = 1.0 - (float)fail_cnt / (float)test_cnt;

    std::cout
        << "Successful on " << success_rate * 100.0 << " % of test samples"
        << std::endl;

    if (learn_rate > success_rate) ++error_cnt;

    std::cout << "Network:" << std::endl << nn;

    std::cout << "Perceptron NN test END" << std::endl;

    return error_cnt;
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    size_t loops = 1000;
    if (1 < argc) loops = ::atoi(argv[1]);

    double alpha = 0.1;  // initial learning factor
    if (2 < argc) alpha = ::atof(argv[2]);

    double sigma = 1e-10;  // acceptable error
    if (3 < argc) sigma = ::atof(argv[3]);

    float learn_rate = 0.95;  // acceptable learn rate
    if (4 < argc) learn_rate = ::atof(argv[4]);

    bool verbose = false;  // verbose output
    static const std::string verbose_true("verbose");
    if (5 < argc) verbose = verbose_true == argv[5];

    unsigned rng_seed = time(NULL);  // RNG seed
    if (6 < argc) rng_seed = ::atol(argv[6]);

    ::srand(rng_seed);

    std::cerr << "RNG seeded with " << rng_seed << std::endl;

    do {  // pragmatic do ... while (0) loop allowing for breaks
        exit_code = test_perceptron(loops, alpha, sigma, learn_rate, verbose);
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
