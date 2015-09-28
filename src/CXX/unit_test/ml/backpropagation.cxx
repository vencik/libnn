/**
 *  Neural network backpropagation algorithm unit test
 *
 *  \date    2015/09/13
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

#include <libnn/topo/nn.hxx>
#include <libnn/io/nn.hxx>
#include <libnn/ml/backpropagation.hxx>

#include <vector>
#include <algorithm>
#include <iostream>
#include <exception>
#include <stdexcept>


/** Identity activation functor */
template <typename Base_t>
class identity {
    public:

    /** Identity function */
    Base_t operator () (const Base_t & x) const { return x; }

    /** Identity derivation (i.e. 1) */
    Base_t d(const Base_t & x) const { return 1; }

};  // end of template class identity

/** Simple linear neural network model */
typedef libnn::topo::nn<double, identity<double> > nn_t;

/** Simple linear neural network backpropagation algorithm */
typedef libnn::ml::backpropagation<double, identity<double> >
    backpropagation_t;


/** Identity activation functor serialisation */
template <typename Base_t>
std::ostream & operator << (
    std::ostream & out,
    const identity<Base_t> & id)
{
    return out << "identity";
}


/**
 *  \brief  NN backpropagation on-line test
 *
 *  \param  loops  Training loop count
 *  \param  alpha  Learning factor
 *  \param  sigma  Acceptable error
 *
 *  \return Count of errors
 */
static int test_backpropagation_online(
    size_t loops,
    double alpha,
    double sigma)
{
    std::cout << "NN backpropagation on-line test BEGIN" << std::endl;

    int error_cnt = 0;

    //
    // Create the network
    //

    nn_t nn;

    // Input layer
    nn_t::neuron & in1 = nn.add_neuron(nn_t::neuron::INPUT);
    nn_t::neuron & in2 = nn.add_neuron(nn_t::neuron::INPUT);
    nn_t::neuron & in3 = nn.add_neuron(nn_t::neuron::INPUT);
    nn_t::neuron & in4 = nn.add_neuron(nn_t::neuron::INPUT);

    // Inner layer
    nn_t::neuron & x1 = nn.add_neuron();
    nn_t::neuron & x2 = nn.add_neuron();

    double in1_x1 = 0.01;
    double in2_x1 = 0.01;
    double in3_x1 = 0.01;

    x1.set_dendrite(in1, in1_x1);
    x1.set_dendrite(in2, in2_x1);
    x1.set_dendrite(in3, in3_x1);

    double in2_x2 = 0.01;
    double in3_x2 = 0.01;
    double in4_x2 = 0.01;

    x2.set_dendrite(in2, in2_x2);
    x2.set_dendrite(in3, in3_x2);
    x2.set_dendrite(in4, in4_x2);

    // Output layer
    nn_t::neuron & out1 = nn.add_neuron(nn_t::neuron::OUTPUT);
    nn_t::neuron & out2 = nn.add_neuron(nn_t::neuron::OUTPUT);
    nn_t::neuron & out3 = nn.add_neuron(nn_t::neuron::OUTPUT);

    double x1_out1 = 0.01;
    double x2_out1 = 0.01;

    out1.set_dendrite(x1, x1_out1);
    out1.set_dendrite(x2, x2_out1);

    double x1_out2 = 0.01;
    double x2_out2 = 0.01;

    out2.set_dendrite(x1, x1_out2);
    out2.set_dendrite(x2, x2_out2);

    double x1_out3 = 0.01;
    double x2_out3 = 0.01;

    out3.set_dendrite(x1, x1_out3);
    out3.set_dendrite(x2, x2_out3);

    //
    // Train
    //

    std::cout << "Learning factor: " << alpha << std::endl;
    std::cout << "Acceptable error: " << sigma << std::endl;

    backpropagation_t nn_bprop(nn);

    const std::vector<double> input({1, 2, 3, 4});
    const std::vector<double> output({4, 8, 12});

    double en2 = 0;

    for (size_t i = 0; i < loops; ++i) {
        en2 = nn_bprop(input, output, alpha);

        std::cout
            << "Loop " << i + 1 << ": |err|^2 == " << en2
            << std::endl;
    }

    // We can learn this
    if (!(en2 <= sigma)) {
        std::cout << "Failed to learn" << std::endl;

        ++error_cnt;
    }

    std::cout << "Network:" << std::endl << nn;

    std::cout << "NN backpropagation on-line test END" << std::endl;

    return error_cnt;
}


/**
 *  \brief  NN backpropagation batch test
 *
 *  \param  loops  Training loop count
 *  \param  alpha  Learning factor
 *  \param  sigma  Acceptable error
 *
 *  \return Count of errors
 */
static int test_backpropagation_batch(
    size_t loops,
    double alpha,
    double sigma)
{
    std::cout << "NN backpropagation batch test BEGIN" << std::endl;

    int error_cnt = 0;

    //
    // Create the network
    //

    nn_t nn;

    // Input layer
    nn_t::neuron & in1 = nn.add_neuron(nn_t::neuron::INPUT);
    nn_t::neuron & in2 = nn.add_neuron(nn_t::neuron::INPUT);
    nn_t::neuron & in3 = nn.add_neuron(nn_t::neuron::INPUT);
    nn_t::neuron & in4 = nn.add_neuron(nn_t::neuron::INPUT);

    // Inner layer
/*
    nn_t::neuron & x1 = nn.add_neuron();
    nn_t::neuron & x2 = nn.add_neuron();
    nn_t::neuron & x3 = nn.add_neuron();

    double in1_x1 = 0.01;
    double in2_x1 = 0.01;
    double in3_x1 = 0.01;
    double in4_x1 = 0.01;

    x1.set_dendrite(in1, in1_x1);
    x1.set_dendrite(in2, in2_x1);
    x1.set_dendrite(in3, in3_x1);
    x1.set_dendrite(in4, in4_x1);

    double in1_x2 = 0.01;
    double in2_x2 = 0.01;
    double in3_x2 = 0.01;
    double in4_x2 = 0.01;

    x2.set_dendrite(in1, in1_x2);
    x2.set_dendrite(in2, in2_x2);
    x2.set_dendrite(in3, in3_x2);
    x2.set_dendrite(in4, in4_x2);

    double in1_x3 = 0.01;
    double in2_x3 = 0.01;
    double in3_x3 = 0.01;
    double in4_x3 = 0.01;

    x3.set_dendrite(in1, in1_x3);
    x3.set_dendrite(in2, in2_x3);
    x3.set_dendrite(in3, in3_x3);
    x3.set_dendrite(in4, in4_x3);
*/

    // Output layer
    nn_t::neuron & out1 = nn.add_neuron(nn_t::neuron::OUTPUT);
    nn_t::neuron & out2 = nn.add_neuron(nn_t::neuron::OUTPUT);
    nn_t::neuron & out3 = nn.add_neuron(nn_t::neuron::OUTPUT);

/*
    double x1_out1 = 0.01;
    double x2_out1 = 0.01;
    double x3_out1 = 0.01;

    out1.set_dendrite(x1, x1_out1);
    out1.set_dendrite(x2, x2_out1);
    out1.set_dendrite(x3, x3_out1);

    double x1_out2 = 0.01;
    double x2_out2 = 0.01;
    double x3_out2 = 0.01;

    out2.set_dendrite(x1, x1_out2);
    out2.set_dendrite(x2, x2_out2);
    out2.set_dendrite(x3, x3_out2);

    double x1_out3 = 0.01;
    double x2_out3 = 0.01;
    double x3_out3 = 0.01;

    out3.set_dendrite(x1, x1_out3);
    out3.set_dendrite(x2, x2_out3);
    out3.set_dendrite(x3, x3_out3);
*/

    out1.set_dendrite(in1, 0.1);
    out1.set_dendrite(in2, 0.1);
    out1.set_dendrite(in3, 0.1);
    out1.set_dendrite(in4, 0.1);

    out2.set_dendrite(in1, 0.1);
    out2.set_dendrite(in2, 0.1);
    out2.set_dendrite(in3, 0.1);
    out2.set_dendrite(in4, 0.1);

    out3.set_dendrite(in1, 0.1);
    out3.set_dendrite(in2, 0.1);
    out3.set_dendrite(in3, 0.1);
    out3.set_dendrite(in4, 0.1);

    //
    // Train
    //

    std::cout << "Learning factor: " << alpha << std::endl;
    std::cout << "Acceptable error: " << sigma << std::endl;

    backpropagation_t nn_bprop(nn);

    // f([x, y, z, q]) = q[3, 2, 1] + 2[x, y, z]
    const std::vector<double> input1({1, 2, 3, 4});
    const std::vector<double> output1({12 + 2, 8 + 4, 4 + 6});

    const std::vector<double> input2({2, 4, 6, 8});
    const std::vector<double> output2({24 + 4, 16 + 8, 8 + 12});

    const std::vector<double> input3({3, 6, 9, 12});
    const std::vector<double> output3({36 + 6, 24 + 12, 12 + 18});

    const std::vector<double> input4({4, 8, 12, 16});
    const std::vector<double> output4({48 + 8, 32 + 16, 16 + 24});

    const std::vector<double> input5({5, 10, 15, 20});
    const std::vector<double> output5({60 + 10, 40 + 20, 20 + 30});

    std::vector<std::pair<
        const std::vector<double> &,
        const std::vector<double> &> > set;
    set.reserve(4);

    set.emplace_back(input1, output1);
    set.emplace_back(input2, output2);
    set.emplace_back(input3, output3);
    set.emplace_back(input4, output4);
    set.emplace_back(input5, output5);

    double en2 = 0;

    for (size_t i = 0; i < loops; ++i) {
        en2 = nn_bprop(set, alpha);

        std::cout
            << "Loop " << i + 1 << ": |err|^2 == " << en2
            << std::endl;
    }

    // We can learn this
    if (!(en2 <= sigma)) {
        std::cout << "Failed to learn" << std::endl;

        ++error_cnt;
    }

    std::cout << "Network:" << std::endl << nn;

    std::cout << "NN backpropagation batch test END" << std::endl;

    return error_cnt;
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    size_t loops = 100;
    if (1 < argc) loops = ::atoi(argv[1]);

    double alpha = 0.005;  // learning factor
    if (2 < argc) alpha = ::atof(argv[2]);

    double sigma = 1e-20;  // acceptable error
    if (3 < argc) alpha = ::atof(argv[3]);

    do {  // pragmatic do ... while (0) loop allowing for breaks
        exit_code = test_backpropagation_online(loops, alpha, sigma);
        if (0 != exit_code) break;

        exit_code = test_backpropagation_batch(loops, alpha, sigma);
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
