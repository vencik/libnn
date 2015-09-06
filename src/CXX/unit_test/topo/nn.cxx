/**
 *  Neural network topology unit test
 *
 *  \date    2015/09/05
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

};  // end of template class identity

/** Simple linear neural network model */
typedef libnn::topo::nn<double, identity<double> > nn_t;


/** NN topology test */
static int test_nn() {
    std::cout << "NN topology test BEGIN" << std::endl;

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

    // Internal layer
    nn_t::neuron & x1 = nn.add_neuron();
    nn_t::neuron & x2 = nn.add_neuron();

    double in1_x1 = 0.5;
    double in2_x1 = 0.3;
    double in3_x1 = 0.2;

    x1.set_dendrite(in1, in1_x1);
    x1.set_dendrite(in2, in2_x1);
    x1.set_dendrite(in3, in3_x1);

    double in2_x2 = 0.2;
    double in3_x2 = 0.3;
    double in4_x2 = 0.5;

    x2.set_dendrite(in2, in2_x2);
    x2.set_dendrite(in3, in3_x2);
    x2.set_dendrite(in4, in4_x2);

    // Output layer
    nn_t::neuron & out1 = nn.add_neuron(nn_t::neuron::OUTPUT);
    nn_t::neuron & out2 = nn.add_neuron(nn_t::neuron::OUTPUT);
    nn_t::neuron & out3 = nn.add_neuron(nn_t::neuron::OUTPUT);

    double x1_out1 = 0.2;
    double x2_out1 = 0.8;

    out1.set_dendrite(x1, x1_out1);
    out1.set_dendrite(x2, x2_out1);

    double x1_out2 = 0.5;
    double x2_out2 = 0.5;

    out2.set_dendrite(x1, x1_out2);
    out2.set_dendrite(x2, x2_out2);

    double x1_out3 = 0.8;
    double x2_out3 = 0.2;

    out3.set_dendrite(x1, x1_out3);
    out3.set_dendrite(x2, x2_out3);

    //
    // Compute
    //

    nn_t::state nn_state(nn);

    // Set input
    const std::vector<double> input({1, 2, 3, 4});

    nn_state.set_input(input);

    // Compute
    nn_state.compute();

    // Get output
    auto output = nn_state.get_output();

    std::cout << "Output:";
    std::for_each(output.begin(), output.end(), [](double x) {
        std::cout << ' ' << x;
    });
    std::cout << std::endl;

    // Test output
    double x1_f =
        input[0] * in1_x1 +
        input[1] * in2_x1 +
        input[2] * in3_x1;
    double x2_f =
        input[1] * in2_x2 +
        input[2] * in3_x2 +
        input[3] * in4_x2;

    double out1_f =
        x1_f * x1_out1 +
        x2_f * x2_out1;
    double out2_f =
        x1_f * x1_out2 +
        x2_f * x2_out2;
    double out3_f =
        x1_f * x1_out3 +
        x2_f * x2_out3;

    std::cout
        << "Expected: "
        << out1_f << ' '
        << out2_f << ' '
        << out3_f << std::endl;

    if (output[0] != out1_f || output[1] != out2_f || output[2] != out3_f)
        ++error_cnt;

    std::cout << "NN topology test END" << std::endl;

    return error_cnt;
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        if (0 != (exit_code = test_nn())) break;

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
