/**
 *  SIgmoid functions unit test
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


#include "config.hxx"

#include <libnn/math/sigmoid.hxx>

#include <iostream>
#include <exception>
#include <stdexcept>


/** Signum function in double precision fp */
typedef libnn::math::sign_fn<double> sign_fn_t;

/** Standard logistic function in double precision fp */
typedef libnn::math::logistic_fn<double> std_logistic_fn_t;

/** Error function in double precision fp */
typedef libnn::math::error_fn<double> error_fn_t;

/** Arctangent in double precision fp */
typedef libnn::math::arctangent_fn<double> arctangent_t;

/** Hyperbolic tangent in double precision fp */
typedef libnn::math::hyperbolic_tangent_fn<double> hyperbolic_tangent_t;


/** Logistic function test */
static int sign_fn_test() {
    std::cout << "Signum function test BEGIN" << std::endl;

    int error_cnt = 0;

    static const double args[] = {-6, -2, 0, 0.5, 4.7 };

    sign_fn_t sign_fn;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); ++i) {
        double x = args[i];

        std::cout
            << "f(" << x << ") == "
            << sign_fn(x)
            << std::endl;
    }

    std::cout << "Signum function test END" << std::endl;

    return error_cnt;
}


/** Logistic function test */
static int logistic_fn_test() {
    std::cout << "Logistic function test BEGIN" << std::endl;

    int error_cnt = 0;

    static const double args[] = {
        -6, -5, -4, -3, -2, -1, -0.5, -0.2, -0.1,
        0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1,
        1.5, 1.9, 2.7, 3.9, 4.5, 5.2
    };

    std_logistic_fn_t std_logistic_fn;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); ++i) {
        double x = args[i];

        std::cout
            << "f(" << x << ") == "
            << std_logistic_fn(x)
            << std::endl;
    }

    std::cout << "Logistic function test END" << std::endl;

    return error_cnt;
}


/** Error function test */
static int error_fn_test() {
    std::cout << "Error function test BEGIN" << std::endl;

    int error_cnt = 0;

    static const double args[] = {
        -6, -5, -4, -3, -2, -1, -0.5, -0.2, -0.1,
        0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1,
        1.5, 1.9, 2.7, 3.9, 4.5, 5.2
    };

    error_fn_t error_fn;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); ++i) {
        double x = args[i];

        std::cout
            << "f(" << x << ") == "
            << error_fn(x)
            << std::endl;
    }

    std::cout << "Error function test END" << std::endl;

    return error_cnt;
}


/** Arctangent test */
static int atan_test() {
    std::cout << "Arctangent test BEGIN" << std::endl;

    int error_cnt = 0;

    static const double args[] = {
        -6, -5, -4, -3, -2, -1, -0.5, -0.2, -0.1,
        0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1,
        1.5, 1.9, 2.7, 3.9, 4.5, 5.2
    };

    arctangent_t arctangent;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); ++i) {
        double x = args[i];

        std::cout
            << "f(" << x << ") == "
            << arctangent(x)
            << std::endl;
    }

    std::cout << "Arctangent test END" << std::endl;

    return error_cnt;
}


/** Hyperbolic tangent test */
static int htan_test() {
    std::cout << "Hyperbolic tangent test BEGIN" << std::endl;

    int error_cnt = 0;

    static const double args[] = {
        -6, -5, -4, -3, -2, -1, -0.5, -0.2, -0.1,
        0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1,
        1.5, 1.9, 2.7, 3.9, 4.5, 5.2
    };

    hyperbolic_tangent_t hyperbolic_tangent;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); ++i) {
        double x = args[i];

        std::cout
            << "f(" << x << ") == "
            << hyperbolic_tangent(x)
            << std::endl;
    }

    std::cout << "Hyperbolic tangent test END" << std::endl;

    return error_cnt;
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        if (0 != (exit_code = sign_fn_test())) break;

        if (0 != (exit_code = logistic_fn_test())) break;

        if (0 != (exit_code = error_fn_test())) break;

        if (0 != (exit_code = atan_test())) break;

        if (0 != (exit_code = htan_test())) break;

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
