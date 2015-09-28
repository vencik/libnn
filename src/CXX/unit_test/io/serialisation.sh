#!/bin/sh

nn_in='
NNTopology
    # Neurons
    Neuron 0
        type = INPUT
        f    = 2x
    NeuronEnd
    Neuron 1
        type = INPUT
        f = 2x
    NeuronEnd
    Neuron 2
        type=INNER
        f=2x
    NeuronEnd
    Neuron 3
        type = OUTPUT
        f      = 2x
    NeuronEnd

    # Synapses
    Synapsis 0 -> 2  weight=0.123
    Synapsis 1-> 2   weight =0.456
    Synapsis 2->3    weight =   0.78910
NNTopologyEnd'

nn_out='NNTopology
    Neuron 0
        type = INPUT
        f    = 2x
    NeuronEnd
    Neuron 1
        type = INPUT
        f    = 2x
    NeuronEnd
    Neuron 2
        type = INNER
        f    = 2x
    NeuronEnd
    Neuron 3
        type = OUTPUT
        f    = 2x
    NeuronEnd
    Synapsis 0 -> 2 weight = 0.123
    Synapsis 1 -> 2 weight = 0.456
    Synapsis 2 -> 3 weight = 0.7891
NNTopologyEnd'


prefix=serialisation.$$

echo "$nn_in" | ./serialisation > ${prefix}.out

echo "$nn_out" > ${prefix}.exp

diff ${prefix}.* || exit 1

rm ${prefix}.*
