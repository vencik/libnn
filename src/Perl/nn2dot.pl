#!/usr/bin/env perl

use strict;
use warnings;


use constant LEVEL_TOP    => 0;  # Top level
use constant LEVEL_NN     => 1;  # NN definition level
use constant LEVEL_NEURON => 2;  # Neuron definition level


my @neurons;
my $neuron;
my $level = LEVEL_TOP;

while (<>) {
    chomp;              # remove \n
    s/#.*//;            # remove comments
    s/\s+$//;           # remove trailing spaces
    '' eq $_ and next;  # skip empty lines

    # Top level
    if (LEVEL_TOP == $level) {
        # NN definition begins
        if (/^\s*NNTopology(\s+.*)?$/) {
            my $id = "NN"; defined $1 and $id = $1;

            print "digraph \"$id\" {\n" .
                  "    rankdir=LR;\n" .
                  "    edge [\n" .
                  "        style=solid,\n" .
                  "        dir=back,\n" .
                  "        arrowtail=odot,\n" .
                  "        fontsize=10\n" .
                  "    ];\n";

            $level = LEVEL_NN;
        }

        # Garbage
        else {
            print STDERR "Garbage on line $., ignoring\n";
        }
    }

    # NN definition level
    elsif (LEVEL_NN == $level) {
        # NN definition ends
        if (/^\s*NNTopologyEnd$/) {
            print "}\n";

            $level = LEVEL_TOP;
        }

        # Neuron definition begins
        elsif (/^\s*Neuron\s+(\d+)$/) {
            $neuron = $1;
            $neurons[$neuron] = {};

            $level = LEVEL_NEURON;
        }

        # Synapsis definition
        elsif (/^\s*Synapsis\s+(\d+)\s*->\s*(\d+)\s+weight\s*=\s*(.*)$/) {
            print "    $1 -> $2 [label=\"$3\"];\n";
        }

        # Parse error
        else {
            die "Syntax error on line $. (NN level)";
        }
    }

    # Neuron definition level
    elsif (LEVEL_NEURON == $level) {
        # Neuron definition ends
        if (/^\s*NeuronEnd$/) {
            my $style = "";
            if ("INPUT" eq $neurons[$neuron]->{type}) {
                $style = ",style=dashed";
            }
            elsif ("OUTPUT" eq $neurons[$neuron]->{type}) {
                $style = ",style=bold";
            }

            print "    $neuron [shape=circle$style];\n";

            $level = LEVEL_NN;
        }

        # Neuron type
        elsif (/^\s*type\s*=\s*(.*)$/) {
            $neurons[$neuron]->{type} = $1;
        }

        # Neuron activation function
        elsif (/^\s*f\s*=\s*(.*)$/) {
            $neurons[$neuron]->{f} = $1;
        }

        # Parse error
        else {
            die "Syntax error on line $. (neuron level)";
        }
    }

    # Unexpected level
    else {
        die "INTERNAL ERROR: unexpected parsing level";
    }
}
