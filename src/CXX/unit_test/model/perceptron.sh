#!/bin/sh

# Training attempts before pronouncing the test failed
attempts_max=10

# Set to "verbose" for verbose output
verbose=no

attempt=1
while ! ./perceptron 100000 0.01 0.001 $verbose; do
    if test $attempt -ge $attempts_max; then
        echo "$attempt failed attempts made; deeming it a failure :-("
        exit 1
    fi

    echo "Attempt $attempt wasn't successful, trying again..."
    attempt=`expr $attempt + 1`
done
