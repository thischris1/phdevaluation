#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/usr/include/scotch -I/usr/include/suitesparse -I/usr/lib/petscdir/3.7.3/x86_64-linux-gnu-real/include -I/usr/lib/slepcdir/3.7.2/x86_64-linux-gnu-real/include -I/usr/lib/openmpi/include/openmpi -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent/include -I/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent -I/usr/include/hdf5/openmpi -I/usr/include/eigen3 -I/home/chris/.cache/dijitso/include dolfin_expression_0c0ef36c4fe12b7776a10c36e99bfacf.cpp -L/usr/lib/openmpi/lib -L/usr/lib/petscdir/3.7.3/x86_64-linux-gnu-real/lib -L/usr/lib/slepcdir/3.7.2/x86_64-linux-gnu-real/lib -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib -L/home/chris/.cache/dijitso/lib -Wl,-rpath,/home/chris/.cache/dijitso/lib -olibdijitso-dolfin_expression_0c0ef36c4fe12b7776a10c36e99bfacf.so