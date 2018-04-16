Trials with POTTS_SPIN
=======================

You need [Eigen][].  Then everything is easy..

    git clone https://github.com/jenniferdavid/potts_spin.git
    g++ -o potts_spin potts_spin.cpp -I /usr/local/include/eigen3 -lboost_iostreams -lboost_system -lboost_filesystem

As the name implies, this is based on [POTTS SPIN BASED NN][].

[Eigen]: http://eigen.tuxfamily.org/
[POTTS SPIN BASED NN]: http://www.worldscientific.com/doi/abs/10.1142/9789814354776_0011
