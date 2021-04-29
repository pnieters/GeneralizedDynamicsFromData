This is the code repository for the paper **A trajectory's guide to the state space - learning missing
terms in bifurcating dynamical systems** (Vortmeyer-Kley, Nieters, Pipa, 2021).


### Code organization
The core components of the experiments are organized in the local package ``GeneralizedDyanmicsFromData``
inluding a locally defined environment that can be activated and installed using the Julia Package
Manager.

For each of the dynamical systems used in the paper, and example file is provided to run
one UDE with our standard neural network model and a SInDy identification of the missing term
approximated by the neural network.

For each dynamical system, we calculated the distribution of losses after training from 100 random
initializations of the neural network. The corresponding files for each system and each loss
are provided.

The implementations of the loss functions can be found in ``src/losslib.jl``, the descriptions
of the UDEs and ODEs can be found in ``src/eqlib.jl``.