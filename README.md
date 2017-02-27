# ProPPA

Software tool for the ProPPA language (Probabilistic Programming Process Algebra).

ProPPA is a language for describing continuous-time stochastic systems where the structure of the interactions is known but there is uncertainty about some of the parameters controlling the dynamics. Observations from the system can then help refine this uncertainty. The language definition can be found in the paper:

["Probabilistic Programming Process Algebra"](http://link.springer.com/chapter/10.1007/978-3-319-10696-0_21)
by A. Georgoulas, J. Hillston and G. Sanguinetti


The purpose of this tool is to enable automatic inference of the uncertain parameters through a number of methods. Additional functionality includes the ability to simulate models for different values of parameters, either before or after inference has been performed.

Simple examples can be found in the [models](models) directory.

## Usage
python proppa.py model_name [-o output_file]

This loads the specified file and proceeds to infer the uncertain parameters. The output is a set of samples from the learned distribution of parameters, which can then be visualised as needed or otherwised analysed.
See 'SIR.proppa' for an example model. The observations are provided in a different file, and are simply measurements of the different species at intermittent times (see example 'obs_SIR'). The inference algorithm is specified in the model file and can be configured in an optional configuration file ('config_SIR').

The code can also be used as a library to programmatically analyse and simulate ProPPA files.

## Dependencies
The tool is written in Python 3 and requires the following packages:
- pyparsing for parsing the model file
- docopt for argument parsing (possibly to be removed in a future update)
- numpy and scipy for computations
- matplotlib (only if plotting)

See also: [pepapot](https://github.com/allanderek/pepapot), a simulation and analysis tool for models written in PEPA/Bio-PEPA, from which this implementation was initially inspired.
