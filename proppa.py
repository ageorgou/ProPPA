# -*- coding: utf-8 -*-
"""
ProPPA - Probabilistic Programming Process Algebra

Usage:
    proppa.py <model_file> [--out <output_file>]
    proppa.py <model_file> infer [--out output_file]
    
Options:
    -o FILE --out FILE  Output file (default: <model_file>_out)
    -h --help           Show this help message
"""

""" Copyright notice:
A lot of this, especially the parsing, has been ported from the pepapot
repository, written by Allan Clark, and subsequently modified and adapted.
See the LICENSE file for more information.
"""


import itertools
import functools
import operator
import math
import os.path

import pyparsing
from pyparsing import Combine,Or,Optional,Literal,Suppress,MatchFirst,delimitedList
import numpy as np
import scipy as sp
from docopt import docopt

import model_utilities as mu
import samplers

# See pepapot comment on why this is useful (also:
# http://pyparsing.wikispaces.com/share/view/26068641?replyId=26084853 ,
# http://pyparsing.sourcearchive.com/documentation/1.4.7/
# classpyparsing_1_1ParserElement_81dd508823f0bf29dce996d788b1eeff.html )
pyparsing.ParserElement.enablePackrat()

def make_identifier_grammar(start_characters):
    identifier_start = pyparsing.Word(start_characters, exact=1)
    identifier_remainder = Optional(pyparsing.Word(pyparsing.alphanums + "_"))
    identifier_grammar = identifier_start + identifier_remainder
    return pyparsing.Combine(identifier_grammar)

lower_identifier = make_identifier_grammar("abcdefghijklmnopqrstuvwxyz")
upper_identifier = make_identifier_grammar("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
identifier = make_identifier_grammar(pyparsing.alphas)

plusorminus = Literal('+') | Literal('-')
number_grammar = pyparsing.Word(pyparsing.nums)
integer_grammar = Combine(Optional(plusorminus) + number_grammar)
decimal_fraction = Literal('.') + number_grammar
scientific_enotation = pyparsing.CaselessLiteral('E') + integer_grammar
floatnumber = Combine(integer_grammar + Optional(decimal_fraction) +
                      Optional(scientific_enotation))

def evaluate_function_app(name, arg_values):
    """ Used in the evaluation of expressions. This evaluates the application
        of a function.
    """
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches
    def check_num_arguments(expected_number):
        """ For some of the functions we evaluate below they have a fixed
            number of arguments, here we check that they have been supplied
            the correct number.
        """
        if len(arg_values) != expected_number:
            if expected_number == 1:
                message = "'" + name + "' must have exactly one argument."
            else:
                message = ("'" + name + "' must have exactly " +
                           str(expected_number) + " arguments.")
            raise ValueError(message)
    if name == "plus" or name == "+":
        return sum(arg_values)
    elif name == "times" or name == "*":
        return functools.reduce(operator.mul, arg_values, 1)
    elif name == "minus" or name == "-":
        # What should we do if there is only one argument, I think we
        # should treat '(-) x' the same as '0 - x'.
        if not arg_values:
            return 0
        elif len(arg_values) == 1:
            return 0 - arg_values[0]
        else:
            return functools.reduce(operator.sub, arg_values)
    elif name == "divide" or name == "/":
        if arg_values:
            return functools.reduce(operator.truediv, arg_values)
        else:
            return 1
    elif name == "power" or name == "**":
        # power is interesting because it associates to the right
        # counts downwards from the last index to the 0.
        # As an example, consider power(3,2,3), the answer should be
        # 3 ** (2 ** 3) = 3 ** 8 = 6561, not (3 ** 2) ** 3 = 9 ** 3 = 81
        # going through our loop here we have
        # exp = 1
        # exp = 3 ** exp = 3
        # exp = 2 ** exp = 2 ** 3 = 8
        # exp = 3 ** exp = 3 ** 8 = 6561
        exponent = 1
        for i in range(len(arg_values) - 1, -1, -1):
            exponent = arg_values[i] ** exponent
        return exponent
    elif name == "exp":
        check_num_arguments(1)
        return math.exp(arg_values[0])
    elif name == "floor":
        check_num_arguments(1)
        return math.floor(arg_values[0])
    elif name == "H" or name == "heaviside":
        check_num_arguments(1)
        # H is typically not actually defined for 0, here we have defined
        # H(0) to be 0. Generally it won't matter much.
        return 1 if arg_values[0] > 0 else 0
    else:
        raise ValueError("Unknown function name: " + name)

class Expression:
    """ A new simpler representation of expressions in which we only have
        one kind of expression. The idea is that reduce and get_value can be
        coded as in terms of a single recursion.
    """
    def __init__(self):
        self.name = None
        self.number = None
        self.arguments = []

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.name == other.name and
                self.number == other.number and
                self.arguments == other.arguments)

    @classmethod
    def num_expression(cls, number):
        expression = cls()
        expression.number = number
        return expression

    @classmethod
    def name_expression(cls, name):
        expression = cls()
        expression.name = name
        return expression

    @classmethod
    def apply_expression(cls, name, arguments):
        expression = cls()
        expression.name = name
        expression.arguments = arguments
        return expression

    @classmethod
    def addition(cls, left, right):
        return cls.apply_expression("+", [left, right])

    @classmethod
    def subtract(cls, left, right):
        return cls.apply_expression("-", [left, right])

    @classmethod
    def multiply(cls, left, right):
        return cls.apply_expression("*", [left, right])

    @classmethod
    def divide(cls, left, right):
        return cls.apply_expression("/", [left, right])

    @classmethod
    def power(cls, left, right):
        return cls.apply_expression("**", [left, right])

    def used_names(self):
        names = set()
        # For now we do not add function names to the list of used rate names
        # This seems correct, but if we did allow user-defined functions then
        # obviously we might wish to know which ones are used.
        if self.name and not self.arguments:
            names.add(self.name)
        for arg in self.arguments:
            names.update(arg.used_names())

        return names

    def get_value(self, environment=None):
        """ Returns the value of an expression in the given environment if
            any. Raises an assertion error if the expression cannot be reduced
            to a value.
        """
        reduced_expression = self.reduce_expr(environment=environment)
        assert reduced_expression.number is not None
        return reduced_expression.number

    def reduce_expr(self, environment=None):
        if self.number is not None:
            return self
        if not self.arguments:
            # We have a name expression so if the environment is None or
            # or the name is not in the environment then we cannot reduce
            # any further so just return the current expression.
            if not environment or self.name not in environment:
                return self
            expression = environment[self.name]
            return expression.reduce_expr(environment=environment)

        # If we get here then we have an application expression, so we must
        # first reduce all the arguments and then we may or may not be able
        # to reduce the entire expression to a number or not.
        arguments = [a.reduce_expr(environment)
                     for a in self.arguments]
        arg_values = [a.number for a in arguments]

        if any(v is None for v in arg_values):
            return Expression.apply_expression(self.name, arguments)
        else:
            result_number = evaluate_function_app(self.name, arg_values)
            return Expression.num_expression(result_number)
    
    def differentiate(self,diff_variable):
        if self.number is not None:
            return Expression.num_expression(0)
        if not self.arguments: #this is a name expression
            #if it is the variable (species) wrt which we are differentiating:
            if self.name == diff_variable:
                return Expression.num_expression(1)
            else: # if a different species or a parameter:
                return Expression.num_expression(0)
        # otherwise, we have an apply expression
        diff_args = [a.differentiate(diff_variable) for a in self.arguments]
        if self.name == '+':
            return Expression.apply_expression('+',diff_args)
        if self.name == '-':
            return Expression.apply_expression('-',diff_args)
        if self.name == '*':
            term1 = Expression.multiply(diff_args[0],self.arguments[1])
            term2 = Expression.multiply(self.arguments[0],diff_args[1])
            return Expression.addition(term1,term2)
        if self.name == '/':
            term1 = Expression.multiply(diff_args[0],self.arguments[1])
            term2 = Expression.multiply(diff_args[1],self.arguments[0])
            numerator = Expression.subtract(term1,term2)
            denominator = Expression.power(self.arguments[1],
                                           Expression.num_expression(2))
            return Expression.divide(numerator,denominator)
        if self.name == '**':
            # assuming exponents are constants (not species) for the time being
            factor = self.arguments[1]
            base = self.arguments[0]
            expon = Expression.subtract(self.arguments[1],
                                        Expression.num_expression(1))
            return Expression.multiply(factor,Expression.power(base,expon))
        raise ValueError("""Could not construct an expression of the derivative.
                Are you using a strange operator?""")
# A helper to create grammar element which must be surrounded by parentheses
# but you then wish to ignore the parentheses
def parenthetical_grammar(element_grammar):
    return Suppress("(") + element_grammar + Suppress(")")


def create_expression_grammar(identifier_grammar):
    this_expr_grammar = pyparsing.Forward()

    def num_expr_parse_action(tokens):
        return Expression.num_expression(float(tokens[0]))

    num_expr = floatnumber.copy()
    num_expr.setParseAction(num_expr_parse_action)

    def apply_expr_parse_action(tokens):
        if len(tokens) == 1:
            return Expression.name_expression(tokens[0])
        else:
            return Expression.apply_expression(tokens[0], tokens[1:])
    arg_expr_list = pyparsing.delimitedList(this_expr_grammar)
    opt_arg_list = Optional(parenthetical_grammar(arg_expr_list))
    apply_expr = identifier_grammar + opt_arg_list
    apply_expr.setParseAction(apply_expr_parse_action)

    atom_expr = Or([num_expr, apply_expr])

    multop = pyparsing.oneOf('* /')
    plusop = pyparsing.oneOf('+ -')

    def binop_parse_action(tokens):
        elements = tokens[0]
        operators = elements[1::2]
        exprs = elements[::2]
        assert len(exprs) - len(operators) == 1
        exprs_iter = iter(exprs)
        result_expr = next(exprs_iter)
        # Note: iterating in this order would not be correct if the binary
        # operator associates to the right, as with **, since for
        # [2, ** , 3, ** 2] we would get build up the apply expression
        # corresponding to (2 ** 3) ** 2, which is not what we want. However,
        # pyparsing seems to do the correct thing and give this function
        # two separate calls one for [3, **, 2] and then again for
        # [2, ** , Apply(**, [3,2])].
        for oper, expression in zip(operators, exprs_iter):
            args = [result_expr, expression]
            result_expr = Expression.apply_expression(oper, args)
        return result_expr

    precedences = [("**", 2, pyparsing.opAssoc.RIGHT, binop_parse_action),
                   (multop, 2, pyparsing.opAssoc.LEFT, binop_parse_action),
                   (plusop, 2, pyparsing.opAssoc.LEFT, binop_parse_action),
                  ]
    # pylint: disable=expression-not-assigned
    this_expr_grammar << pyparsing.operatorPrecedence(atom_expr, precedences)
    return this_expr_grammar

lower_expr_grammar = create_expression_grammar(lower_identifier)
expr_grammar = create_expression_grammar(identifier)


class DistributionTerm(object):
    def to_distribution(self):
        raise NotImplementedError

class GaussianTerm(DistributionTerm):
    def __init__(self,mean,var):
        self.mean = float(mean)
        self.var = float(var)

    grammar = (Literal("Normal") | Literal("Gaussian")) + "(" + \
                floatnumber.setResultsName("mean") + "," + \
                floatnumber.setResultsName("variance") + ")"
   
    @classmethod
    def from_tokens(cls,tokens):
        return cls(tokens[2],tokens[4])
    
    def to_distribution(self):
        return sp.stats.norm(self.mean,self.var)

class UniformTerm(DistributionTerm):
    def __init__(self,lower,upper):
        self.lower = float(lower)
        self.upper = float(upper)

    grammar = Literal("Uniform") + "(" + \
                floatnumber("lower") + "," + \
                floatnumber("upper") + ")"
                
    @classmethod
    def from_tokens(cls,tokens):
        return cls(tokens["lower"],tokens["upper"])
    
    def to_distribution(self):
        return sp.stats.uniform(self.lower,self.upper-self.lower)

class GammaTerm(DistributionTerm):
    def __init__(self,shape,rate):
        self.shape = float(shape)
        self.rate = float(rate)

    grammar = Literal("Gamma") + "(" + \
                floatnumber("shape") + "," + \
                floatnumber("rate") + ")"
                
    @classmethod
    def from_tokens(cls,tokens):
        return cls(tokens["shape"],tokens["rate"])
    
    def to_distribution(self):
        return sp.stats.gamma(self.shape,scale=1/self.rate)

class ExponentialTerm(DistributionTerm):
    def __init__(self,rate):
        self.rate = float(rate)
    
    grammar = Literal("Exponential") + "(" + floatnumber("rate") + ")"
                
    @classmethod
    def from_tokens(cls,tokens):
        return cls(tokens["rate"])
    
    def to_distribution(self):
        return sp.stats.expon(scale=1/self.rate)

dist_terms = [GaussianTerm,UniformTerm,GammaTerm,ExponentialTerm]
for d in dist_terms:
    d.grammar.setParseAction(d.from_tokens)
dist_grammar = MatchFirst([d.grammar for d in dist_terms])

class ConstantDefinition(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = identifier + "=" + (dist_grammar|expr_grammar) + ";"
    list_grammar = pyparsing.Group(pyparsing.ZeroOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[2])

ConstantDefinition.grammar.setParseAction(ConstantDefinition.from_tokens)


class DistributionDefinition(object):
    def __init__(self,lhs,rhs):
        self.lhs = lhs
        self.rhs = rhs
    
    grammar = identifier + "=" + dist_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.ZeroOrMore(grammar))
    
    @classmethod
    def from_tokens(cls,tokens):
        return cls(tokens[0], tokens[2])

DistributionDefinition.grammar.setParseAction(DistributionDefinition.from_tokens)

class DefaultDictKey(dict):
    # pylint: disable=too-few-public-methods
    """ The standard library provides the defaultdict class which
        is useful, but sometimes the value that we wish to produce
        depends upon the key. This class provides that functionality
    """
    def __init__(self, factory):
        super(DefaultDictKey, self).__init__()
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]

class RateDefinition(object):
    # pylint: disable=too-few-public-methods
    """ A class representing a rate definition in a ProPPA model. It will
        look like: "kineticLawOf r : expr;"
        Where 'r' is the rate being definined and "expr" is an arbitrary
        expression, usually involving the reactants of the reaction 'r'.
    """
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = "kineticLawOf" + identifier + ":" + expr_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        """ The parser method for ProPPA rate definitions."""
        return cls(tokens[1], tokens[3])

RateDefinition.grammar.setParseAction(RateDefinition.from_tokens)


class Behaviour(object):
    """ A class representing a behaviour. Species definitions consist of a list
        of behaviours that the species are involved in. This class represents
        one element of such a list. So for example "(a, 1) >> E" or a shorthand
        version of that "a >>"
    """
    def __init__(self, reaction, stoich, role, species):
        self.reaction_name = reaction
        self.stoichiometry = stoich
        self.role = role
        self.species = species

    # If the stoichiometry is 1, then instead of writing "(r, 1)" we allow
    # the modeller to simply write "r".
    # TODO: Consider making the parentheses optional in any case, and then
    # we can simply make the comma-stoich optional.
    prefix_identifier = identifier.copy()
    prefix_identifier.setParseAction(lambda tokens: (tokens[0], 1))

    full_prefix_grammar = "(" + identifier + "," + integer_grammar + ")"
    full_prefix_parse_action = lambda tokens: (tokens[1], int(tokens[3]))
    full_prefix_grammar.setParseAction(full_prefix_parse_action)

    prefix_grammar = Or([prefix_identifier, full_prefix_grammar])

    op_strings = ["<<", ">>", "(+)", "(-)", "(.)"]
    role_grammar = Or([Literal(op) for op in op_strings])

    # The true syntax calls for (a,r) << P; where P is the name of the process
    # being updated by the behaviour. However since this is (in the absence
    # of locations) always the same as the process being defined, it is
    # permitted to simply omit it.
    process_update_identifier = Optional(identifier, default=None)
    grammar = prefix_grammar + role_grammar + process_update_identifier

    @classmethod
    def from_tokens(cls, tokens):
        """ The parser action method for a species behaviour. """
        return cls(tokens[0][0], tokens[0][1], tokens[1], tokens[2])

    def get_population_precondition(self):
        """ Returns the pre-condition to fire this behaviour in a discrete
            simulation. So for example a reactant with stoichiometry 2 will
            require a population of at least 2.
        """
        # TODO: Not quite sure what to do with general modifier here?
        if self.role in ["<<", "(+)"]:
            return self.stoichiometry
        else:
            return 0

    def get_population_modifier(self):
        """ Returns the effect this behaviour has on the associated population
            if the behaviour were to be 'fired' once. So for example a
            reactant with stoichiometry 2, will return a population modifier
            of -2.
        """
        if self.role == "<<":
            return -1 * self.stoichiometry
        elif self.role == ">>":
            return 1 * self.stoichiometry
        else:
            return 0

    def get_expression(self, kinetic_laws):
        """ Return the expression that would be used in an ordinary
            differential equation for the associated species. In other words
            the rate of change in the given species population due to this
            behaviour.
        """
        modifier = self.get_population_modifier()
        expr = kinetic_laws[self.reaction_name]
        if modifier == 0:
            expr = Expression.num_expression(0.0)
        elif modifier != 1:
            modifier_expr = Expression.num_expression(modifier)
            expr = Expression.multiply(modifier_expr, expr)

        return expr


Behaviour.grammar.setParseAction(Behaviour.from_tokens)


class Reaction(object):
    # pylint: disable=too-few-public-methods
    """ Represents a reaction in a biological system. This does not necessarily
        have to be a reaction which is produced by parsing and processing a
        Bio-PEPA model, but the definition is here because we wish to be able
        to compute the set of all reactions defined by a Bio-PEPA model.
    """
    def __init__(self, name):
        self.name = name
        self.reactants = []
        self.activators = []
        self.products = []
        self.inhibitors = []
        self.modifiers = []

    def format(self):
        """ Format the reaction as a string """
        def format_name(behaviour):
            """ format a name within the reaction as a string."""
            if behaviour.stoichiometry == 1:
                species = behaviour.species
            else:
                species = ("(" + behaviour.species + "," +
                           str(behaviour.stoichiometry) + ")")
            if behaviour.role == "(+)":
                prefix = "+"
            elif behaviour.role == "(-)":
                prefix = "-"
            elif behaviour.role == "(.)":
                prefix = "."
            else:
                prefix = ""
            return prefix + species
        pre_arrows = itertools.chain(self.reactants, self.activators,
                                     self.inhibitors, self.modifiers)
        pre_arrow = ", ".join(format_name(b) for b in pre_arrows)
        post_arrow = ", ".join(format_name(b) for b in self.products)

        return " ".join([self.name + ":", pre_arrow, "-->", post_arrow])



class SpeciesDefinition(object):
    # pylint: disable=too-few-public-methods
    """ Class that represents a ProPPA species definition. We store the name
        and the right hand side of the definition which is a list of behaviours
        the species is involved in.
    """
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    behaviours = pyparsing.delimitedList(Behaviour.grammar, delim="+")
    grammar = identifier + "=" + pyparsing.Group(behaviours) + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        """ The parser action method for ProPPA species definition."""
        species_name = tokens[0]
        behaviours = tokens[2]
        for behaviour in behaviours:
            if behaviour.species is None:
                behaviour.species = species_name
        return cls(species_name, behaviours)


SpeciesDefinition.grammar.setParseAction(SpeciesDefinition.from_tokens)

class Observable(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = "observable" + identifier + "=" + expr_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        """ The parser method for ProPPA rate definitions."""
        return cls(tokens[1], tokens[3])

Observable.grammar.setParseAction(Observable.from_tokens)


class Population(object):
    # pylint: disable=too-few-public-methods
    """ Represents a ProPPA population. This is a process in the main
        system equation, such as "E[100]"
    """
    def __init__(self, species, amount):
        self.species_name = species
        self.amount = amount

    grammar = identifier + "[" + expr_grammar + "]"

    @classmethod
    def from_tokens(cls, tokens):
        """ The parser action method for a ProPPA population. """
        return cls(tokens[0], tokens[2])

Population.grammar.setParseAction(Population.from_tokens)

system_grammar = pyparsing.delimitedList(Population.grammar,
                                            delim="<*>")

filename_characters = pyparsing.alphanums + "." + "_" + "/"
observe_grammar = (Literal("observe") + "(" + 
               pyparsing.Word(filename_characters)("file") +
               ")" + ";")
infalg_grammar = Literal("infer") + "(" + identifier("alg") + ")" + ";"
conf_grammar = Literal("configure") + \
               "(" + pyparsing.Word(filename_characters)("file") + ")" + ";"

class ParsedModel(object):
    """ Class representing a parsed ProPPA model. It contains the grammar
        description for a model.
    """
    def __init__(self, constants, kinetic_laws, species, populations,
                 obsfile, algorithm, conffile, observables):
        self.constants, self.uncertain = self.split_constants(constants)
        self.kinetic_laws = kinetic_laws
        self.species_defs = species
        self.populations = populations
        self.obsfile = obsfile
        self.algorithm = algorithm
        self.conffile = conffile
        self.observables = observables

    # Note, this parser does not insist on the end of the input text.
    # Which means in theory you could have something *after* the model text,
    # which might indeed be what you are wishing for.
    grammar = (ConstantDefinition.list_grammar('constants') +
               RateDefinition.list_grammar('rates') +
               SpeciesDefinition.list_grammar('species') +
               Optional(Observable.list_grammar('observables')) +
               pyparsing.Group(system_grammar)('populations') +
               pyparsing.OneOrMore(pyparsing.Group(observe_grammar))('obsfile') +
               infalg_grammar('algorithm') +
               Optional(conf_grammar('conffile')) )
               
    whole_input_grammar = grammar + pyparsing.StringEnd()
    whole_input_grammar.ignore(pyparsing.dblSlashComment)

    @classmethod
    def from_tokens(cls, tokens):
        """ The parser action method for a ProPPA model. """
        return cls(tokens['constants'], tokens['rates'],
                   tokens['species'], tokens['populations'],
                   [t['file'] for t in tokens['obsfile']],
                   tokens['algorithm']['alg'],
                   tokens['conffile']['file'] if 'conffile' in tokens else None,
                   tokens['observables'] if 'observables' in tokens else [])
    
    @staticmethod
    def split_constants(constants):
        fixed = []
        uncertain = []
        for c in constants:
            if isinstance(c.rhs,DistributionTerm):
                uncertain.append(c)
            else:
                fixed.append(c)
        return fixed,uncertain
    
    @staticmethod
    def remove_rate_laws(expression, multipliers):
        """ Given an expression we remove calls to the rate laws methods.
            Currently this only includes the fMA method, or law of mass action.
            This means that within a given expression whenever the
            sub-expression fMA(e) appears it is replaced by (e * R1 .. * Rn)
            where R1 to Rn are the reactants and activators of the given
            reaction.
        """
        if not expression.arguments:
            return expression
        arguments = [ParsedModel.remove_rate_laws(arg, multipliers)
                     for arg in expression.arguments]
        if expression.name and expression.name == "fMA":
            # TODO: If there are no reactants? I think just the rate
            # expression, which is what this does.
            assert len(arguments) == 1
            result_expr = arguments[0]
            for (species, stoich) in multipliers:
                species_expr = Expression.name_expression(species)
                if stoich != 1:
                    # If the stoichiometry is not 1, then we have to raise the
                    # speices to the power of the stoichiometry. So if we have
                    # fMA(1.0), on a reaction X + Y -> ..., where X has
                    # stoichiometry 2, then we get fMA(1.0) = X^2 * Y * 1.0
                    stoich_expr = Expression.num_expression(stoich)
                    species_expr = Expression.power(species_expr, stoich_expr)
                result_expr = Expression.multiply(result_expr, species_expr)
            return result_expr
        else:
            # So we return a new expression with the new arguments. If we were
            # doing this inplace, we could just replace the original
            # expression's arguments.
            return Expression.apply_expression(expression.name, arguments)


    def expand_rate_laws(self):
        """ A method to expand the rate laws which are simple convenience
            functions for the user. So we wish to turn:
            kineticLawOf r : fMA(x);
            into
            kineticLawOf r : x * A * B;
            Assuming that A and B are reactants or activators for the
            reaction r
        """
        reaction_dict = self.get_reactions()
        for kinetic_law in self.kinetic_laws:
            reaction = reaction_dict[kinetic_law.lhs]
            multipliers = [(b.species, b.stoichiometry)
                           for b in reaction.reactants + reaction.activators]
            new_expr = self.remove_rate_laws(kinetic_law.rhs, multipliers)
            kinetic_law.rhs = new_expr

    def get_reactions(self):
        """ Returns a list of reactions from the parsed  model. """
        reactions = DefaultDictKey(Reaction)
        for species_def in self.species_defs:
            behaviours = species_def.rhs
            for behaviour in behaviours:
                reaction = reactions[behaviour.reaction_name]
                if behaviour.role == "<<":
                    reaction.reactants.append(behaviour)
                elif behaviour.role == ">>":
                    reaction.products.append(behaviour)
                elif behaviour.role == "(+)":
                    reaction.activators.append(behaviour)
                elif behaviour.role == "(-)":
                    reaction.inhibitors.append(behaviour)
                elif behaviour.role == "(.)":
                    reaction.modifiers.append(behaviour)
        return reactions
    
    def configure(self,sampler):
        self.config = mu.read_configuration(self.conffile)
    
    def numerize(self):
        # Gather species names in alphabetical order:
        self.species_order = [s.lhs for s in self.species_defs]
        self.species_order.sort()
        
        # Read observations from file; if there is a species order mentioned
        # there, enforce it for the rest of the model:
        if len(self.obsfile) > 1:
                raise mu.ProPPAException("""Only one observations file can be
                    provided with this solver.""")
        obs_filename = os.path.join(self.location,self.obsfile[0])
        self.obs, self.obs_order = mu.load_observations(obs_filename)
        if self.obs_order is None: # if observations don't label species
            if len(self.obs[0]) - 1 != len(self.species_order): # if some are missing
                raise mu.ProPPAException("""Only some species are observed --- 
                    I cannot figure out which ones.""")
            self.observed_species = [i for i in range(len(self.species_order))]
        else: # rearrange the observations to match alphabetical order
            rearrange = [0] + [self.obs_order.index(name)+1 for name in self.species_order
                            if name in self.obs_order]
            self.obs = [[o[index] for index in rearrange] for o in self.obs]
            self.observed_species = [i for i in range(len(self.species_order))
                                        if self.species_order[i] in self.obs_order]
        # Updates (stoichiometry matrix):
        self.updates,self.react_order = mu.get_updates(self,self.species_order)
        # Initial state:
        d = dict([(p.species_name,p.amount.number) for p in self.populations])
        self.init_state = tuple(d[s_name] for s_name in self.species_order)
        # Reorder reactions too:
        d = dict([(r.lhs,r) for r in self.kinetic_laws])
        self.kinetic_laws = [d[r_name] for r_name in self.react_order]
#        self.kinetic_laws.sort(key=lambda r: self.react_order.index(r.lhs))
        # Concrete parameters:
        self.concrete = [(c.lhs,c.rhs.get_value()) for c in self.constants]
        #should maybe change this in case of references to other variables?       

    def numerize_enhanced(self):
        # Gather species names in alphabetical order:
        self.species_order = [s.lhs for s in self.species_defs]
        self.species_order.sort()
        # Read observations from file; if there is a species order mentioned
        # there, enforce it for the rest of the model:
        self.obs = []
        for file in self.obsfile:
            ### Here we are making a big assumption! (which should be checked)
            #   that the species/observables are the same in every file, and in
            #   the same order.
            #TODO enforce this check, or at least provide a warning
            obs_filename = os.path.join(self.location,file)
            exp_obs, self.obs_order = mu.load_observations(obs_filename)
            if self.obs_order is None: # if observations don't label species
                if len(exp_obs[0]) - 1 != len(self.species_order): # if some are missing
                    raise mu.ProPPAException("""Only some species are observed --- 
                        I cannot figure out which ones.""")
                self.species_mapping = [(i,i) for i in range(len(self.species_order))]
                self.obs_names = []
                self.observed_species = [i for i in range(len(self.species_order))]
            else:
                self.species_mapping,self.obs_mapping =  mu.split_indices(
                                                                self.obs_order,
                                                                self.species_order)
                self.observed_species = [self.species_order[i] for (i,v) in self.species_mapping]
            self.obs.append(exp_obs)
        # Updates (stoichiometry matrix):
        self.updates,self.react_order = mu.get_updates(self,self.species_order)
        # Initial state:
        d = {p.species_name : p.amount.number for p in self.populations}
        self.init_state = tuple(d[s_name] for s_name in self.species_order)
        # Reorder reactions too:
        d = {r.lhs : r for r in self.kinetic_laws}
        self.kinetic_laws = [d[r_name] for r_name in self.react_order]
#        self.kinetic_laws.sort(key=lambda r: self.react_order.index(r.lhs))
        # Concrete parameters:
        self.concrete = [(c.lhs,c.rhs.get_value()) for c in self.constants]
        #should maybe change this in case of references to other variables?       
    
    def observation_mapping(self):
        n = len(self.obs_order)
        mapping = [None] * n
        # map the species components to the right element of the state
        for (i,index) in self.species_mapping:
            mapping[i] = lambda p,index=index : (lambda s : s[index])
        obs_evaluator = self.get_observables()
        for (i,name) in self.obs_mapping:
            mapping[i] = obs_evaluator[name]
        return mapping
            

    def reaction_functions(self):
        return self.reaction_functions4()
    
    def reaction_functions5(self):
        """Using the reduce_expr() method of Expression objects.
           Slower than reaction_functions3/4."""
        param_names = [p.lhs for p in self.uncertain]
        
        def apply_state(expr,state):
            env = {name : Expression.num_expression(value)
                        for name,value in zip(self.species_order,state)}
            return expr.reduce_expr(env).number
        
        def apply_parameters(expr,params):
            env = {name : Expression.num_expression(value)
                        for name,value in zip(param_names,params)}
            return lambda state: apply_state(expr.reduce_expr(env),state)

        return [lambda p,rf=rf: apply_parameters(rf.rhs,p)
                        for rf in self.kinetic_laws]            
            

    def reaction_functions3(self):
        """ Much faster! """
        species_names = list(self.species_order)
        param_names = [p.lhs for p in self.uncertain]
        conc_names = [c.lhs for c in self.constants]
        conc_vals = [float(mu.as_string(c.rhs)) for c in self.constants]
        args_list = ",".join(conc_names+param_names+species_names)
        
        kinetic_funcs = []
        scope = {}
        exec("from math import floor", scope)
        for (i,r) in enumerate(self.kinetic_laws):
            exec("""def kinetic_func_{0}({1}):
                        return {2}""".format(i,args_list,mu.as_string(r.rhs)),
                    scope)
            kinetic_funcs.append(scope['kinetic_func_'+str(i)])

        def part_eval(f,part_args):
            return lambda more_args: f(*tuple(part_args+more_args))


        # For numerized models, kinetic law expressions will be sorted already
        # so we can just assume the order is correct
        # NB: the f=f named argument part is necessary to avoid problems with
        # closures, so for the time being it stays even though it's ugly/weird
        return [lambda p,f=f: part_eval(f,conc_vals+p) for f in kinetic_funcs]

    def reaction_functions4(self):
        """ For solvers which definitely use numpy arrays """
        species_names = list(self.species_order)
        param_names = [p.lhs for p in self.uncertain]
        conc_names = [c[0] for c in self.concrete]
        conc_vals = [c[1] for c in self.concrete]
#        if len(self.concrete) > 0:
#            conc_names, conc_vals = zip(*self.concrete)
#        else:
#            conc_names = conc_vals = []
        args_list = ",".join(conc_names+param_names+species_names)
        
        kinetic_funcs = []
        scope = {}
        exec("from math import floor", scope)
        for (i,r) in enumerate(self.kinetic_laws):
            exec("""def kinetic_func_{0}({1}):
                        return {2}""".format(i,args_list,mu.as_string(r.rhs)),
                    scope)
            kinetic_funcs.append(scope['kinetic_func_'+str(i)])

        def part_eval(f,part_args):
            #return lambda more_args: f(*tuple(numpy.hstack((part_args,more_args))))
            return lambda more_args: f(*tuple(list(part_args) + list(more_args)))

        # For numerized models, kinetic law expressions will be sorted already
        # so we can just assume the order is correct
        # NB: the f=f named argument part is necessary to avoid problems with
        # closures, so for the time being it stays even though it's ugly/weird
        return [lambda p,f=f: part_eval(f,list(conc_vals)+list(p)) for f in kinetic_funcs]
    
    
    def reaction_functions2(self):
        """ Slower than reaction_functions (at least as written) """
        class FormatDict(dict):
            def __missing__(self,key):
                return '{' + key + '}'

        species_names = self.species_order
        param_names = [p.lhs for p in self.uncertain]
        
        def text_eval_part(s,names,values):
            return s.format_map(FormatDict(zip(names,values)))
#            return s.format(**FormatDict(zip(names,values)))
#            for (n,v) in zip(names,values):
#                s = s.replace('{' + n +'}',v)
#            return s
        
        def text_eval_full(s,names,values):
            return eval(s.format(**dict(zip(names,values))))
         
        def make_func_of_state(t,param_values):
            t2 = text_eval_part(t,param_names,param_values)
            return lambda state: text_eval_full(t2,species_names,state)
            
        def make_func_of_param(expr):
            t = mu.as_string2(expr)
            return lambda param_values: make_func_of_state(t,param_values)

        # For numerized models, kinetic law expressions will be sorted already
        # so we can just assume the order is correct
        return [make_func_of_param(e.rhs) for e in self.kinetic_laws]


    def reaction_functions1(self,order=None):
        """Constructs a list of functions that evaluate the kinetic laws.
            Returns a list of functions, each of which takes a parameter vector
            as input. The result of that call is another function which takes
            a state vector as input, and returns the rate of that reaction for
            the given parameterisation and state.
            This makes sense on a deep level. Very deep.
        """
        species_names = self.species_order
        if order is None:
            order = self.react_order
        param_names = [p.lhs for p in self.uncertain]
        def apply_params(params,expr):
            env = {}
            for i in range(len(params)):
                env[param_names[i]] = Expression.num_expression(params[i])
            def f(state):
                def apply_state_inner(state,species_names,expr):
                    #env2 = {} # is it better to have a second dictionary?
                    for i in range(len(state)):
                        #env2[species_names[i]] = proppa.Expression.num_expression(state[i])
                        env[species_names[i]] = \
                                Expression.num_expression(state[i])
                    #return expr.get_value(env2)
                    return expr.get_value(env)
                #return apply_state_inner(state,species_names,expr.reduce_expr(env))
                return apply_state_inner(state,species_names,expr)
            return f
            
        def abstract_function(expr):
            return lambda params: apply_params(params,expr)
        r_dict = dict([(e.lhs,abstract_function(e.rhs))
                        for e in self.kinetic_laws])
        return [r_dict[r_name] for r_name in order]
        #return [abstract_function(e.rhs) for e in self.kinetic_laws]
    
    def derivative_functions(self):
        """Similar to the above but returns the derivatives of the rate laws
            with respect to each species.
            TODO: Change to use the strategy in reactions_functions3
        """
        species_names = self.species_order
        react_names = self.react_order
        
        param_names = [p.lhs for p in self.uncertain]
        def apply_params(params,expr):
            env = {}
            for i in range(len(params)):
                env[param_names[i]] = Expression.num_expression(params[i])
            def f(state):
                def apply_state_inner(state,species_names,expr):
                    #env2 = {} # is it better to have a second dictionary?
                    for i in range(len(state)):
                        #env2[species_names[i]] = proppa.Expression.num_expression(state[i])
                        env[species_names[i]] = \
                                Expression.num_expression(state[i])
                    #return expr.get_value(env2)
                    return expr.get_value(env)
                #return apply_state_inner(state,species_names,expr.reduce_expr(env))
                return apply_state_inner(state,species_names,expr)
            return f
            
        def abstract_function(expr):
            return lambda params: apply_params(params,expr)
        r_dict = dict([(e.lhs,e.rhs)
                        for e in self.kinetic_laws])
#        function_list = []
#        for r_name in react_names:
#            derivs = []
#            for s_name in species_names:
#                diff_expr = r_dict[r_name].differentiate(s_name)
#                derivs.append(abstract_function(diff_expr))
#            function_list.append(derivs)
#        return derivs
        return [ [abstract_function(r_dict[r].differentiate(s))
                        for s in species_names]
                                for r in react_names ]
    
    def prepare_sampler(self):
        pass
        # check requirements
        # read configuration file or use explicitly specified/default values
        # make "configuration" arguments for sampler
    
    def make_empty_configuration(sampler_class):
        conf = {}
        # things that all samplers require
        conf['priors'] = []
        conf['rate_funcs'] = []
        # specific instances:
        for c in sampler_class.required_conf:
            conf[c] = []
        return conf
    
    def apply_configuration_file(self,conf,sampler):
        #conf['n_samples'] = 100
        #for pc in conf['parameters']:
        #    pc['proposal'] = lambda x: scipy.stats.norm(loc=x,scale=0.02)
        conf['observed_species'] = self.observed_species
#        if self.conffile is None:
#            print("No configuration file provided. Using default configuration.")
#            return
        file_conf = (mu.read_config(self) if self.conffile is not None
                        else {'proposals': {}})
        # apply all configuration options specified (except proposals):
        for alg_par in file_conf:
            if alg_par != 'proposals':
                conf[alg_par] = file_conf[alg_par]        
        # check if any parameters have not been specified
        for alg_par in sampler.required_conf:
            if alg_par not in file_conf and alg_par != 'proposals':
                msg = ("Configuration does not specify " + alg_par + 
                        ". Using default value of " + str(conf[alg_par]) + ".")
                print(msg)
        # proposals are handled separately, if the sampler requires them
        if 'proposals' in sampler.required_conf:
            for par_def in self.uncertain:
                par = par_def.lhs
                #get index of parameter
                ind = [pc['name'] for pc in conf['parameters']].index(par)
                if par in file_conf['proposals']:
                    var = float(file_conf['proposals'][par])
                else:
                    var = 0.01
                    msg = ("Configuration does not specify proposal for " +
                           par + ". Using default variance (0.01).")
                    print(msg)
                #using var=var because of how closures work (cf reaction funcs)
                prop_func = lambda x,var=var: sp.stats.norm(loc=x,scale=var)
                #print("Proposal for " + par + " (" + str(ind) + "):", var)
                conf['parameters'][ind]['proposal'] = prop_func
    
    def infer(self,n_samples=1000):
        # prepare sampler
        try:
            sampler_type = samplers.get_sampler(self.algorithm)
        except mu.ProPPAException as ex:
            print("Could not complete sampling. Problem:")
            print(ex)
            return None
        if sampler_type.supports_enhanced:
            self.numerize_enhanced()
        else:
            self.numerize()        
        
        # get a basic configuration for the chosen sampler...
        conf = sampler_type.prepare_conf(self)
        # ...and now apply any configuration parameters the use has specified
        self.apply_configuration_file(conf,sampler_type)
        # set the number of samples, if specified
        if 'n_samples' in conf:
            n_samples = conf['n_samples']
        sampler = sampler_type(self,conf)
        # take samples
        print('Will take %d samples...' % n_samples)
        samples = sampler.gather_samples(n_samples)
        print('Sampling finished (%d samples).' % len(samples))
        # print mean, variance
        print('Mean:    ',np.mean(samples,axis=0))
        print('Variance:',np.var(samples,axis=0))
        return samples

    def test_sampler(self,n_samples=100):
        self.numerize()
        # prepare sampler
        try:
            sampler_type = samplers.get_sampler(self.algorithm)
        except mu.ProPPAException as ex:
            print("Could not complete sampling. Problem:")
            print(ex)
            return None
        conf = sampler_type.prepare_conf(self)
        # extra parameters here (eg n_samples or other things set / overriden)
        self.apply_configuration_file(conf,sampler_type)
        #n_samples = conf['n_samples']
        sampler = sampler_type(self,conf)
        # take samples
        test_props = sampler.test_propose(n_samples)
        return (test_props,conf)

    
    def get_observables(self):
        species_names = list(self.species_order)
        param_names = [p.lhs for p in self.uncertain]
        conc_names = [c[0] for c in self.concrete]
        conc_vals = [c[1] for c in self.concrete]
#        if len(self.concrete) > 0:
#            conc_names, conc_vals = zip(*self.concrete)
#        else:
#            conc_names = conc_vals = []
        args_list = ",".join(conc_names+param_names+species_names)
        
        observable_funcs = {}
        scope = {}
        exec("from math import floor", scope)
        for (i,o) in enumerate(self.observables):
            fun_name = "observable_" + o.lhs
            exec("""def {0}({1}):
                        return {2}""".format(fun_name,args_list,
                                            mu.as_string(o.rhs)),scope)
            observable_funcs[o.lhs] = scope[fun_name]

        def part_eval(f,part_args):
            #return lambda more_args: f(*tuple(numpy.hstack((part_args,more_args))))
            return lambda more_args: f(*tuple(list(part_args) + list(more_args)))

        # For numerized models, kinetic law expressions will be sorted already
        # so we can just assume the order is correct
        # NB: the f=f named argument part is necessary to avoid problems with
        # closures, so for the time being it stays even though it's ugly/weird
        return {f : (lambda p,f=d: part_eval(f,list(conc_vals)+list(p)))
                              for (f,d) in observable_funcs.items()}
    
    def concretise(self,assignment):
        print("""Warning: concretising is irreversible. You may need to reload
                 the model for any future experiments""")
        for name in assignment:
            # if name is a concrete parameter, just change its value:
            if name in self.concrete:
                ind = self.concrete.index(name)
                self.concrete[ind][1] = assignment[name]
            else: # we are setting an uncertain parameter
                try:
                    ind = [u.lhs for u in self.uncertain].index(name)
                except ValueError:
                    msg = "There is no parameter named " + name + "."
                    raise mu.ProPPAException(msg)
                self.uncertain.pop(ind) #remove it from the list of uncertain pars
                self.concrete.append((name,assignment[name]))
        return

ParsedModel.grammar.setParseAction(ParsedModel.from_tokens)

def parse_model(model_string):
    """Parses a model ensuring that we have consumed the entire input"""
    return ParsedModel.whole_input_grammar.parseString(model_string)[0]

def load_model(filename):
    with open(filename,"r") as modelfile:
        try:
            model = parse_model(modelfile.read())
            # set the relative location of the model, so that the observation
            # and configuration files are specified relative to that
            model.location = os.path.dirname(filename)
        except pyparsing.ParseException as pe:
            print("Could not parse model:")
            print(pe)
            return
    return model

def analyse_proppa_file(filename,arguments):
    """ The main interface method for the analysing of ProPPA files. """
    with open(filename, "r") as modelfile:
        try:
            model = parse_model(modelfile.read())
        except pyparsing.ParseException as pe:
            print("Could not parse model:")
            print(pe)
            return
    print('Read file.')
    sampler = mu.setup_sampler(model)
    model.configure(sampler)
    stoich = mu.get_updates(model,[s.lhs for s in model.species_defs])
    print(stoich)
    obs,names = mu.load_observations(model.obsfile)
    # and then other things we don't care about right now
    return
            

if __name__ == "__main__":
    #import sys
    arguments = docopt(__doc__)

    model_name = arguments['<model_file>']
    model = load_model(model_name)
    if model is not None:
        samples = model.infer()
        out_name = (model_name + '_out' if not arguments['--out']
                            else arguments['--out'])
        mu.write_results(samples,out_name)