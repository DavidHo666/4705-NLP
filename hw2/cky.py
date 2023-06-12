"""
COMS W4705 - Natural Language Processing - Summer 2022
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        pi = defaultdict(lambda: defaultdict(tuple)) # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        n = len(tokens)
        # initialization
        for i in range(n):
            for rule in self.grammar.rhs_to_rules[(tokens[i], )]:
                pi[i, i+1][rule[0]] = tokens[i]

        # main loop
        for length in range(2, n+1):
            for i in range(0, n-length+1):
                j = i + length
                for k in range(i+1, j):
                    for b in pi[i, k]:
                        for c in pi[k, j]:
                            if self.grammar.rhs_to_rules[b, c]:
                                for rule in self.grammar.rhs_to_rules[b, c]:
                                    pi[i, j][rule[0]] = ((b, i, k), (c, k, j))

        if pi[0, n][self.grammar.startsymbol]:
            return True
        else:
            return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table= defaultdict(lambda: defaultdict(tuple))
        probs = defaultdict(lambda: defaultdict(lambda: float('-inf'))) # https://stackoverflow.com/questions/30356892/defaultdict-with-default-value-1
        n = len(tokens)

        # initialization
        for i in range(n):
            for rule in self.grammar.rhs_to_rules[(tokens[i], )]:
                table[i, i+1][rule[0]] = tokens[i]
                probs[i ,i+1][rule[0]] = math.log(rule[-1])

        # main loop
        for length in range(2, n+1):
            for i in range(0, n-length+1):
                j = i + length
                for k in range(i+1, j):
                    for b in table[i, k]:
                        for c in table[k, j]:
                            if self.grammar.rhs_to_rules[b, c]:
                                for rule in self.grammar.rhs_to_rules[b, c]:
                                    if math.log(rule[-1]) + probs[i, k][b] + probs[k, j][c] > probs[i, j][rule[0]]:
                                        probs[i, j][rule[0]] = math.log(rule[-1]) + probs[i, k][b] + probs[k, j][c]
                                        table[i, j][rule[0]] = ((b, i, k), (c, k, j))

        return dict(table), probs


def get_tree(chart, i,j,nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if isinstance(chart[i, j][nt], str):
        return (nt, chart[i, j][nt])

    left = chart[i, j][nt][0]
    right = chart[i, j][nt][1]
    return (nt, get_tree(chart, left[1], left[2], left[0]), get_tree(chart, right[1], right[2], right[0]))



if __name__ == "__main__":

    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        # toks = ['miami', 'flights', 'cleveland', 'from', 'to', '.']
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        print(get_tree(table, 0, len(toks), grammar.startsymbol))
        #assert check_table_format(chart)
        #assert check_probs_format(probs)

