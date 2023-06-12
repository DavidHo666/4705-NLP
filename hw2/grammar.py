"""
COMS W4705 - Natural Language Processing - Summer 2022 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)
        if self.verify_grammar():
            print("Success! The grammar is a valid PCFG in CNF")
        else:
            print("Fail! The grammar is NOT a valid PCFG in CNF")
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        import string
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        nonterminals = set()
        for symbol, rules in self.lhs_to_rules.items():
            nonterminals.add(symbol)
            prob = 0
            for rule in rules:
                prob += rule[-1]
            if not math.isclose(prob, 1):
                return False

        for nonterminal in nonterminals:
            if not nonterminal.isupper():
                return False

        for symbol, rules in self.lhs_to_rules.items():
            for rule in rules:
                if len(rule[1]) == 2 and not all(x in nonterminals for x in rule[1]):
                    print(rule[1])
                    return False
                elif len(rule[1]) == 1 and not (rule[1][0].islower() or
                                                rule[1][0].isnumeric() or
                                                (rule[1][0] in string.punctuation) or
                                                (rule[1][0].upper in nonterminals)):
                    return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)

    tmp = 1
