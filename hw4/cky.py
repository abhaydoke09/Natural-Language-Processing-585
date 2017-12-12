from pprint import pprint

# The productions rules have to be binarized.

# grammar_text = """
# S -> NP VP
# NP -> Det Noun
# VP -> Verb NP
# PP -> Prep NP
# NP -> NP PP
# VP -> VP PP
# """
grammar_text = """
S -> NPZ VP
S -> NP VBZ
NP -> Det Noun
NPZ -> Det Nouns
VP -> Verb NP
VBZ -> Verbs NP
PP -> Prep NP
NP -> NP PP
VP -> VP PP
"""

lexicon = {
    'Nouns': set(['cats', 'dogs']),
    'Verbs': set(['attacks', 'attacked']),
    'Noun': set(['cat', 'dog', 'table', 'food']),
    'Verb': set(['saw', 'loved', 'hated', 'attack']),
    'Prep': set(['in', 'of', 'on', 'with']),
    'Det': set(['the', 'a']),
}

# Process the grammar rules.  You should not have to change this.
grammar_rules = []

for line in grammar_text.strip().split("\n"):
    if not line.strip(): continue
    left, right = line.split("->")
    left = left.strip()
    children = right.split()
    rule = (left, tuple(children))
    grammar_rules.append(rule)

grammar_dict = {}
for rule in grammar_rules:
    grammar_dict[rule[1]] = rule[0]

possible_parents_for_children = {}
for parent, (leftchild, rightchild) in grammar_rules:
    if (leftchild, rightchild) not in possible_parents_for_children:
        possible_parents_for_children[leftchild, rightchild] = []
    possible_parents_for_children[leftchild, rightchild].append(parent)
# Error checking
all_parents = set(x[0] for x in grammar_rules) | set(lexicon.keys())
for par, (leftchild, rightchild) in grammar_rules:
    if leftchild not in all_parents:
        assert False, "Nonterminal %s does not appear as parent of prod rule, nor in lexicon." % leftchild
    if rightchild not in all_parents:
        assert False, "Nonterminal %s does not appear as parent of prod rule, nor in lexicon." % rightchild

# print "Grammar rules in tuple form:"
# pprint(grammar_rules)
# print "Rule parents indexed by children:"
# pprint(possible_parents_for_children)


def cky_acceptance(sentence):
    # return True or False depending whether the sentence is parseable by the grammar.
    global grammar_rules, lexicon, grammar_dict

    # Set up the cells data structure.
    # It is intended that the cell indexed by (i,j)
    # refers to the span, in python notation, sentence[i:j],
    # which is start-inclusive, end-exclusive, which means it includes tokens
    # at indexes i, i+1, ... j-1.
    # So sentence[3:4] is the 3rd word, and sentence[3:6] is a 3-length phrase,
    # at indexes 3, 4, and 5.
    # Each cell would then contain a list of possible nonterminal symbols for that span.
    # If you want, feel free to use a totally different data structure.
    N = len(sentence)
    cells = {}
    for i in range(N):
        for j in range(i + 1, N + 1):
            cells[(i, j)] = []

    for i,word in enumerate(sentence):
        for lex in lexicon:
            if word in lexicon[lex]:
                cells[(i,i+1)].append(lex)

    # for l in range(1,N):
    #     for s in range(N-l):
    #         for p in range(l-1):
    #             pprint((p,s))
    #     print ""
    #pprint(cells)
    path_dict = {}
    for j in range(1,N):
        for i in range(N-j):
            #pprint((i, i+j))
            print("Updating for", (i, i+j+1))
            possible_left_positions = []
            possible_down_positions = []
            # for cellJ in range(i, i+j):
            #     possible_left_positions.append((i, cellJ+1))
            # for cellI in range(i+1,i+j+1):
            #     possible_down_positions.append((cellI, i+j+1))
            #

            for partition in range(i+j-i):
                left_cell = (i, i+j-partition)
                down_cell = (i+j-partition, i+j+1)
                print left_cell, down_cell
                for first_symbol in cells[left_cell]:
                    for second_symbol in cells[down_cell]:
                        #pprint((first_symbol, second_symbol))
                        if (first_symbol, second_symbol) in grammar_dict:
                            #print("Rule found")
                            # pprint(grammar_dict[(first_symbol, second_symbol)])
                            cells[(i,i+j+1)].append(grammar_dict[(first_symbol, second_symbol)])
                            path_dict[(i, i+j+1, grammar_dict[(first_symbol, second_symbol)])] = \
                                    ((left_cell[0],left_cell[1],first_symbol), (down_cell[0],down_cell[1],second_symbol))


            # for left_cell in possible_left_positions:
            #     for down_cell in possible_down_positions:
            #         for first_symbol in cells[left_cell]:
            #             for second_symbol in cells[down_cell]:
            #                 #pprint((first_symbol, second_symbol))
            #                 if (first_symbol, second_symbol) in grammar_dict:
            #                     #print("Rule found")
            #                     # pprint(grammar_dict[(first_symbol, second_symbol)])
            #                     cells[(i,i+j+1)].append(grammar_dict[(first_symbol, second_symbol)])
            #                     path_dict[(i, i+j+1, grammar_dict[(first_symbol, second_symbol)])] = \
            #                             ((left_cell[0],left_cell[1],first_symbol), (down_cell[0],down_cell[1],second_symbol))



    if (0, N, 'S') in path_dict:
        print(path_dict[(0, N, 'S')])
    print(cells)
    # TODO replace the below with an implementation
    return 'S' in cells[(0,N)]

def get_parsed_path(node, path_dict, N, sentence):
    if node not in path_dict:
        return [node[2], sentence[node[0]]]
    return [node[2], [get_parsed_path(path_dict[node][0], path_dict, N, sentence),get_parsed_path(path_dict[node][1], path_dict, N, sentence)]]


def cky_parse(sentence):
    # Return one of the legal parses for the sentence.
    # If nothing is legal, return None.
    # This will be similar to cky_acceptance(), except with backpointers.
    global grammar_rules, lexicon

    N = len(sentence)
    cells = {}
    for i in range(N):
        for j in range(i + 1, N + 1):
            cells[(i, j)] = []

    for i, word in enumerate(sentence):
        for lex in lexicon:
            if word in lexicon[lex]:
                cells[(i, i + 1)].append(lex)

    # for l in range(1,N):
    #     for s in range(N-l):
    #         for p in range(l-1):
    #             pprint((p,s))
    #     print ""
    # pprint(cells)
    path_dict = {}
    for j in range(1, N):
        for i in range(N - j):
            # pprint((i, i+j))
            # print("Updating for", (i, i+j+1))
            possible_left_positions = []
            possible_down_positions = []
            for cellJ in range(i, i + j):
                possible_left_positions.append((i, cellJ + 1))
            for cellI in range(i + 1, i + j + 1):
                possible_down_positions.append((cellI, i + j + 1))

            for left_cell in possible_left_positions:
                for down_cell in possible_down_positions:
                    for first_symbol in cells[left_cell]:
                        for second_symbol in cells[down_cell]:
                            # pprint((first_symbol, second_symbol))
                            if (first_symbol, second_symbol) in grammar_dict:
                                # print("Rule found")
                                # pprint(grammar_dict[(first_symbol, second_symbol)])
                                cells[(i, i + j + 1)].append(grammar_dict[(first_symbol, second_symbol)])
                                path_dict[(i, i + j + 1, grammar_dict[(first_symbol, second_symbol)])] = \
                                    ((left_cell[0], left_cell[1], first_symbol),
                                     (down_cell[0], down_cell[1], second_symbol))

    if (0, N, 'S') in path_dict:
        print get_parsed_path((0, N, 'S'), path_dict, N, sentence)

    # TODO replace the below with an implementation
    return 'S' in cells[(0, N)]


## some examples of calling these things...
## you probably want to call only one sentence at a time to help debug more easily.

# print cky_acceptance(['the','cat','attacked','the','food'])
# pprint( cky_parse(['the','cat','attacked','the','food']))
# pprint( cky_acceptance(['the','the']))
# pprint( cky_parse(['the','the']))
# print cky_acceptance(['the','cat','attacked','the','food','with','a','dog'])
# pprint( cky_parse(['the','cat','attacked','the','food','with','a','dog']) )
# pprint( cky_parse(['the','cat','with','a','table','attacked','the','food']) )
#
