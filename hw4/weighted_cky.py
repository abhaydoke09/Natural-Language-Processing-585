from pprint import pprint

grammar_rules = []
lexicon = {}
probabilities = {}
possible_parents_for_children = {}


def populate_grammar_rules():
    global grammar_rules, lexicon, probabilities, possible_parents_for_children
    # TODO Fill in your implementation for processing the grammar rules.
    f = open('pcfg_grammar_modified','rb')
    grammar_text = f.readlines()
    f.close()
    grammar_rules = []
    lexicon_flag = False
    for line in grammar_text:
        line = line.strip()

        if not line.strip(): continue
        if line == "##":
            lexicon_flag = True
            continue

        left, right = line.split("->")
        left = left.strip()
        if lexicon_flag:
            children = right.split()[:1]
            if children[0] != "":
                if left not in lexicon:
                    lexicon[left.strip()] = set([children[0].strip()])
                else:
                    lexicon[left.strip()].add(children[0].strip())
        else:
            children = right.split()[:2]
        probability = float(right.split()[-1])

        rule = (left, tuple(children), probability)
        grammar_rules.append(rule)
        possible_parents_for_children[tuple(children)] = left
        probabilities[tuple(children)] = probability


    #print "Grammar rules in tuple form:"
    #pprint(grammar_rules)
    # print "Rule parents indexed by children:"
    # pprint(possible_parents_for_children)
    # print "probabilities"
    # pprint(probabilities)
    #print "Lexicon"
    #pprint(lexicon)



def pcky_parse(sentence):
    # Return the most probable legal parse for the sentence
    # If nothing is legal, return None.
    # This will be similar to cky_parse(), except with probabilities.
    populate_grammar_rules()
    pprint(sentence)
    global grammar_rules, lexicon, probabilities, possible_parents_for_children
    grammar_dict = {}

    #pprint(grammar_rules)
    for rule in grammar_rules:
        if rule[1] == ('Verb','NP'):
            pprint(rule)
        grammar_dict[rule[1]] = (rule[0], rule[2])

    #pprint(grammar_dict[('Verb','NP')])
    #pprint(grammar_dict)
    N = len(sentence)
    cells = {}
    for i in range(N):
        for j in range(i + 1, N + 1):
            cells[(i, j)] = []

    for i, word in enumerate(sentence):
        for lex in lexicon:
            if word in lexicon[lex]:
                cells[(i, i + 1)].append(lex)
    pprint(cells)
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

    pprint(cells)
    if (0, N, 'S') in path_dict:
        #print get_parsed_path((0, N, 'S'), path_dict, N, sentence)
        pass

    # TODO replace the below with an implementation
    return 'S' in cells[(0, N)]

    # TODO complete the implementation
    return None

