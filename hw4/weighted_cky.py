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
        probabilities[(left, tuple(children))] = probability


    #print "Grammar rules in tuple form:"
    #pprint(grammar_rules)
    # print "Rule parents indexed by children:"
    # pprint(possible_parents_for_children)
    # print "probabilities"
    # pprint(probabilities)
    #print "Lexicon"
    #pprint(lexicon)

def get_parsed_path(node, path_dict, N, sentence):
    if node not in path_dict:
        return [node[2], sentence[node[0]]]
    return [node[2], [get_parsed_path(path_dict[node][0], path_dict, N, sentence),get_parsed_path(path_dict[node][1], path_dict, N, sentence)]]


def pcky_parse(sentence):
    # Return the most probable legal parse for the sentence
    # If nothing is legal, return None.
    # This will be similar to cky_parse(), except with probabilities.
    populate_grammar_rules()
    #pprint(sentence)
    global grammar_rules, lexicon, probabilities, possible_parents_for_children
    grammar_dict = {}

    #pprint(grammar_rules)
    for rule in grammar_rules:
        grammar_dict[rule[1]] = []
    for rule in grammar_rules:
        grammar_dict[rule[1]].append((rule[0], rule[2]))

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
                cells[(i, i + 1)].append((lex, probabilities[(lex, tuple([word]))]))
    #pprint(cells)
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

            for partition in range(i+j-i):
                left_cell = (i, i+j-partition)
                down_cell = (i+j-partition, i+j+1)
                #print left_cell, down_cell
                for first_symbol in cells[left_cell]:
                    for second_symbol in cells[down_cell]:
                            l = first_symbol[0]
                            r = second_symbol[0]
                            if (l,r) in grammar_dict:
                                # print("Rule found")
                                # pprint(grammar_dict[(first_symbol, second_symbol)])
                                #cells[(i, i + j + 1)].append(grammar_dict[(first_symbol, second_symbol)])
                                for r in grammar_dict[(l,r)]:
                                    cells[(i, i + j + 1)].append((r[0], first_symbol[1]*second_symbol[1]*r[1]))
                                    path_dict[(i, i + j + 1, r[0], first_symbol[1]*second_symbol[1]*r[1])] = \
                                        ((left_cell[0], left_cell[1], first_symbol[0], first_symbol[1]),
                                         (down_cell[0], down_cell[1], second_symbol[0], second_symbol[1]))

    #pprint(path_dict)
    maximum_prob_val = -100.0
    target_pattern = None
    for key in path_dict:
        if key[0]==0 and key[1]==N:
            if key[2] == 'S':
                if key[3] > maximum_prob_val:
                    maximum_prob_val = key[3]
                    target_pattern = key
    #print target_pattern
    pprint("Sentence: "+ " ".join(sentence))
    if maximum_prob_val == -100:
        pprint("Not a valid parse tree for this sentence")
        return False
    pprint("Parse tree with max probability of "+ str(maximum_prob_val))
    print get_parsed_path(target_pattern, path_dict, N, sentence)

    # TODO replace the below with an implementation


    # TODO complete the implementation
    return True

