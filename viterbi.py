def max_helper(states, state_probabilities_with_prev, t, state, transition_probabilities, max_tracked_probability,
               prev_st_selected):
    for previous_state in states[1:]:
        tracked_probability = state_probabilities_with_prev[t - 1][previous_state]["probability"] * \
                              transition_probabilities[previous_state][state]
        if tracked_probability > max_tracked_probability:
            max_tracked_probability = tracked_probability
            prev_st_selected = previous_state
    return max_tracked_probability, prev_st_selected


def viterbi_helper(observations, states, initial_probabilities, transition_probabilities, observation_probabilities):
    state_probabilities_with_prev = [{}]
    for state in states:
        state_probabilities_with_prev[0][state] = {
            "probability": initial_probabilities[state] * observation_probabilities[state][observations[0]],
            "previous_state": None}

    # Forward Pass
    N = len(observations)

    for t in range(1, N):
        state_probabilities_with_prev.append({})
        for state in states:
            max_tracked_probability = state_probabilities_with_prev[t - 1][states[0]]["probability"] * \
                                      transition_probabilities[states[0]][state]
            prev_st_selected = states[0]

            argmax = max_helper(states, state_probabilities_with_prev, t, state, transition_probabilities,
                                max_tracked_probability, prev_st_selected)

            max_prob = argmax[0] * observation_probabilities[state][observations[t]]
            state_probabilities_with_prev[t][state] = {"probability": max_prob, "previous_state": argmax[1]}

    state_sequency = []
    sequence_probability = 0.0
    b_state = None

    # Find the most likely path’s ending point

    for state, data in state_probabilities_with_prev[-1].items():
        if data["probability"] > sequence_probability:
            sequence_probability = data["probability"]
            b_state = state
    state_sequency.append(b_state)
    previous = b_state

    # Work backwards through our most likely path and find the hidden states

    for t in range(len(state_probabilities_with_prev) - 2, -1, -1):
        state_sequency.insert(0, state_probabilities_with_prev[t + 1][previous]["previous_state"])
        previous = state_probabilities_with_prev[t + 1][previous]["previous_state"]

    # Connecting the states' values

    state_probabilities = {}
    for i in state_probabilities_with_prev:
        for j in i:
            if j in state_probabilities.keys():
                state_probabilities[j].append(i[j]['probability'])
            else:
                state_probabilities[j] = []
                state_probabilities[j].append(i[j]['probability'])

    return state_sequency, sequence_probability, state_probabilities


def viterbi(problem_file_name):
    with open(problem_file_name) as f:
        lines = f.read().splitlines()
    lines.pop(0)  # states
    states = lines.pop(0).split("|")
    lines.pop(0)  # start probabilities
    initial_probabilities_inp = lines.pop(0).split("|")
    initial_probabilities = {}
    for value in initial_probabilities_inp:
        value = value.split(":")
        initial_probabilities[str(value[0])] = float(value[1])
    lines.pop(0)  # transition probabilities
    transition_probabilities_inp = lines.pop(0).split("|")
    transition_probabilities = {}
    for value in transition_probabilities_inp:
        value = value.split(":")
        val = float(value[1])
        st1 = value[0].split("-")[0]
        st2 = value[0].split("-")[1]
        if st1 in transition_probabilities.keys():
            transition_probabilities[st1][st2] = val
        else:
            transition_probabilities[st1] = {}
            transition_probabilities[st1][st2] = val
    lines.pop(0)  # observation probabilities
    observation_probabilities_inp = lines.pop(0).split("|")
    observation_probabilities = {}
    for value in observation_probabilities_inp:
        value = value.split(":")
        val = float(value[1])
        st1 = value[0].split("-")[0]
        st2 = value[0].split("-")[1]
        if st1 in observation_probabilities.keys():
            observation_probabilities[st1][st2] = val
        else:
            observation_probabilities[st1] = {}
            observation_probabilities[st1][st2] = val
    lines.pop(0)  # observations
    observations = lines.pop(0).split("|")
    return viterbi_helper(observations, states, initial_probabilities, transition_probabilities,
                           observation_probabilities)

