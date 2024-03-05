import pickle


def main():
    tool = 'Wendigo'
    tool2 = 'EvoMaster'
    app = 'DVGA'
    agent1 = 'Random-Greedy-Regular'
    agent2 = 'Random-Regular'
    agent3 = 'PPO-Regular'
    agent4 = 'Black-Box-Regular'
    num_steps = 1280
    combined_steps1 = 1280
    combined_steps2 = 1280
    combined_steps3 = 1280
    combined_steps4 = 1280

    results1 = pickle.load(open('paper-results/Regular/' + tool + '-' + app + '-' + agent1 + '-Step-'
                                + str(combined_steps1) + '-combined.p', 'rb'))

    results2 = pickle.load(open('paper-results/Regular/' + tool + '-' + app + '-' + agent2 + '-Step-'
                                + str(combined_steps2) + '-combined.p', 'rb'))

    results3 = pickle.load(open('paper-results/Regular/' + tool + '-' + app + '-' + agent3 + '-Step-'
                                + str(combined_steps3) + '-combined.p', 'rb'))

    results4 = pickle.load(open('paper-results/Regular/' + tool2 + '-' + app + '-' + agent4 + '-Step-'
                                + str(combined_steps4) + '-combined.p', 'rb'))

    r1_filtered = []
    r2_filtered = []
    r3_filtered = []
    r4_filtered = []

    for i in range(num_steps):
        if not (results1[i][2] == 0 or results1[i][3]):
            r1_filtered += [results1[i]]

        if not (results2[i][2] == 0 or results2[i][3]):
            r2_filtered += [results2[i]]

        if not (results3[i][2] == 0 or results3[i][3]):
            r3_filtered += [results3[i]]

        if not (results4[i][2] == 0 or results4[i][3]):
            r4_filtered += [results4[i]]

    print('stop')


if __name__ == '__main__':
    main()