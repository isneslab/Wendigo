import pickle


def main():
    tool = 'EvoMaster'
    app = 'DVGA'
    agent = 'Black-Box-Regular'
    num_steps = 1280
    combined_steps = 1280
    file_name = tool + '-' + app + '-' + agent + '-Step'

    results = pickle.load(open('paper-results/Regular/'
                               + file_name + '-' + str(combined_steps) + '-combined.p', 'rb'))

    q_skipped = 0
    q_rejected = 0
    q_max = 0

    for i in range(0, num_steps):
        if results[i][2] == 0 and not results[i][3]:
            q_skipped += 1
        elif results[i][3]:
            q_rejected += 1
        elif results[i][2] > q_max:
            q_max = results[i][2]

    print('Skipped: ' + str(q_skipped))
    print('Rejected: ' + str(q_rejected))
    print('Max: ' + str(q_max))


if __name__ == '__main__':
    main()
