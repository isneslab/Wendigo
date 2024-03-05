import pickle


def main():
    num_steps = 1280
    steps = []

    tool = 'Wendigo'
    app = 'DVGA'
    agent = 'PPO-Regular'
    file_name = tool + '-' + app + '-' + agent + '-Step'

    for i in range(num_steps):
        steps.append(pickle.load(open(file_name + '/' + file_name + '-' + str(i+1) + '.p', 'rb')))

    with open(file_name + '/' + file_name + '-' + str(num_steps) + '-combined.p', 'wb') as file:
        pickle.dump(steps, file)


if __name__ == '__main__':
    main()
