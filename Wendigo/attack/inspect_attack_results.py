import pickle
import math


def main():

    attack_results = []
    benign_results = []

    # for i in range(1222):
    #     attack_results += [pickle.load(open('results-temp/attack-' + str(i+1) + '.p', 'rb'))]
    #
    # for i in range(1254):
    #     benign_results += [pickle.load(open('results-temp/benign-' + str(i+1) + '.p', 'rb'))]
    #
    # with open('results-temp/attack_results-combined.p', 'wb') as file:
    #     pickle.dump(attack_results, file)
    #
    # with open('results-temp/benign_results-combined.p', 'wb') as file:
    #     pickle.dump(benign_results, file)

    PPO_attack_results = pickle.load(open('results/PPO_attack_results-combined.p', 'rb'))
    PPO_benign_results = pickle.load(open('results/PPO_benign_results-combined.p', 'rb'))

    PPO_attack_results_large = pickle.load(open('results/PPO_attack_results-large-combined.p', 'rb'))
    PPO_benign_results_large = pickle.load(open('results/PPO_benign_results-large-combined.p', 'rb'))

    Random_attack_results = pickle.load(open('results/Random_attack_results-combined.p', 'rb'))
    Random_benign_results = pickle.load(open('results/Random_benign_results-combined.p', 'rb'))

    Random_attack_results_large = pickle.load(open('results/Random_attack_results-large-combined.p', 'rb'))
    Random_benign_results_large = pickle.load(open('results/Random_benign_results-large-combined.p', 'rb'))

    Random_Greedy_attack_results = pickle.load(open('results/Random-Greedy_attack_results-combined.p', 'rb'))
    Random_Greedy_benign_results = pickle.load(open('results/Random-Greedy_benign_results-combined.p', 'rb'))

    Random_Greedy_attack_results_large = pickle.load(open('results/Random-Greedy_attack_results-large-combined.p', 'rb'))
    Random_Greedy_benign_results_large = pickle.load(open('results/Random-Greedy_benign_results-large-combined.p', 'rb'))

    Evo_attack_results = pickle.load(open('results/EvoMaster_attack_results-combined.p', 'rb'))
    Evo_benign_results = pickle.load(open('results/EvoMaster_benign_results-combined.p', 'rb'))

    Evo_attack_results_large = pickle.load(open('results/EvoMaster_attack_results-large-combined.p', 'rb'))
    Evo_benign_results_large = pickle.load(open('results/EvoMaster_benign_results-large-combined.p', 'rb'))

    benign_avg_results = pickle.load(open('results/benign_results-avg-combined.p', 'rb'))

    avg = sum(benign_avg_results) / len(benign_avg_results)
    std = math.sqrt(sum([(x-avg)**2 for x in benign_avg_results])/(len(benign_avg_results)-1))

    PPO_denial = sum([x-avg for x in PPO_benign_results if x > (avg + 2*std)])/sum(PPO_benign_results)
    PPO_queries = len(PPO_attack_results)

    PPO_large_denial = sum([x - avg for x in PPO_benign_results_large if x > (avg + 2 * std)]) / sum(
        PPO_benign_results_large)
    PPO_large_queries = len(PPO_attack_results_large)

    Random_denial = sum([x - avg for x in Random_benign_results if x > (avg + 2 * std)]) / sum(Random_benign_results)
    Random_queries = len(Random_attack_results)

    Random_large_denial = sum([x - avg for x in Random_benign_results_large if x > (avg + 2 * std)]) / sum(
        Random_benign_results_large)
    Random_large_queries = len(Random_attack_results_large)

    Random_Greedy_denial = sum([x - avg for x in Random_Greedy_benign_results if x > (avg + 2 * std)]) / sum(
        Random_Greedy_benign_results)
    Random_Greedy_queries = len(Random_Greedy_attack_results)

    Random_Greedy_large_denial = sum([x - avg for x in Random_Greedy_benign_results_large if x > (avg + 2 * std)]) / sum(
        Random_Greedy_benign_results_large)
    Random_Greedy_large_queries = len(Random_Greedy_attack_results_large)

    Evo_denial = sum([x - avg for x in Evo_benign_results if x > (avg + 2 * std)]) / sum(Evo_benign_results)
    Evo_queries = len(Evo_attack_results)

    Evo_large_denial = sum([x - avg for x in Evo_benign_results_large if x > (avg + 2 * std)]) / sum(
        Evo_benign_results_large)
    Evo_large_queries = len(Evo_attack_results_large)

    print('test')


if __name__ == '__main__':
    main()
