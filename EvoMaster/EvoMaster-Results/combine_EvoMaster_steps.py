import pickle
import ast


def main():
    num_steps = 6855
    steps = []

    for i in range(num_steps):
        with open('exchange-' + str(i) + '.txt', 'r') as file:
            data = file.read()

            response = data.split('"response": ')[1]
            response = response.split(',')[0]

            request = data.split('"request": ')[1]
            request = request.split(',')[0]

            time = data.split('"time": ')[1]
            time = time.split(',')[0]

            status = data.split('"status code": ')[1]
            status = status.split('}')[0]

            if not ('mutation' in request) and len(steps) < 1280:
                is_rejected = int(status) == 400
                steps += [(i+1, request, float(time), is_rejected)]

    with open('evomaster-combined.p', 'wb') as file:
        pickle.dump(steps, file)


if __name__ == '__main__':
    main()
