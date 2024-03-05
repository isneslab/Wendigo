import json
import multiprocessing
import time
import pickle
import docker
import socket
from multiprocessing import Process

from Wendigo.environments.GraphQLEnv import connect


def send_query(query, connection, connection_settings):
    body = query
    error = None
    response = b''
    response_time = 0

    body = json.dumps(body)
    host = connection_settings["targetHost"]
    path = connection_settings["targetPath"]
    method = connection_settings["method"]
    headers = connection_settings["headers"]

    request = method + " " + path + " HTTP/1.1\r\n"
    request += "Host: " + host + "\r\n"
    for header, value in headers.items():
        request += header + ":" + value + "\r\n"
    request += f"Content-Length: {len(body)}" + "\r\n"
    request += "\r\n"
    request += body
    request += "\r\n"

    time_r = time.time()
    try:
        connection.sendall(request.encode('utf-8'))

        try:
            chunk = connection.recv(4096)
            while chunk:
                chunk = connection.recv(4096)
            response_time = (time.time() - time_r)

        except socket.error as e:
            response = None if response == b'' else response
            response_time = (time.time() - time_r)
            error = e.errno
            print('Error: ' + str(error))

    except BrokenPipeError:
        response = None if response == b'' else response
        response_time = (time.time() - time_r)
        error = 'BrokenPipe'
        print('Error: ' + error)

    return response, response_time, error


def attack(query, connection_settings):
    start_time = time.time()
    count = 1
    while (time.time() - start_time) < 3600:
        connection, connected = connect(connection_settings)
        if connected:
            response, response_time, error = send_query(query=query, connection=connection,
                                                        connection_settings=connection_settings)
            with (open('results-temp/attack' + '-' + str(count) + '.p', "wb") as file):
                pickle.dump(response_time, file)
            count += 1


def benign(query, connection_settings):
    start_time = time.time()
    count = 1
    while (time.time() - start_time) < 3600:
        connection, connected = connect(connection_settings)
        if connected:
            response, response_time, error = send_query(query=query, connection=connection,
                                                        connection_settings=connection_settings)
            with (open('results-temp/benign' + '-' + str(count) + '.p', "wb") as file):
                pickle.dump(response_time, file)
            count += 1


def main():
    with open('../settings/general_settings.json', 'r') as settings_file:
        settings = json.load(settings_file)
    connection_settings = settings['connection-settings']

    # Init Docker container
    app_name = connection_settings['appName']
    dockerClient = docker.from_env()
    container = dockerClient.containers.run(app_name, detach=True,
                                            ports={'5013/tcp': str(connection_settings['targetPort'])},
                                            environment=["WEB_HOST=0.0.0.0"])
    time.sleep(3)

    delay, attack_query = pickle.load(open('queries/EvoMaster-query-large.p', 'rb'))

    benign_query = {'query': 'query{pastes{content}}'}

    att = Process(target=attack, args=(attack_query, connection_settings))
    ben = Process(target=benign, args=(benign_query, connection_settings))

    att.start()
    ben.start()

    ben.join()
    att.join()

    container.stop()
    container.remove()


if __name__ == '__main__':
    main()
