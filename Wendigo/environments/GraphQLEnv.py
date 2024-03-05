import json
import socket
import ssl
import time
import enum
import gymnasium as gym
import docker


class ErrorType(enum.Enum):
    ConnectionResetError = 1
    ConnectionError = 2
    HostError = 3
    TimeoutError = 4
    BrokenPipe = 5
    UnknownError = 99


def connect(connection_settings):
    host = connection_settings["targetHost"]
    port = int(connection_settings["targetPort"])
    connected = False

    if not connection_settings["useSSL"]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # TODO: add retry on failure

        try:
            sock.connect((host, port))

        except ConnectionRefusedError:
            # print("Connection refused, is the server running?")
            # exit(1)
            sock = None

    else:
        # TODO: test if SSL works
        if not connection_settings["verifySSL"]:
            ssl_context = ssl._create_unverified_context()

        else:
            ssl_context = ssl.create_default_context()
            if connection_settings["certPath"] is not None:
                ssl_context.load_cert_chain(certfile=connection_settings["certPath"],
                                            keyfile=connection_settings["keyFile"])

        with socket.create_connection((host, port)) as sock:
            sock = ssl_context.wrap_socket(sock, server_hostname=host)

    if sock is not None:
        connected = True

    return sock, connected


class GraphQLEnv(gym.Env):
    metadata = {"render_modes": ["None"], "render_fps": 0}

    index = 0

    def __init__(self, connection_settings=None):

        if connection_settings is None:
            connection_settings = {
                "useSSL": False,
                "verifySSL": False,
                "certPath": None,
                "targetHost": "localhost",
                "targetPort": 8080,
                "targetPath": "/graphql",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 5
            }

        connection_settings['targetPort'] = connection_settings['targetPort'] + GraphQLEnv.index
        GraphQLEnv.index += 1

        # Random number generator seed
        self._seed = None

        # Init Docker container
        self.app_name = connection_settings['appName']
        self._dockerClient = docker.from_env()
        self._container = None

        # Connection Data
        self._connection = None
        self._connection_settings = connection_settings

        # Seen data (type: [data])
        self._seen_data = {}

        # None as we are not rendering the environment
        self.render_mode = None
        self.window = None
        self.clock = None

    def seed(self, seed):
        self._seed = seed

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def _login(self):
        # TODO: Add authentication if needed
        return None

    def send_query(self, query):
        # embed query in query { xxx }
        body = query
        error = None
        response = b''
        response_time = 0

        body = json.dumps(body)
        host = self._connection_settings["targetHost"]
        port = self._connection_settings["targetPort"]
        path = self._connection_settings["targetPath"]
        method = self._connection_settings["method"]
        headers = self._connection_settings["headers"]
        timeout = self._connection_settings["timeout"]

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
            self._connection.sendall(request.encode('utf-8'))

            try:
                # get the full response from chunks of 4096 bytes
                chunk = self._connection.recv(4096)
                response += chunk
                response_time = (time.time() - time_r)
                collect = 0
                collect_max = 10

                while True:
                    chunk = self._connection.recv(4096)

                    if not chunk:
                        break

                    if collect < collect_max:
                        response += chunk
                        collect += 1

                # print('Time to receive complete response: ' + str(time.time() - time_r))

            except socket.timeout:
                response = None
                response_time = timeout
                error = ErrorType.TimeoutError

            except socket.error as e:
                response = None if response == b'' else response
                response_time = (time.time() - time_r)

                if (e.errno == 10054 or e.errno == 54) and response is not None:
                    # 10054: Connection reset by peer after response was sent
                    error = ErrorType.ConnectionResetError

                elif e.errno == 10054 or e.errno == 10061 or e.errno == 54 or e.errno == 61:
                    # 10054: Connection reset by peer
                    # 10061: Connection refused
                    error = ErrorType.HostError

                elif e.errno == 10051 or e.errno == 10050 or e.errno == 51 or e.errno == 50:
                    # 10051: Network is unreachable
                    # 10050: Network is down
                    error = ErrorType.ConnectionError

                else:
                    error = ErrorType.UnknownError

        except BrokenPipeError:
            response = None if response == b'' else response
            response_time = (time.time() - time_r)
            error = ErrorType.BrokenPipe

        return response, response_time, error

    def reset(self, seed=None, options=None):
        # seeds self.np_random
        if seed is None:
            seed = self._seed

        super().reset(seed=seed, options=options)

        # Run docker run -t -p 5013:5013 -e WEB_HOST=0.0.0.0 dolevf/dvga
        self._container = self._dockerClient.containers.run(self.app_name, detach=True,
                                                            ports={'5013/tcp':
                                                                   str(self._connection_settings['targetPort'])},
                                                            environment=["WEB_HOST=0.0.0.0"])
        # wait for docker container to start up
        time.sleep(3)

        # Connect to the docker container
        self._connection, connected = connect(self._connection_settings)

        if not connected:
            print("Connection refused, is the server running?")
            exit(1)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _perform_action(self, action):
        pass

    def queryify(self, query, indent=1):
        pass

    def step(self, action):
        pass

    def render(self):
        return None

    def close(self):
        if self._connection:
            self._connection.close()

        # close the docker container
        self._container.stop()

        # delete it
        self._container.remove()

        return None
