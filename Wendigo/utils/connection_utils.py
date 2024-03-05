import json
import ssl
import socket
import requests


#  this class establishes the connection to the GraphQL endpoint
class ConnectionHandler:
    # connection params
    socket = None  # the socket
    connection = None  # the connection
    ip = "localhost"  # the ip address
    port = 80  # the port
    path = "/graphql"  # the GraphQL endpoint

    # SSL params
    no_ssl = True  # flag to use SSL or not
    verify_ssl = False  # flag to verify SSL certificates or not
    ssl_context = None  # the SSL context
    cert_file = None

    # HTTP params
    user_agent = "Wendigo"  # the user agent
    headers = []
    timeout = 1  # seconds
    delay = 0  # if a delay is necessary

    def __init__(self, settings):
        """
        Settings is a dict that defines the connection params
        """
        self.ip = settings["ip"] if "ip" in settings else self.ip
        self.port = settings["port"] if "port" in settings else self.port
        self.path = settings["path"] if "path" in settings else self.path
        self.no_ssl = settings["no_ssl"] if "no_ssl" in settings else self.no_ssl
        self.ssl_context = None
        self.cert_file = settings["cert_file"] if "cert_file" in settings else self.cert_file
        self.socket = None
        self.connection = None
        self.user_agent = settings["user_agent"] if "user_agent" in settings else self.user_agent
        self.headers = settings["headers"] if "headers" in settings else self.headers
        self.headers.append("User-Agent: " + self.user_agent)
        self.timeout = settings["timeout"] if "timeout" in settings else self.timeout
        self.delay = settings["delay"] if "delay" in settings else self.delay
        self.connect()
        self.login()

    def connect(self):
        """
        Connects to the GraphQL endpoint
        """
        if self.no_ssl:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
        else:
            # TODO: check if it works
            if self.verify_ssl:
                self.ssl_context = ssl._create_unverified_context()
            else:
                self.ssl_context = ssl.create_default_context()
                if self.cert_file:
                    # TODO add also the key file
                    self.ssl_context.load_cert_chain(certfile=self.cert_file)
            with self.socket.create_connection(self.ip, self.port) as sock:
                self.socket = self.ssl_context.wrap_socket(sock, server_hostname=self.ip)
        # TODO: check if it is necessary
        self.connection = self.socket.makefile('rw')

    def login(self):
        """
        Login to the GraphQL endpoint
        """
        pass

    def buildHTTPRequest(self, query):
        """
        Builds the HTTP request
        """
        body = {}
        body["query"] = query
        body["variables"] = None
        body["operationName"] = None
        body = json.dumps(body)

        request = "POST " + self.path + " HTTP/1.1\r\n"
        request += "Host: " + self.ip + ":" + str(5013) + "\r\n"
        request += "Content-Type: application/json\r\n"
        request += "Content-Length: " + str(len(body)) + "\r\n"
        request += "Connection: close\r\n"
        request += "\r\n"
        request += str(body)
        request += "\r\n\r\n"
        return request

    def send(self, payload):
        # TODO: convert to socket.send(message.encode('utf-8'))
        self.socket.sendall(self.buildHTTPRequest(payload).encode('utf-8'))

    def receive(self):
        self.socket.settimeout(self.timeout)
        response = self.socket.recv(4096)
        print(response)


def test():
    settings = {}
    settings["port"] = 8080  # dvga
    connHandler = ConnectionHandler(settings)
    connHandler.send("{systemHealth}")
    connHandler.receive()


if __name__ == "__main__":
    test()
