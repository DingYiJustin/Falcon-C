import socket
import json
import threading

global single_instance

class ObservationClient:
    _instance_lock = threading.Lock()
    
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = (self.hostname, self.port)
    
    @classmethod
    def instance(cls, *args, **kwargs):
        print("oc instance")
        with ObservationClient._instance_lock:
            print("oc instance1 ")
            # if not hasattr(ObservationClient, "_instance"):
            if single_instance is None:
                print("oc instance2")
                single_instance = ObservationClient(*args, **kwargs)
                print("oc instance3")
        print("oc instance4")
        return single_instance
    
    def sendToServer(self, msg):
        # 发送数据
        try:
            # print(f'{msg}')
            self.client_socket.connect(self.server_address)
            sendall = self.client_socket.send(msg.encode())
            
            # print("senall", sendall)
            # 接收响应
            # print("开始接收响应")
            data = self.client_socket.recv(1024)
            
            self.client_socket.close()
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            return json.loads(data.decode())
        except socket.error as e:
            print(f'socket error {e}')

# if __name__ == "__main__":
#     oc = ObservationClient('10.21.0.110', 65535)
#     oc.sendToServer('hi')