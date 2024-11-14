import socket
import json
import time

def generate_comments(parsed_data):
    comments = []
    for function in parsed_data["functions"]:
        comments.append({"lineno": function["lineno"], "comment": f"# Function '{function['name']}' definition"})
    for assign in parsed_data["assignments"]:
        comments.append({"lineno": assign["lineno"], "comment": "# Variable assignment"})
    for loop in parsed_data["loops"]:
        comments.append({"lineno": loop["lineno"], "comment": "# Loop starts here"})
    for cond in parsed_data["conditionals"]:
        comments.append({"lineno": cond["lineno"], "comment": "# Conditional statement"})
    return comments

def client_1_and_server_2():
    # Client: connect to the Parsing container
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        retries = 5
        while retries > 0:
            try:
                s.connect(('parsing_container', 5000))
                break  # Success, exit the loop
            except socket.gaierror as e:
                print(f"Connection failed, retrying... ({retries} retries left)")
                retries -= 1
                time.sleep(2)  # Wait before retrying
        if retries == 0:
            print("Failed to connect after multiple attempts")
        data = s.recv(4096).decode()
        parsed_data = json.loads(data)

    # Generate comments based on parsed data
    comments = generate_comments(parsed_data)

    # Server: send comments to the Adding Comments container
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 5001))
        s.listen()
        print("Commenting server listening on port 5001...")
        
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            conn.sendall(json.dumps(comments).encode())

if __name__ == "__main__":
    client_1_and_server_2()