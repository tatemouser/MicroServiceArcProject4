import ast
import socket
import json
import argparse
import os

def parse_code(file_path):
    with open(os.path.join("./files/", file_path), "r") as file:
        code = file.read()
    
    tree = ast.parse(code)
    parsed_data = {"functions": [], "assignments": [], "loops": [], "conditionals": []}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            parsed_data["functions"].append({"name": node.name, "lineno": node.lineno})
        elif isinstance(node, ast.Assign):
            parsed_data["assignments"].append({"lineno": node.lineno})
        elif isinstance(node, ast.For):
            parsed_data["loops"].append({"lineno": node.lineno})
        elif isinstance(node, ast.If):
            parsed_data["conditionals"].append({"lineno": node.lineno})

    return parsed_data

def server_1(fname):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 5000))
        s.listen()
        print("Parsing server listening on port 5000...")

        print(os.listdir())
        
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            parsed_data = parse_code(fname)
            conn.sendall(json.dumps(parsed_data).encode())





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing server")
    parser.add_argument("file", help="The filename of the Python script to parse")
    args = parser.parse_args()

    server_1(args.file)