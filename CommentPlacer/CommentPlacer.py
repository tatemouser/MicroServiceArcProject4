import socket
import json
import argparse
import time
import os
import csv
import datetime

def add_comments(file_path, comments):
    with open(os.path.join("./files/", file_path), "r") as file:
        code = file.readlines()
    num_comments = 0
    for comment in sorted(comments, key=lambda x: x["lineno"]):
        code.insert(comment["lineno"] - 1 + num_comments, comment["comment"] + "\n")
        num_comments += 1


    new_file_path = os.path.join("./files/", file_path).replace(".py", "_commented.py")
    with open(new_file_path, "w") as new_file:
        new_file.writelines(code)

    print(f"Commented code saved to {new_file_path}")

def client_2(fname):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        retries = 5
        while retries > 0:
            try:
                s.connect(('commenting_container', 5001))
                break  # Success, exit the loop
            except socket.gaierror as e:
                print(f"Connection failed, retrying... ({retries} retries left)")
                retries -= 1
                time.sleep(2)  # Wait before retrying
        if retries == 0:
            print("Failed to connect after multiple attempts")
        data = s.recv(1048576).decode()
        comments = json.loads(data)
        add_comments(fname, comments)
    
    with open("./logs/log.csv", 'a') as f:
        f.write(str(datetime.datetime.now()))
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adding comments client")
    parser.add_argument("file", help="The filename of the Python script to add comments to")
    args = parser.parse_args()

    client_2(args.file)
