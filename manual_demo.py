from tello_morelo import Tello
from utils import connect_to_wifi
from termcolor import cprint 

def main():
    server_name = "TELLO-MORELO"
    password = "twojastara"
    
    if not connect_to_wifi(server_name, password):
        cprint(f"Failed to connect to wifi {server_name}", "red")
        exit(1)
    else:
        cprint(f"Connected to {server_name}", "green")
    
    tello = Tello()
    while True:
        tello.get_data()


if __name__ == "__main__":
    main()
