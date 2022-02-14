from subprocess import DEVNULL, check_call
from termcolor import cprint 

def connect_to_wifi(name, password, num_retries=20):
    cmd_refresh_list = "sudo iwlist wlp4s0 scan"
    cmd_connect = "nmcli d wifi connect {} password {}".format(name, password)
  
    for i in range(num_retries):
        try:       
            check_call(cmd_refresh_list.split(), stdout=DEVNULL, stderr=DEVNULL)
            check_call(cmd_connect.split(), stdout=DEVNULL, stderr=DEVNULL)
        except:
            cprint(f"\rCould not connect to {name}. Retrying {i + 1}/{num_retries} ...", "yellow", end="") 
            continue
        if i:
            print()
        return True
    
    print()
    return False
   
