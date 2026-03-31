import os
from typing import List

login_time = ''
log_filepath = ''

class Entry:
    def __init__(self, entry_name):
        self.entry = entry_name
        self.error = None
        self.warning = []
    
    def have_error(self):
        return self.error != None
    
    def have_warning(self):
        return len(self.warning) != 0

logs: List[Entry] = []

def new_log(path, log_name):
    global login_time, log_filepath
    log_filepath = os.path.join(path, f"{log_name}.txt")
    os.makedirs(path, exist_ok=True)
    with open(log_filepath, 'a') as file:
        file.write(f"Log: {log_name}\n")

def end_log():
    global log_filepath
    with open(log_filepath, 'a') as file:
        file.write(f"End of file\n")

def new_entry(entry_name):
    global log_filepath
    print(f"\033[32mNow processing {entry_name}...\033[0m")
    logs.append(Entry(entry_name))

def add_error(error):
    global log_filepath
    print(f"\033[31mError found when processing {logs[-1].entry}: {error}\033[0m")
    logs[-1].error = error
    with open(log_filepath, 'a') as file:
        file.write(f"Entry: {logs[-1].entry}, Error: {error}\n")

def add_warning(warning):
    global log_filepath
    print(f"\033[33mWarning found when processing {logs[-1].entry}: {warning}\033[0m")
    logs[-1].warning.append(warning)
    with open(log_filepath, 'a') as file:
        file.write(f"Entry: {logs[-1].entry}, Warning: {warning}\n")