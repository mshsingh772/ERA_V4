import os

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.fp = open(log_file, 'w', encoding='utf-8')

    def log(self, message):
        self.fp.write(message + '\n')
        self.fp.flush()  # write immediately

    def close(self):
        self.fp.close()
