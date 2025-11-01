from colorama import init, Back, Style

init(autoreset=True)

class Logger:
    def __init__(self, name, displayName=False) -> None:
        self.name = name 
        self.display = displayName

    def warn(self, text):
        if self.display:
            print(f"{Back.YELLOW} [{self.name}] [WARNING] {text}") 
        else:
            print(f"{Back.YELLOW} [WARNING] {text}")
        
    def info(self, text):
         if self.display:
            print(f"{Back.BLUE} [{self.name}] [i] {text}") 
        else:
            print(f"{Back.BLUE} [i] {text}")
         
    def error(self, text):
         if self.display:
            print(f"{Back.RED} [{self.name}] [ERROR] {text}") 
        else:
            print(f"{Back.RED} [ERROR] {text}")
         


    @classmethod
    def get_logging_module(self):
        return self.name
