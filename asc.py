# 创建者：Cmyu
# 创建时间： 2023-06-23 16:58
from pyfiglet import Figlet
from colorama import Fore

f = Figlet(font='slant')
text = f.renderText('Python')

print(Fore.GREEN + text)