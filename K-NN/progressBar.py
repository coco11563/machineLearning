'''
Created on 2016年7月9日

@author: Shaow
'''
import sys 
import time
class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, s):
        sys.stdout.write('进度条：')
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = int(self.width * self.count / self.total)
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('O' * progress + 'o' * int(self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()
def barProcess(times , num ):   
    bar = ProgressBar(total = 10)  
    for i in range(10):  
        bar.move()  
        bar.log('We have arrived at: ' + str(i + 1))  
        time.sleep(1)        
barProcess(1, 3)