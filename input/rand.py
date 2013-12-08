import random

f = open('random.csv', 'w')

for i in range(10000):
   f.write(str(random.random()))
   f.write("\t")
   f.write(str(random.random()))
   f.write("\t")
   f.write(str(random.random()))
   f.write("\n")


