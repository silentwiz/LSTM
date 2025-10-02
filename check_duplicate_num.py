import numpy as np

a = [
    23,
    15,
    6,
    33,
    25,
    37,
    11,
    38,
    20,
    5
  ]
b = [2,15,25,27,38,42]
c = []

a.sort()
b.sort()
print(f"a : {a}\nb : {b}")
for num in a:
    if num in b:
        c.append(num)
print(f'the duplicated number is : {c}')



'''
d = "[42 6 38 36 23 26 19 16 15 22]"
e = []
print(str(d))
a = d.split()
for i in a:
    i = i.replace('[', '')
    i = i.replace(']', '')
    if i == '' or i == ' ':
        continue
    e.append(i)
print(e)
print(d == e)
'''