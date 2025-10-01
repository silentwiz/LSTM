import numpy as np

a = [23, 15, 6, 25, 11, 33, 37, 2, 20, 38]
b = [6, 10, 36, 18, 19, 22, 38, 42, 3, 23]
c = []

a.sort()
b.sort()
print(f"a : {a}\nb : {b}")
for num in a:
    if num in b:
        c.append(num)
print(f'the number is : {c}')

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