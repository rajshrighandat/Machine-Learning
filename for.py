

b=[1,2,3,4,5]
mul=2
for i in (b):
    mul *= i
print(b)

b.sort()
print(b)
print(b[4])

b[0],b[3] = b[3],b[0]
print(b)
