# List[], Tuple(), Sets{}

# print(round(10.523,2))
lis = [1,2,3,4,5,6]
# print(dir(list))
lis.append(9)
# lis.insert(0,"uzama")
lis.remove(2)
del lis[1]
print(lis.index(6))

a = lis.copy()
lis.remove(1)

lis.extend(a)
lis.pop()

lis.sort(reverse=False)
a = sorted(a, reverse=True)
lis.reverse()
print(a)
print(lis)
print(min(a))
print(max(a))
print(sum(a))
for i, j in enumerate(lis,start = 5):
	print(str(i) + "----" +str(j))
# print(a,b)
print(1 in a)
tuples = (1,2,3,1,2)
print(tuples)
sets = {1,2,3,4,1}
sets.add(7)
print(dir(set)) 
# for i in sets:
	# print(i)
print(sets) #intersection, difference, union ..... methods 

c= set() #this emty set
d = tuple() #empty tuple

# String

import sys

print("________".join("Hellos World".split(' ')))

li = ['1','2','3','4','5','6','7','8','9']

print("--".join(li).split('--'))

message = "Uzama Zaid"

print(message.upper()[::-1])
6
print(message.lower().count('a'))
print(len(message)-1-message[::-1].find('d'))


message = message.lower().replace("zaid", "Jaward").capitalize().split(" ")

message = " ".capitalize().join(message)

print(message)

q = "How Are You?"
print(f"{message.split(' ')}")

print('Hello {}. Welcome! \n {} {}'.format(message, q, 'cgcf'))

print(dir(message))

print(help(int))





# print "hello"

# Dictionary

student = {
	
	'name' : "uzama",
	'age'  : 34,
	"university" : "Moratuwa"
	}

student.update({'name':"zaid", 'department':"CSE"})

# del student['age']
age = student.pop('age')
print(student)
# print(dir(dict))
print(student.keys())
print(student.values())
print(student.items())
print(len(student))

for i, j in student.items():
	print(i, j)

#If elif else

x = 0

if x:
	print("Hello")

a = b = 3
c = [12,12,12]
d = [12,12,12]
print(a is b)
print(c is d)

print(id(c), id(d))

while x:
	print(x)
# 0112650912
#LOOPS
#Functions

def main():
	print("Hello")
	
def one(main):
	print(main)

print(main)
one(main)