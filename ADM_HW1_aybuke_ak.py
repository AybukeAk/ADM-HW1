# -*- coding: utf-8 -*-

# INTRODUCTION
# Say "Hello, World!" With Python
print("Hello, World!")


# INTRODUCTION
# Python If-Else
n = int(input())
if n % 2 == 0:
    if n in range(2,6):
        print("Not Weird")

    elif n in range(6,21):
        print("Weird")

    elif n > 20:
        print("Not Weird")
else:
    print("Weird")


# INTRODUCTION
# Arithmetic Operators
a= int(input())
b= int(input())
print((a + b), (a - b), (a * b), sep='\n')


# INTRODUCTION
# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b,a/b,sep='\n')
    
    
# INTRODUCTION
# Loops
n = int(input())
for i in range(n):
    print(i**2)
    
    
# INTRODUCTION
# Write a function
def is_leap(year):
   return year % 4 == 0 and (year % 400 == 0 or year % 100 != 0)


# INTRODUCTION
# Print Function
n = int(input())
for i in range(1,n+1):
    print(i,end="")
    
# STRINGS

#!!!!NOTE: When printing in string exercises, I used it to print "if __name__ == '__main__':"  which recommended by the editor . 
#Because Hackerrank did not allow to delete this pattern and write new code.

# STRINGS
#sWAP cASE
def swap_case(s):
    r = ""
    for letter in s:
        if letter == letter.upper():
            r += letter.lower()
        else:
            r += letter.upper()
    return r

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)



# STRINGS
# String Split and Join

def split_and_join(line):
    x=line.split(" ") 
    x='-'.join(x)    
    return x

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print ( result )    
    
# STRINGS
# What's Your Name?
      
def print_full_name(a, b):
    print("Hello %s %s! You just delved into python."%(a,b))
    
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)   
    
# STRINGS
# Mutations

def mutate_string(string, position, character):
    return string[:position] + character + string[position + 1:]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# STRINGS
# Find a string

def count_substring(string, sub_string):
    count = 0
    i = string.find(sub_string)
    while i != -1:
        count += 1
        i = string.find(sub_string, i+1)
    return count

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    count = count_substring(string, sub_string)
    print (count)
    
    
# STRINGS
# String Validators

s = input()
print(any(c.isalnum() for c in s))
print(any(c.isalpha() for c in s))
print(any(c.isdigit() for c in s))
print(any(c.islower() for c in s))
print(any(c.isupper() for c in s))    
     

# STRINGS
# Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)) 
    
    
# STRINGS
# Text Wrap

import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
    
    
# STRINGS
# Designer Door Mat

N, M = map(int, input().split()) 
for i in range(1,N,2): 
    print((i*'.|.').center(M,'-'))
print('WELCOME'.center(M,'-')) 
for i in range(N-2,-1,-2): 
    print((i*'.|.').center(M, '-'))    
    
# STRINGS
# String Formatting

def print_formatted(number):
    b1=len(bin(n)[2:])
    for i in range(1,n+1):
        o=oct(i)[2:]
        h=hex(i)[2:].upper()
        b=bin(i)[2:]
        print (str(i).rjust(b1),str(o).rjust(b1),str(h).rjust(b1),str(b).rjust(b1))
        
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)
    

# STRINGS
# Alphabet Rangoli

import string

def print_rangoli(size):
    alphabet = string.ascii_lowercase
    for i in range(size - 1, 0, -1):
        row = ["-"] * (size * 2 - 1)
        for j in range(0, size - i):
            row[size - 1 - j] = alphabet[j + i]
            row[size - 1 + j] = alphabet[j + i]
        print("-".join(row))

    for i in range(0, size):
        row = ["-"] * (size * 2 - 1)
        for j in range(0, size - i):
            row[size - 1 - j] = alphabet[j + i]
            row[size - 1 + j] = alphabet[j + i]
        print("-".join(row))
if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)   
    
# STRINGS 
# Capitalize!

def solve(s):
    for x in s[:].split():
        s = s.replace(x, x.capitalize())
    return s

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()   
    
    
# STRINGS 
# The Minion Game

def minion_game(s):
    vowels = 'AEIOU'
    kevin_scr = 0
    stuart_scr = 0
    for i in range(len(s)):
        if s[i] in vowels:
            kevin_scr += (len(s)-i)
        else:
            stuart_scr += (len(s)-i)

    if kevin_scr > stuart_scr:
        print ("Kevin", kevin_scr)
    elif kevin_scr < stuart_scr:
        print ("Stuart", stuart_scr )
    else:
        print ("Draw")
if __name__ == '__main__':
    s = input()
    minion_game(s)
    


# STRINGS
# Merge the Tools!

S = input()
K =int(input())
temp = []
len_temp = 0
for item in S:
    len_temp += 1
    if item not in temp:
        temp.append(item)
    if len_temp == K:
        print (''.join(temp) )
        temp = []
        len_temp = 0
    
 
    
    
# DATA TYPES
# List Comprehensions
x = int(input())
y = int(input())
z = int(input())
n = int(input())
print([[i,j,k] 
for i in range(x+1) 
for j in range(y+1) 
for k in range(z+1) 
if sum([i,j,k]) != n])


# DATA TYPES
# Find the Runner-Up Score!
n = int(input())
arr = map(int, input().split())
max = max2 = -101
for n in arr:
    if n > max2:
        if n > max:
            max,max2 = n,max
        elif n < max:
            max2 = n
print(max2)


# DATA TYPES
# Nested Lists
grades = []
for _ in range(int(input())):
    name = input()
    grade = float(input())
    grades.append([name,grade])

second_high = sorted(set([i[1] for i in grades]))[1]
print("\n".join(sorted([i[0] for i in grades if i[1] == second_high])))


# DATA TYPES
# Finding the percentage
n = int(input())
student_marks = {}
for _ in range(n):
    name, *line = input().split()
    scores = list(map(float, line))
    student_marks[name] = scores
query_name = input()
x=student_marks[query_name]
print( format(sum(x)/len(x), ".2f") )


# DATA TYPES
# Lists
N = int(input())
List = []
for n in range(N):
    commands = input().strip().split(" ")
    if commands[0] == "append":
        List.append(int(commands[1]))
    elif commands[0] == "insert":
        List.insert(int(commands[1]), int(commands[2]))
    elif commands[0] == "remove":
        List.remove(int(commands[1]))
    elif commands[0] == "pop":
        List.pop()
    elif commands[0] == "sort":
        List.sort()
    elif commands[0] == "reverse":
        List.reverse()
    elif commands[0] == "print":
        print(List)
        
        
# DATA TYPES
# Tuples
n = int(input())
integer_list = map(int, input().split())
tuple1 = tuple(integer_list)
print(hash(tuple1))

# DATE AND TIME 
# Calendar Module
import calendar
#The editor default brought "TextCalendar", but I couldn't run in my python 3 console, 
#I decided to use day_name and weekday based on my Internet search

m, d, y = map(int, input().split())
print(calendar.day_name[calendar.weekday(y, m, d)].upper())


# DATE AND TIME 
# Time Delta

import os
from datetime import datetime

# The time_delta function 
# I decided to use strptime. References: https://www.programiz.com/python-programming/datetime/strptime . 
# For writing part I used formula provided by Hackerrank

def time_delta(t1, t2):
    t1 = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2 = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    return int(abs((t1-t2).total_seconds())) 

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(str(delta) + '\n')
    fptr.close()


# ERRORS and EXCEPTIONS
# Exceptions

for i in range(int(input())):
    try:
        a,b = map(int,input().split()) 
        division_result = a // b
        print(division_result)
        
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)
        

# PYTHON FUNCTIONALS 
# Map and Lambda Function

cube = lambda x:x**3 

def fibonacci(n):
    fib_list = [0,1]
    for i in range(2,n):
        fib_list.append(fib_list[i-2]+fib_list[i-1]) 
    return fib_list[0:n]

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))


# NUMPY
# Arrays

import numpy
def arrays(arr):
    return numpy.array(arr[::-1], float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# NUMPY
# Shape and Reshape

import numpy
my_array=numpy.array(list(map(int,input().split())))
my_array.shape=(3,3)
print(my_array)


# NUMPY
# Transpose and Flatten

import numpy
n, m = map(int, input().split())
my_array = numpy.array([input().strip().split() for i in range(n)], int)
print (my_array.transpose())
print (my_array.flatten())


# NUMPY
# Concatenate

import numpy
a, b, c = map(int,input().split())
array1 = numpy.array([input().split() for i in range(a)],int)
array2 = numpy.array([input().split() for i in range(b)],int)
print(numpy.concatenate((array1, array2), axis = 0))


# NUMPY
# Zeros and Ones

import numpy
dim = list(map(int,input().split()))
print (numpy.zeros(dim, dtype = numpy.int))
print (numpy.ones(dim, dtype = numpy.int))


# NUMPY
# Eye and Identity

import numpy
n,m= map(int,input().split(" "))
#set_printoptions function solved the problem of much distance than expected from us between numbers 
#references : https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html 

numpy.set_printoptions(sign=' ') 
print(numpy.eye(n,m))


# NUMPY
# Array Mathematics

import numpy as np 
n,m =map(int,input().split())
arrA=np.array([input().split() for i in range(n)],int) 
arrB=np.array([input().split() for i in range(n)],int) 
print(arrA+arrB, arrA-arrB,arrA*arrB,arrA//arrB,arrA%arrB,arrA**arrB, sep ='\n')

  
# NUMPY
# Floor, Ceil and Rint

import numpy as np
nums =np.array(input().split(),float)
np.set_printoptions(sign=' ')
print(np.floor(nums))
print(np.ceil(nums))
print(np.rint(nums))


# NUMPY
# Sum and Prod

import numpy as np
n,m = map(int,input().split())
arr = np.array([input().split() for i in range(n)], int)
print(np.prod((np.sum(arr,axis=0))))


# NUMPY
# Min and Max

import numpy as np
n, m = map(int, input().split())
A = np.array([input().split() for i in range(n)],int)
print(np.max(np.min(A, axis=1), axis=0))


# NUMPY
# Mean, Var, and Std

import numpy as np
n,m = map(int, input().split())
arr = np.array([list(map(int, input().split())) for i in range(n)])
np.set_printoptions(legacy='1.13')
print(np.mean(arr, axis=1))
print(np.var(arr, axis=0))
print(np.std(arr, axis=None))


# NUMPY
# Dot and Cross

import numpy as np
n = int(input())
a = np.array([input().split() for i in range(n)],int)
b = np.array([input().split() for i in range(n)], int)
print(np.dot(a,b))


# NUMPY
# Inner and Outer

import numpy as np
A = np.array(input().split(), int)
B = np.array(input().split(), int)
print(np.inner(A,B), np.outer(A,B), sep='\n')


# NUMPY
# Polynomials

import numpy as np
cof = list(map(float,input().split()));
x = input();
print(np.polyval(cof,int(x)));


# NUMPY
# Linear Algebra

import numpy as np
n = int(input())
arr = np.array([input().split() for i in range(n)], float)
det_arr = np.linalg.det(arr)
print(round(det_arr, 2))


# SETS
# Introduction to Sets

def average(array):
    s=set(array) 
    res=sum(s)/ len(s) #result 
    return res

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    result = average(arr)
    print ( result )
    

# SETS
# Symmetric Difference 

M,m=input(),set(list(map(int,input().split())))
N,n=input(),set(list(map(int,input().split())))
dif = sorted(list(m.difference(n))+list(n.difference(m)))
for i in dif:
    print (i)    
 
# SETS
# Set .add()

n = int(input())
s = set()
for i in range(n):
    string = input()
    s.add(string)
print(len(s))
  

# SETS
# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
for _ in range(int(input())):
    p = input().split()
    if p[0]=='pop':
        s.pop()
    elif p[0]=='discard':
        s.discard(int(p[1]))
    else:
        s.remove(int(p[1]))
print(sum(s))



#*********************************************************************************************#
  
#QUESTION 2
# Birthday Cake Candles

n = int(input().strip())
height = [int(height_temp) for height_temp in input().strip().split(' ')]

print(height.count(max(height)))


# I couldn't be successful on Kangaroo problem


# Viral Advertising

m = [2]
for i in range(int(input())-1):
    m.append(int(3*m[i]/2))
print(sum(m))



# Insertion Sort - Part 1

def insertionSort1(n, arr1):
    x = arr1[-1]
    arr = arr1[::-1]
    for i in range(0,n-1):
        if x < arr[i+1]:
            arr[i] = arr[i+1]
            print(" ".join(str(i) for i in arr[::-1]))
        else:
            arr[i] = x
            print(" ".join(str(i) for i in arr[::-1]))
            break
        if(i+1 == n-1 and min(arr1)==x):
                temp = arr[i+1]
                arr[i+1] = x
                arr[i] = temp
                print(" ".join(str(i) for i in arr[::-1]))
                
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)
    
    
# Insertion Sort - Part 2


import os

def insertionSort2(n, arr):
    for i in range(n):
        if(i == 0):
            continue
        for j in range(0, i):
            if(arr[j] > arr[i]):
                arr0 = arr[i]
                arr[i] = arr[j]
                arr[j] = arr0
            else:
                continue
        print(*arr)

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

