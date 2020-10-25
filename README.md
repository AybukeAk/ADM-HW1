# ADM-HW1

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
    
