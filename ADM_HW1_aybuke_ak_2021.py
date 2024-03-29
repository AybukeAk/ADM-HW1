### Regex and Parsing challenges
## Detect Floating Point Number

import re
for _ in range(int(input())):
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())))


### XML
## XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    return sum([len(element.items()) for element in tree.iter()])

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))
    
## XML2 - Find the Maximum Depth
    
import xml.etree.ElementTree as etree

maxdepth = 0

def depth(elem, level):
    global maxdepth
    # your code goes here
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for c in elem:
        depth(c, level)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)
    
    
    
 ### Closures and Decorators
 
 ## Standardize Mobile Number Using Decorators
 
def wrapper(f):
    def phone(l):
        f("+91 "+c[-10:-5]+" "+c[-5:] for c in l)
    return phone 

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 
