# commenti su riga


x=3.4
print(2*x)

if(x>0):
  print("Valore positivo")
else:
  print("Valore negativo")
print("questo è fuori dall'if")

flag= True
flag2=not False

r=[2,4,7,5,0] #list
for i in range(5):
  print("element ", i, "=",r[i])

#concatenation
q=[28,46,31]
z=r+q
z.append(100)
print("concatenation: ",z)

#tubble
a=(1,2,3,4)

#subsection of the vector
print("first tree elements: ",z[0:3])
n=len(z) #oppure z.shape
print("last tree elements: ",z[-3:])
print("last tree elements: ",z[n-3:n])

#function
def f(a,b,c):
  sum=a+b+c
  prod=a*b*c
  return (sum, prod) #si possono avere diverse restituzioni


print("function result: ",f(0,2,4))
#print("function result: ",f(0,2)) # last default value=0 (A ME DA ERRORE)
print("sum, product: ", f(b=0,a=2,c=4)) #assegnazione specifica delle variabili

#class
class point2D:
  def __init__(self, x, y): #contructor+self=this(pointer)
    self.x=x
    self.y=y
  def print_p(self):
    print("x",self.x,"y",self.y)
  def increment_x(self):
    self.x+=1
  def norm(self):
    # alternativa: from math import sqrt
    #               norm=sqrt(self.x^2+self.y^2)
    import math
    norm=math.sqrt(self.x^2+self.y^2)
    return norm

p=point2D(2,6)
print("punto iniziale: ")
p.print_p()
p.increment_x()
print("punto incrementato: ")
p.print_p()
print("norm: ",p.norm())

import numpy as np
x=np.array[3,6.2,8.3]
y=np.array[1.9,2.5,7]
z=x+y
print("somma di array (no concatenazione): ",z)
