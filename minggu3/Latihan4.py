# Nama     : Firman Naufal Aryaputra
# NIM      : A11.2022.14181
# Kelompok : A11.4419

# Module Packages
# Diibaratkannya Pengguna (User) = menggunakan module yang sudah ada 
# Diibaratkannya Penyedia (supplier) = menggunakan module baru 
# Module dapat didefinisikan berdasarkan nama 

# Importing modules
# Mengimport module berdasarkan nama dari module tersebut
# Didalamnya terdapat "Import" keyword untuk memasukan module dan "name" nama dari module yang dimasukkan 

# Modul diimport berdasarkan nama dari modul tersebut. 
# Didalamnya terdapat keyword import untuk memasukan modul dan name nama dari modul yang dimasukkan.
# Contohnya : 
import math
import math, sys

print ()

# Namespace 
# Dipahami sebagai non-pyshical context yang dimana nama tersebut ada 
# dan tidak menimbulkan konflik satu sama lain 

# Contohnya 
import numpy
import math
import scipy

print(math.pi)
print(math.e)
print(numpy.pi)
print(scipy.pi)
# Pi tidak akan mempengaruhi jalannya prgram utama 
print()

from math import pi,e

print(pi)
print(e)

print()

# dir(math)
# Import Modul:
# import math: Mengimport modul math ke dalam program.
# import math, sys: Mengimport modul math dan sys ke dalam program.

# Namespace:
# print(math.pi): Mencetak nilai pi dari modul math.
# print(math.e): Mencetak nilai e dari modul math.
# print(numpy.pi): Mencetak nilai pi dari modul numpy.
# print(scipy.pi): Mencetak nilai pi dari modul scipy.
# from math import pi, e: Mengimport nama pi dan e dari modul math.
# print(pi): Mencetak nilai pi yang diimport dari modul math.
# print(e): Mencetak nilai e yang diimport dari modul math.

# Menampilkan Daftar Nama:
# dir(math): Menampilkan daftar nama dalam modul math.

## override nilai sin dan pi
from math import sin, pi

print(sin(pi/2))
# Kode diatas befungsi untuk memanggil function sin kedalam module math untuk 
# Menginput angka didalam tanda kurung
print("=====")

pi = 3.14

def sin(x):
    if 2 * x == pi: # Diberi if apabila kondisinya benar 
        return 0.99999999
    else: # else apabila kondisinya salah 
        return None

print(sin(pi/2))

print()

# Mengimport Modeul Math 
from math import *
print(tan(0))

print()

# Kode dibawah ditambahkan "as" dikarenakan untuk menyebarkan module 
# yang didefinisikan dengan namayang berbeda dari aslinya 
# PI dimaksud kan dengan angka (3,14) 
import math as m

print(m.pi)

print()
# Kode dibawah merupakan contoh dari implementasi Aliasing ("as")
from math import pi as PI, sin as sine

print(sine(PI/2))

print()

# Standart Module 
import math
print(dir(math)) # dir merupakan function return untuk mengembalikan 
# nama dan atribut method yang digunakan 

# Contoh implementasinya
import math
# Pow merupakan fucntion yang digunakan untuk memberi exponen
a = math.pow(2,3) #ini merupakan masukan angka dari 2 pangkat 3 
print(a)

for name in dir(math): # for merupakan looping, yang dimana dia melooping list attribut math 
    print(name, end="\t")

print() 

# Module math 
# - `sin(x)` → the sin dari x;
# - `cos(x)` → the cos dari x;
# - `tan(x)` → the tan dari x.

# Versi invertednya 
# - `asin(x)` → the arcsine of x;
# - `acos(x)` → the arccosine of x;
# - `atan(x)` → the arctangent of x.
# `x` is merupakan radian

# Contoh implementasi dari math module 
from math import pi, radians, degrees, sin, cos, tan, asin

ad = 90

ar = radians(ad)
print(ar)
ad = degrees(ar)
print(ad)

print(ad == 90.)
print(ar == pi / 2.)
print(sin(ar) / cos(ar) == tan(ar))
print(asin(sin(ar)) == ar)

print()

# - `e` → a constant with a value that is an approximation of Euler's number (e)
# - `exp(x)` → finding the value of ex;
# - `log(x)` → the natural logarithm of x
# - `log(x, b)` → the logarithm of x to base b
# - `log10(x)` → the decimal logarithm of x (more precise than log(x, 10))
# - `log2(x)` → the binary logarithm of x (more precise than log(x, 2))

# Implementasinya 
from math import e, exp, log

print(pow(e, 1) == exp(log(e)))
print(pow(2, 2) == exp(2 * log(2)))
print(log(e, e) == exp(0))

print()

# Build in function 
# the pow() function: `pow(x, y)` untuk menemukan value dari X dan Y 

# The last group consists of some general-purpose functions like:
# - ceil(x) → the ceiling of x (the smallest integer greater than or equal to x)
# - floor(x) → the floor of x (the largest integer less than or equal to x)
# - trunc(x) → the value of x truncated to an integer (__be careful__ - it's __not an equivalent__ either of ceil or floor)
# - factorial(x) → returns x! (x has to be an integral and not a negative)
# - hypot(x, y) → returns the length of the hypotenuse of a right-angle triangle with the leg lengths equal to x and y (the same as sqrt(pow(x, 2) + pow(y, 2)) but more precise)

# Demonstrasi dari Ceil, Floor, Trunc (Meurpakan function dari math module)
from math import ceil, floor, trunc

x = 1.4
y = 2.6

print(floor(x), floor(y)) # floor function rounds 1.4 down to 1 and -1.4 down to -2
print(floor(-x), floor(-y))
print(ceil(x), ceil(y))  # ceil function rounds 1.4 up to 2 and -1.4 up to -1
print(ceil(-x), ceil(-y))
print(trunc(x), trunc(y))
print(trunc(-x), trunc(-y)) # trunc function menghilangkan decimal dari angka, jadi 1.4 menjadi 1 dan -1.4 menjadi -1.

print()

# Random Module 
# Mengantarkan sebuah mekanik untuk menggunakan angkapseudorandom 
# pseudo :  Merupakan angka yang di generate dengan modul yang mungkin akan terlihat acak dalam arti bahwa tidak dapat memprediksi nilai selanjutnya, tetapi jangan lupa bahwa semuanya dihitung menggunakan algoritma yang sangat halus.

from random import random
for i in range(5):
    print(random())

print() 

from random import randrange, randint

print(randrange(200), end='\n') # Randrange itu berfungsi mengimport angka random yang speicific
print(randrange(50, 100), end='\n')
print(randrange(50, 200, 10), end='\n')
print(randint(5, 10)) #Menggenerate angka intereger secara random 

print()

from random import choice, sample

lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
siswa = ['Ani','Budi','Cakra','Desi']
print(choice(siswa))  # choice function returns a random element from a given sequenc
print(sample(lst, 1)) #  sample function returns a specified number of elements from a given sequence, chosen at random without replacement.
print(sample(lst, 10))

print()

from random import randint

for i in range(10):
    print(randint(1, 10), end=',')

print()

# ini merupakan module platform memberikan informasi os yang digunakan
from platform import platform

print(platform())
print(platform(1))
print(platform(0, 1))

print()

from platform import machine # Memberikan informai dari mesin sistem yang digunakan

print(machine())

print()

from platform import processor # Memberikan informasi tentang processor yang digunakan  

print(processor())

print()