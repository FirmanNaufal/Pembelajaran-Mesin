# Nama     : Firman Naufal Aryaputra
# NIM      : A11.2022.14181
# Kelompok : A11.4419

# Function 
# Motivasi function adalah perulangan kode

print("Angka 1 :")
a = int(input())
print(a)

print("Angka 2 :")
b = int(input())
print(b)

print("Angka 3 :")
c = int(input())
print(c)

print()

# Sintaks Dalam Function : 
# def functionName():
    # functionBody

def inputAngka():
    print("Input Angka :")
    a = int(input())
    print(a)
    
print("Start disini... ")

inputAngka()
inputAngka()
inputAngka()

print("Stop disini... ")

print()

# Parametrized Function 
# Pesan
def pesan(m):
    print("Pesan ",m.lower())
    
pesan("Hallo")
pesan("Namanya siapa ?")

print()

# Pesan Teks nama dan no telepon
def pesan(nama, nohp):
    print("Nama : ",nama, "No Hp : ",nohp)
    
pesan("Firman", 811)
pesan("Firman", "0817174")

print()

#Pesan Teks nama dan umur
def tambahUsia(nama, umur):
    umur = umur + 0.5
    print(nama, umur)

tambahUsia("Firman Naufal", 20)

print()

# Posisi parameter
def perkenalan(namaDepan, namaBelakang):
    print("Hallo namanya adalah : ", namaDepan, namaBelakang)

perkenalan("Firman","Naufal")
perkenalan("Safira","Tifanni")

print()

perkenalan(namaDepan = "Firman2", namaBelakang = "Naufal2")
perkenalan(namaBelakang = "Safira2", namaDepan = "Tifanni2")

print()

def jumlah(a, b, c):
    print(a, "+", b, "+", c, "=", a + b + c)
    
jumlah(5,5,4)

print()

# Default Parameters
# Perkenalan 
def introduction(firstName="John",lastName="Smith"):
    print("Hello, my name is", firstName, lastName)
    
#Call the Function here
introduction(firstName="Firman")
introduction(firstName="Firman", lastName="Naufal")
introduction("Firman", lastName="Naufal")

print()

def introductionLengkap(firstName, lastName, middleName="Naufal"):
    print("Hello, my name is", firstName, middleName, lastName)
    
#Call the Function here
introductionLengkap(firstName="Firman", lastName="Naufal", middleName="Aryaputra")

print()

# Function Dengna Return
def hitungLuasLingkaran(r):
    import math
    pi = math.pi
    luas = pi * r * r
    return luas

# buat sebuah fungsi untuk menghitung luas segitiga
def hitungLuasSegitiga(a,t):
    luas = 0.5 * a * t
    return luas


x = hitungLuasLingkaran(6)
y = hitungLuasSegitiga(5,10)

print(x)
print(y)

print()

def boringFunction():
    print("'Boredom Mode' ON.")
    return 123

print("This Lesson is interesting!")
x = boringFunction()
print(x)
print("test")

print()

def jumlah(a,b):
    c = a + b
    return c

a = jumlah(3,4)
print(a)

print()

def strangeFunction(n):
    if(n % 2 == 0):
        return True

#Call the Function here

print(strangeFunction(4))
print(strangeFunction(3))

print()

def sumOfList(lst):
    sum = 0
    
    for elem in lst:
        sum += elem
    
    return sum

#Call the Function here
x = sumOfList([2,2,2,2,2])
print(x)

print()

def strangeListFunction(n):
    strangeList = []
    
    for i in range(0, n):
        print("Insert",i)
        strangeList.insert(0, i)
        print(strangeList)
        
    return strangeList

#Call the Function here
print("Hasil : ", strangeListFunction(10))

print()

# Latihan Pengguna True or False untuk mengetahui tahun kabisat
# Function Test tahun Kabisat
def isYearLeap(year):
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

testData = [1900, 2000, 2016, 1987,2004]
testResults = [False, True, True, False, True]
for i in range(len(testData)):
    yr = testData[i]
    print(yr,"->",end="")
    result = isYearLeap(yr)
    if result == testResults[i]:
        print("OK")
    else:
        print("Failed")

print()

# Scope in Function
def scopeTest():
    print(a)
    b = 120
    print(b)

#Call the Function here

a = 90
scopeTest()

print()

def myFunction():
    print("Do I know that variable?", var)

#Call the Function here
var = 1
myFunction()
print(var)

# The answer is: a variable existing outside a function 
# has a scope inside the functions' bodies.

print()

def myFunction():
    global var
    var = 2
    print("Do I know that variable?", var)

#Call the Function here
var = 1
myFunction()
print(var)

# There's a special Python method which can extend a variable's scope 
# in a way which includes the functions' bodies 
# (even if you want not only to read the values, but also to modify them).

print()

def myFunction(n):
    print("I got", n)
    n += 1
    print("I have", n)

#Call the Function here
var = 1
myFunction(var)

print(var)

#changing the parameter's value doesn't propagate outside the function

print()

# Function parameters BMI 
def bmi(weight, height):
    return weight / height ** 2

print(bmi(65, 1.69))

print()

def bmi(weight, height):
    if height < 1.0 \
    or height > 2.5 \
    or weight < 20 \
    or weight > 200:
        return None
    
    return weight / height ** 2

print(bmi(70, 1.69))

print()

# Segitiga
def isItATriangle(a, b, c):
    if a + b <= c:
        return False
    if b + c <= a:
        return False
    if c + a <= b:
        return False
    
    return True

#Call the Function here
print(isItATriangle(1,1,1))
print(isItATriangle(1,2,2))

print()

def isItATriangle(a, b, c):
    return a + b > c and b + c > a and c + a > b

a = float(input("Enter the first side's length: "))
b = float(input("Enter the second side's length: "))
c = float(input("Enter the third side's length: "))

if isItATriangle(a, b, c):
    print("Congratulations - it can be a triangle.")
else:
    print("Sorry, it won't be a triangle.")

print()

# Faktorial 

def factorialFun(n):
    if n < 0:
        return None
    if n < 2:
        return 1
    
    product = 1
    for i in range(2, n + 1):
        product *= i
    return product

#Call the Function here
for n in range(1, 15): # testing
    print(n, factorialFun(n))

# Fibbonacci
def fib(n):
    if n < 1:
         return None
    if n < 3:
        return 1

    elem1 = elem2 = 1
    sum = 0
    for i in range(3, n + 1):
        sum = elem1 + elem2
        elem1, elem2 = elem2, sum
    return sum

#Call the Function here
for n in range(1, 10): #testing
    print(n, ">", fib(n))

print()

# Recursion 
#recursion 1
def fib(n):
    if n < 1:
        return None
    if n < 3:
        return 1
    return fib(n - 1) + fib(n - 2)

for n in range(1, 10): # testing
    print(n, "->", fib(n))

#recursion 2
    
def fun(a):
    if a > 30:
        return 3
    else:
        return a + fun(a + 3)

print(fun(25))

print()

# Contoh kasus Stop World Removal 
kalimat = "Ibu pergi ke pasar dengan Bapak"
stopw = ['ke','dengan','saya','dia']

def removeStop(sentence):
#     words = []
    words = [word for word in sentence.split() if word not in stopw]
#     for word in sentence.split():
#         if word not in stopw:
#             words.append(word)
            
    return ' '.join(words)
    
# print('ke' in kalimat.split())
kalimat_stop = removeStop(kalimat.lower())

print(kalimat_stop)
# output kalimat_stop: Ibu pergi pasar Bapak"

print()

# Tuples dan Dictionaries
# Perbandingan list dan Tuples
# List datanya bisa dirubah selama eksekusi program, sifatnya bernama mutable
# Sebaliknya tuple tidak bisa, sifatnya dinamakan immutable. 
# Persamaannya keduanya sama-sama bisa menyimpan banyak nilai sekaligus dalam satu tipe data (sequence type)

# Contoh Tuples
myTuple = (1, 2, True, "a string", (3, 4), [5, 6], None)
#Print Function here
print(myTuple)

myList = [1, 2, True, "a string", (3, 4), [5, 6], None]
#Print Function here
print(myList)

print()

tup1 = (1,2,3,4)
tup2 = 1.0, .5, .25, .124

print(tup1)
print(tup2)

print()

emptyTuple = ()
print(type(emptyTuple))

print()

myTuple = (1, 10, 100, 1000)

print(myTuple[0])
print(myTuple[-1])
print(myTuple[1:])
print(myTuple[:-2])

for elem in myTuple:
    print(elem, end = " ")

print()

myTuple = (1, 10, 100, 1000)

#Tuple operation ?

# myTuple.append(10000)
# del myTuple[0]
# myTuple[1] = -10

print()

myTuple = (1, 10, 100)
print(type(myTuple))

#Tuple operation
print(myTuple[0])
print(myTuple[-1])
print(myTuple[1:])
print(myTuple[:-2])
# Print Function here

print()

# Example 1
t1 = (1, 2, 3)
for elem in t1:
    print(elem)

# Example 2
t2 = (1, 2, 3, 4)
print(5 in t2)
print(5 not in t2)

# Example 3
t3 = (1, 2, 3, 5)
print(len(t3))

# Example 4
t4 = t1 + t2
t5 = t3 * 2

print(t4)
print(t5)

print()

# Contoh dari Dictionary
# In Python's world, the word you look for is named a key. The word you get from the dictionary is called a value.
# Dictionaries are unordered*, changeable (mutable), and indexed collections of data. 
# (*In Python 3.6x dictionaries have become ordered by default.

# contohnya
kamus = {"cat" : "chat", "dog" : "chien", "horse" : "cheval"}
phoneNumbers = {'boss' : 5551234567, 'Suzy' : [22657854310,52852352]}
emptyDictionary = {}

print()
print(phoneNumbers)
print(emptyDictionary)

print()

print(kamus['cat'])
print(phoneNumbers['Suzy'])
print(phoneNumbers['Suzy'][0])

print()

dict = {"horse" : "cheval", "cat" : "chat", "dog" : "chien"}

for key in dict.keys():
    print(key,dict[key])

print()

for key in sorted(dict.keys()):
    print(key, "->", dict[key])

# Items dan Values
dict = {"cat" : "chat", "dog" : "chien", "horse" : "cheval"}

for english, french in dict.items():
    print(english, "->", french)

print()

dict = {"cat" : "chat", "dog" : "chien", "horse" : "cheval"}

for french in dict.values():
    print(french)

print()

dict = {"cat" : "chat", "dog" : "chien", "horse" : "cheval"}
words = ['cat', 'lion', 'horse','mouse']

for word in words:
    if word in dict:
        print(word, "->", dict[word])
    else:
        print(word, "is not in dictionary")

print()

# Replace 
dict = {"cat" : "chat", "dog" : "chien", "horse" : "cheval"}

dict['cat'] = 'minou'
print(dict)

print()

# Menambahkan kunci baru 
dict = {"cat" : "chat", "dog" : "chien", "horse" : "cheval"}

dict['swan'] = 'cygne'
dict['animal'] = 'hewan'
print(dict)

for key in sorted(dict.keys()):
    print(key,dict[key])

print()

# Program yang menjalankan Tuples dan Dictionary secara bersamaan 
schoolClass = {}

while True:
    name = input("Enter the student's name (or type exit to stop): ")
    if name == 'exit':
        break
    
    score = int(input("Enter the student's score (0-10): "))
    
    if name in schoolClass:
        schoolClass[name] += (score,)
    else:
        schoolClass[name] = (score,)

print(schoolClass)

for name in sorted(schoolClass.keys()):
    sum = 0
    counter = 0
    for score in schoolClass[name]:
        sum += score
        counter += 1
    print(name, ":", sum / counter)

print()

def tf(sentence):
    d = {}
    for word in sentence.split():
        if word not in d:
            d[word] = 1
        else:
            d[word] += 1
    return d

kalimat = "Ibu pergi ke pasar dengan Bapak dan Ibu Ibu lain"
stopw.append('dan')
stopw.append('lain')
kalimat_stop = removeStop(kalimat.lower())
tf_kalimat = tf(kalimat_stop)

print(tf_kalimat)

print()

# Summary
# 1. Membuat dictionary dengan contoh syntax yang diberikan
myDictionary = {
    "key1" : "value1",
    "key2" : "value2",
    "key3" : "value3"
}

# 2. Mengakses item dari dictionary dengan cara yang disebutkan
polEngDict = {
    "kwiat" : "flower",
    "woda"  : "water",
    "gleba" : "soil"
}
item1 = polEngDict["gleba"]    # ex. 1
print(item1)    # outputs: soil

item2 = polEngDict.get("woda")
print(item2)    # outputs: water

# 3. Mengubah nilai yang terkait dengan kunci tertentu
polEngDict["zamek"] = "castle"
item = polEngDict["zamek"]    # outputs: castle

# 4. Menambah dan menghapus item dari dictionary
myPhonebook = {}    # an empty dictionary
myPhonebook["Adam"] = 3456783958    # create/add a key-value pair
print(myPhonebook)    # outputs: {'Adam': 3456783958}
del myPhonebook["Adam"]
print(myPhonebook)    # outputs: {}

# 5. Menampilkan cara menggunakan for loop untuk melewati dictionary
polEngDict = {
    "zamek" : "castle",
    "woda"  : "water",
    "gleba" : "soil"
}
for item in polEngDict:
    print(item)    # outputs: zamek
                   #          woda
                   #          gleba

# 6. Menampilkan cara menggunakan for loop untuk melewati keys dan values dictionary
polEngDict = {
    "zamek" : "castle",
    "woda"  : "water",
    "gleba" : "soil"
}
for key, value in polEngDict.items():
    print("Pol/Eng ->", key, ":", value)

# 7. Memeriksa apakah kunci tertentu ada dalam dictionary
polEngDict = {
    "zamek" : "castle",
    "woda"  : "water",
    "gleba" : "soil"
}
if "zamek" in polEngDict:
    print("Yes")
else:
    print("No")

# 8. Menghapus item tertentu atau dictionary secara keseluruhan
polEngDict = {
    "zamek" : "castle",
    "woda"  : "water",
    "gleba" : "soil"
}
print(len(polEngDict))    # outputs: 3
del polEngDict["zamek"]    # remove an item
print(len(polEngDict))    # outputs: 2
polEngDict.clear()   # removes all the items
print(len(polEngDict))    # outputs: 0
del polEngDict    # removes the dictionary

# 9. Menyalin sebuah dictionary
polEngDict = {
    "zamek" : "castle",
    "woda"  : "water",
    "gleba" : "soil"
}
copyDict = polEngDict.copy()

print()

# Key Takeaway 
myTuple = (1, 2, True, "a string", (3, 4), [5, 6], None)
#print(myTuple)

myList = [1, 2, True, "a string", (3, 4), [5, 6], None]
#print(myList)

for r in myTuple[::-1]:
    print(r)

print()

emptyTuple = ()
print(type(emptyTuple))    # outputs: <class 'tuple'>

print()

myListTuple = list(myTuple)
print(myListTuple)
myTupleLagi = tuple(myListTuple)
print(myTupleLagi)

print()

oneElemTup1 = ("one",  )    # brackets dan koma, kalau tidak ada komanya jadi string
oneElemTup2 = "one",     # no brackets, just a comma

print(type(oneElemTup1))
print(len(oneElemTup1+oneElemTup2))

oneElemList = ['one']
print(type(oneElemList))

print()

# Example 1
t1 = (1, 2, 3)
for elem in t1:
    print(elem)

# Example 2
t2 = (1, 2, 3, 4)
print(5 in t2)
print(5 not in t2)

# Example 3
t3 = (1, 2, 3, 5)
print(len(t3))

# Example 4
t4 = t1 + t2
t5 = t3 * 2

print(t4)
print(t5)

print()

import pprint
pp = pprint.PrettyPrinter(indent=4)

mahasiswa = [
    {
        'nim' : '08/123123/PA/123213',
        'nama' : 'Agus Budi H',
        'nilai' : [80,90,75,30]
    },
    {
        'nim' : '08/213123/PA/98120',
        'nama' : 'Edo Maulana R',
        'nilai' : [100,90,77,110]
    }    
]
mahasiswa_baru = {
    'nim' : '12/12414/PA/12312',
    'nama' : 'Vicky Putra'
}

mahasiswa.append(mahasiswa_baru)
mahasiswa[0]['nilai'].append(79)
mahasiswa[2]['nilai'] = [80,100,95,25]

#pp.pprint(mahasiswa)
pp.pprint(mahasiswa[0]['nilai'][-1])

print()

sinonim = open('files/sinonim.txt','r')
i = 1
for s in sinonim:
#     print(s)
    baris = s.split(':')
    print(baris)
    kata = baris[1].split(',')
    print(baris[0],kata)

print()

# sebuah_dict = {'Miftah':'Kurus'}
# sebuah_dict['Miftah'] += ',Cantik'
# print(sebuah_dict)

paragraf = """
Kisah ini bermula dari seorang dewa dan seorang dewi yang karena kesalahan yang dibuatnya di kayangan, akhirnya harus menjalani hukuman di dunia. Keduanya dihukum untuk berbuat kebaikan dalam hidupnya di bumi dalam bentuk seekor babi hutan dan seekor anjing. Babi hutan jelmaan dewi itu bernama Wayung Hyang, sedangkan anjing jelmaan dewa itu bernama Tumang. Wayung Hyang karena dihukum sebagai babi hutan atau celeng, maka ia berusaha melakukan berbagai kebaikan di dalam sebuah hutan. Sementara Tumang, sang anjing jelmaan dewa itu mengabdi sebagai anjing pemburu pada seorang raja yang bernama Sumbing Perbangkara.
"""

#sintaks ini
kalimat = [k.strip().upper() for k in paragraf.split('.')] #memisahkan kalimat berdasarkan titik untuk menjadi list
# print(kalimat)

#sama dengan ini
kalimat_baru = []
for k in paragraf.split('.'):
    kalimat_baru.append(k.strip())
#     if k.endswith('dunia'):
#         print("Ada Dunia pada kalimat",k)
    
    if k.find('dunia'):
        print("Ada dunia pada kalimat",k)
    
    
#versi satu satu
kalimat_a = paragraf.split('.') #memisahkan kalimat berdasarkan titik di paragraf
kalimat_a[1] =  kalimat_a[1].strip() #menghilangkan spasi
kalimat_a[2] =  kalimat_a[2].strip() #menghilangkan spasi

print()

k = "Firman Naufal "
print(k.capitalize())
print(k.endswith("Tifanni S"))