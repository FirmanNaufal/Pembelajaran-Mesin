# Nama     : Firman Naufal Aryaputra
# NIM      : A11.2022.14181
# Kelompok : A11.4419

# Object Oriented Programming, atau pemrograman berorientasi objek. Nah selama ini, kita sebetulnya menggunakan pendekatan pemrograman prosedural. 
# Jadi, objek itu memiliki dua karakteristik.
# 1. attributes
# 2. behavior

# Mari kita lihat contohnya:
# Parrot (burung beo) adalah sebuah objek,
# name, age, color (nama, usia, warna) adalah atributnya
# singing, dancing (menyanyi, menari) adalah behavior nya

# Konsep dari OOP dalam python ini berfokus dalam pembuatan kode yang reusable (dapat digunakan kembali). 
# Konsep ini juga dikenal dengan DRY (Don't Repeat Yourself).
# Dalam python, konsep OOP memiliki beberapa prinsip dasar:
# | Sifat         | Deskripsi                                                                                   |
# | Inheritance   | Proses dalam menggunakan atribut dan behaviour dari class yang telah ada sebelumnya.        |
# | Encapsulation | Menyembunyikan atribut dan behavior yang bersifat private dari kelas lainnya.               |
# | Polymorphism  | Sebuah konsep untuk menggunakan operasi yang sama dengan cara yang berbeda pada kelas lain. |

class Parrot: # ini merupakan blueprint yang dimana nantinya akan direalisasikan dalam sebuah instance

    # class attribute
    species = "bird"

    # instance attribute
    # constructor
    def __init__(self, name, age):
        self.name = name
        self.age = age

# # instantiate the Parrot class (Merupakan objek dari sebuah class)
papi = Parrot("Papi", 3)
greeny = Parrot("Greeny", 5)

# access the class attributes
print("Papi is a " + papi.species)
print("Greeny is also a " + greeny.species)

# access the instance attributes
print(papi.name + " is " + str(papi.age) + " years old")
print(greeny.name + " is " + str(greeny.age) + " years old")

print()

## Objek dari class Siswa
## Class ini menjelaskan definisi dari 
class Siswa:
    
    def __init__(self, nis, nama, alamat): # Merupakan constructor method yang menginisialisasi dari atribut baru sisaw
        self.nis = nis
        self.nama = nama
        self.alamat = alamat
        self.__mata_pelajaran = []
        self.__nilai = []
        
    def tampilkan(self): # Untuk menampilkan informasi dari siswa
        print("="*20)
        print(self.nis,self.nama,self.alamat)
        self.tampilkanNilai()
        self.tampilkanRataNilai()
        print("="*20)
    
    def tambahNilai(self, mapel, n): # Menambahkan subject dan representatif dari list
        self.__mata_pelajaran.append(mapel)
        self.__nilai.append(n)
        
    def tampilkanNilai(self): # Menampilkan subject dan grade
        for i,j in enumerate(self.__mata_pelajaran):
            print(i+1, self.__mata_pelajaran[i],self.__nilai[i])
    
    def tampilkanRataNilai(self): # Menampilkan rata-rata nilai 
        jumlah = 0                # Rumus untuk menghitung hasil rata rata
        for n in self.__nilai:
            jumlah += n
        rata = jumlah / len(self.__nilai)
        print("Rata-Rata Nilai",rata)
        
# objek dari class SiswaSMK
class SiswaSMK(Siswa):
    
    def __init__(self, nis, nama, alamat, jurusan):
        self.jurusan = jurusan
        super().__init__(nis, nama, alamat)
        
    def tampilkan(self):
        print("="*20)
        print(self.nis,self.nama,self.alamat)
        super().tampilkanNilai()
        super().tampilkanRataNilai()
        print("Jurusan : ", self.jurusan)
    
# disini untuk menjelaskan NIM dan nis lalu nama mahasiswa / siswa serta kota asalnya dan jurusan dari mahasiswa    
reza = Siswa("14181", "Firman", "Semarang")
adip = SiswaSMK("14177", "Raffa", "Pati", "Rekayasa Perangkat Lunak")

## tambah nilai ( Disini kita memasukan nama dari mata pelajaran / mata kuliah dan nilai dari mata kuliah/pelajaran tersebut)
reza.tambahNilai("Matematika", 98)
reza.tambahNilai("IPA", 95)
reza.tambahNilai("Bahasa Inggris", 100)

reza.tampilkan()

adip.tambahNilai("Pemrograman Web", 100)
adip.tambahNilai("Networking 1", 100)
adip.tambahNilai("Otomata", 80)
adip.tampilkan()

#reza.tampilkanNilai()
#reza.tampilkanRataNilai()
#adip.tampilkan()

print()

### Instance variables
# Dinamakan instance variable karena variabel tersebut adalah milik dari tiap instance/objek dari suatu kelas.
# Yang artinya, jika sebuah kelas memiliki instance variable dengan nama a, maka setiap instance dari kelas tersebut 
# akan memiliki variabel a sendiri-sendiri.
# Di dalam python, setiap variabel yang didefinisikan di dalam fungsi pada sebuah kelas (lewat variabel self),
# maka ia dikatakan instance variable.

### Class variables / Static Variable
# Adalah variabel statis yang jumlah copy-nya hanya ada satu saja selama aplikasi dijalankan.
# Misal kita memiliki sebuah kelas, dan kelas tersebut memiliki seratus instance, tetap saja static variable dari kelas tersebut 
# hanya ada satu saja di memori. Oleh karena itu, static variable juga sering dikenal dengan istilah class variable.

# Methods
# Adalah function yang didefinisikan dalam sebuah class. 
# Function ini seharusnya mendefinisikan behavior dari objeknya.

class Parrot:
    
    # instance attributes
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # instance method
    def sing(self, song): # Method Sing (Instance method) yang dipanggil pada objek "Papi"
        return "{} sings {}".format(self.name, song)

    def dance(self): # Method Dance (instance method) yang dipanggil pada objek "Papi"
        return "{} is now dancing".format(self.name)

print()
# instantiate the object
# call our instance methods

### Inheritance / Pewarisan
# Adalah sebuah cara untuk membuat class baru dengan menggunakan detail dari kelas lainnya. 
# Kelas yang baru ini akan mewariskan atribut serta method yang sudah didefinisikan dari class utamanya. 
# Kelas yang baru ini sering disebut dengan `child class` dan kelas yang digunakan detailnya sering disebut sebagai `parent class`.

class Vehicle: # Class dari Vehicle
    pass

class LandVehicle(Vehicle): # Merupkan SubClass dari Vehicle
    pass

class TrackedVehicle(LandVehicle): # Merupakan SubClass dari LandVehicle yang dimana sebelumnya merupakan SubClass dari LandVehicle dari Vehicle yang
    pass


for cls1 in [Vehicle, LandVehicle, TrackedVehicle]: # Ini merupakan Outer loop mengulangi daftar kelas sekali, dan inner loop mengulangi daftar kelas untuk setiap iterasi outer loop.
    for cls2 in [Vehicle, LandVehicle, TrackedVehicle]:
        print(cls1,cls2,issubclass(cls1, cls2), end="\r\n") # Issubclass Function berfungsi untuk mengetahui apabila class pertama merupakan subclass dari kelas kedua
    print()

print()
### How Python finds properties and methods
# Now we're going to look at how Python deals with inheriting methods.

# Take a look at the example in the editor. Let's analyze it:
# - Ada Sebuah Class dinamakan `Super`, apabila didefinisikan dengan constructor used digunakan untuk menyambungkan objek properti, named `name`.
# - Ada sebuah Class dinamakan `__str__()` method, juga, Yang dimana Class dapat digunakan untuk mempresentasikan teks yang baru.
# - the class is next used as a base to create a subclass named `Sub`. The `Sub` class defines its own constructor, which invokes the one from the superclass. Note how we've done it: `Super.__init__(self, name)`.
# - we've explicitly named the superclass, and pointed to the method to invoke `__init__()`, providing all needed arguments.
# - we've instantiated one object of class `Sub` and printed it.

# Implementasinya 
class Super:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "My name is " + self.name + "."


class Sub(Super):
    def __init__(self, name):
        Super.__init__(self, name)


obj = Sub("Andy")

print(obj)

print()

# Implementasinya di kasus yang berbeda 
# parent class
class Bird:
    
    def __init__(self):
        print("Bird is ready")

    def whoisThis(self):
        print("Bird")

    def swim(self):
        print("Swim faster")

# child class
class Penguin(Bird):

    def __init__(self):
        # call super() function
        super().__init__()
        print("Penguin is ready")

    def whoisThis(self):
        print("Penguin")

    def run(self):
        print("Run faster")

peggy = Penguin()
peggy.whoisThis()
peggy.swim()
peggy.run()

## Hal penting yang dapat diperhatikan:
# - Bird adalah parent class dari Penguin dengan sintaks `Penguin(Bird)`. Relasi pewarisan ini harus di validasi dengan hubungan "is a". 
#   Contohnya Penguin is a Bird merupakan valid karena Penguin adalah termasuk Bird.
# - `super().__init__()` memanggil konstruktor dari kelas parent nya
# - method `whoisThis()` yang ditulis ulang di `Penguin` akan menimpa atau override method yang sudah ada di parent class (`Bird`).
# - method swim dapat dipanggil oleh instance dari `Penguin` karena `Penguin` sudah mewarisi seluruh method yang ada pada kelas `Bird`.

print()

# Encapsulation
# Kita dapat membatasi akses atribut dan method dalam sebuah kelas dengan memanfaatkan sifat private yang di definisikan
# dengan garis bawah atau underscore single `_` atau double `__`

# Implementasinya 
class Rekening:
    def __init__(self):
        self.__saldo = 0
        
    def kredit(self, jumlah):
        if jumlah < 0:
            print("Gagal Kredit, Jumlah tidak bisa kurang dari 0")
            return
        
        self.__saldo += jumlah
        
    def debit(self, jumlah):
        if jumlah > self.__saldo:
            print("Gagal Debit, Jumlah melebihi Saldo!")
            return
        
        if jumlah < 0:
            print("Gagal Debit, Jumlah tidak bisa kurang dari 0")
            return
        
        self.__saldo -= jumlah
        
    def cetakSaldo(self):
        print("Saldo saat ini " + str(self.__saldo))

bniBudi = Rekening()

bniBudi.kredit(1000)
bniBudi.cetakSaldo()
bniBudi.debit(100000)

# Pada program diatas, atribut `saldo` bersifat private karena kita memberikan underscore `__saldo`. Oleh karena itu,
# kita tidak bisa meruba nya secara langsung seperti dengan sintaks `c.__saldo = 1000`. Pada umumnya,
# atribut private ini dapat kita rubah dengan melewatkan pada sebuah method yang publik.
# Dalam hal ini method tersebut adalah `kredit()` dan `debit()`.
print()

# Contoh awal Stacknya
# stack = []
# def push(val):
#    stack.append(val)
# def pop():
#    val = stack[-1]
#    del stack[-1]
#    return val
# push(3)
# push(2)
# push(1)
# print(pop())
# print(pop())
# print(pop())

# Atribut pertamanya 
# class Stack:
#    def __init__(self):
#        self.stackList = []
# stackObject = Stack()
# print(len(stackObject.stackList))

# Kemudian ubah atributnya menjadi private
# class Stack:
#    def __init__(self):
#        self.__stackList = [] # __ membuat private
# stackObject = Stack()
# print(len(stackObject.__stackList)) 

# Hasil setelah diubah atributnya menjadi private dan menambahkan method push dan popnya 
class Stack:
    def __init__(self):
        self.__stackList = []

    def push(self, val):
        self.__stackList.append(val)

    def pop(self):
        val = self.__stackList[-1]
        del self.__stackList[-1]
        return val


stackObject = Stack()

stackObject.push(3)
stackObject.push(2)
stackObject.push(1)

print(stackObject.pop()) # .pop merupakan sebuah method yang berasa dari Stack yang dimana dia menghaus dan mengembalikan dari top element stack
print(stackObject.pop())
print(stackObject.pop())

print()

# Contoh beberapa Instance 
class Stack:
    def __init__(self):
        self.__stackList = []

    def push(self, val):
        self.__stackList.append(val)

    def pop(self):
        val = self.__stackList[-1]
        del self.__stackList[-1]
        return val

littleStack = Stack() # littleStack merupakan instance baru dari stack
anotherStack = Stack() # AnotherStack merupakan instance baru dari stack
funnyStack = Stack() # FunnyStack merupakan instance baru dari Stack

littleStack.push(1)
anotherStack.push(littleStack.pop() + 1) # pop yang dimana top element yang berasal dari stacklist
funnyStack.push(anotherStack.pop() - 2) # Pop yang dimana top elemen yang berasal dari anotherstack

print(funnyStack.pop())

print()

# Inheritance
class Stack:
    def __init__(self):
        self.__stackList = []

    def push(self, val):
        self.__stackList.append(val)

    def pop(self):
        val = self.__stackList[-1]
        del self.__stackList[-1]
        return val


class AddingStack(Stack):
    def __init__(self):
        Stack.__init__(self)
        self.__sum = 0

    def getSum(self):
        return self.__sum

    def push(self, val):
        self.__sum += val
        Stack.push(self, val)

    def pop(self):
        val = Stack.pop(self)
        self.__sum -= val
        return val


stackObject = AddingStack()

for i in range(5):
    stackObject.push(i)
print(stackObject.getSum())

for i in range(5):
    print(stackObject.pop())

print()

# Contoh lain dari Inheritance
class Siswa:
    #class atribut
    __jmlMapel = 0
    __mapel = []
    __nilai = []

    def __init__(self, nama, alamat):
        self.nama = nama
        self.alamat = alamat
        
    def getNama(self):
        return self.nama
    
    def tambahNilaiMapel(self,mapel,nilai):
        self.__mapel.append(mapel)
        self.__nilai.append(nilai)
        
    def cetakNilai(self):
        for i,j in enumerate(self.__mapel):
            print(self.__mapel[i], self.__nilai[i])
    
    
s = Siswa("Firman","Semarang")
print(s.nama) # boleh karena publik
print(s.getNama()) # bisa juga
s.tambahNilaiMapel("Matematika",90)
s.tambahNilaiMapel("Biologi",80)

s.cetakNilai() # akses nilai hanya lewat sini

# print(s.nilai) # ga boleh karena private
# print(s.__nilai) # ga boleh karena private

print()

# Latihan OOP 
import math
class Circle:
    def __init__(self, radius, color):
        self.__radius = radius
        self.__color = color
        
    def getRadius(self):
        return self.__radius
    
    def setRadius(self,radius):
        self.__radius = radius
    
    def getColor(self):
        return self.__color
    
    def setColor(self,color):
        self.__color = color
    
    def getArea(self):
        return math.pi * self.__radius ** 2
    

# radius = 2, color = blue
c1 = Circle(2,"blue")
# c1.setRadius(2)
# c1.setColor("blue")
print(c1.getArea())

# radius = 2, color = red
c2 = Circle(2,"red")
# c2.setRadius(2)
# c2.setColor("red")
print(c2.getArea())

# radius = 1, color = red
c3 = Circle(1,"red")
# c3.setRadius(1)
# c3.setColor("red")
print(c3.getArea())

# Latihan OOP 2 
class Student:
    def __init__(self,name,address):
        self.__name = name
        self.__address = address
        self.__numCourses = 0
        self.__courses = []
        self.__grades = []
        
    def getName(self):
        return self.__name
    
    def getAddress(self):
        return self.__address
    
    def setAddress(self,address):
        self.__address = address
        
    def addCourseGrade(self,course,grade):
        self.__courses.append(course)
        self.__grades.append(grade)
        self.__numCourses += 1
        
    def printGrades(self):
        for i in range(len(self.__courses)):
            print(self.__courses[i] +" : "+ str(self.__grades[i]))
            
    def getAverageGrade(self):
        sum = 0
        for n in self.__grades:
            sum += n
        return sum / len(self.__grades)
            
# Kode dibawah untuk memasukan input dari jumlah siswa dan course yang dimiliki oleh siswa
# yang dimana nantinya inputan itu akan mempengaruhi berapa banyak siswa dan course yang ingin dimasukkan
n_siswa = int(input("Masukkan Jumlah Siswa = "))
n_course = int(input("Masukkkan Jumlah Course = "))

# Kode dibawah untuk memasukan input dari nama siswa, alamat siswa, mapel siswa, dan nilai siswa
siswa = []
for i in range(n_siswa):
    nama = input("Masukkan Nama Siswa = ")
    alamat = input("Masukkan Alamat Siswa = ")
    siswa.append(Student(nama,alamat))
    for j in range(n_course):
        mapel = input("Masukkan Nama Mapel Siswa "+nama+" = ")
        nilai = int(input("Masukkan Nilai Mapel Siswa "+nama+" = "))
        siswa[i].addCourseGrade(mapel,nilai)

print()

# kode dibawha ini untuk menampilkan hasil dari output yang dimasukkan sebelumnya
print("Hasil")
for i in range(n_siswa):
    print(siswa[i].getName(),siswa[i].getAddress())
    siswa[i].printGrades()
    print("Rata-Rata : ", siswa[i].getAverageGrade())

# s1 = Student("Ani","Yogyakarta")
# print(s1.getName(),s1.getAddress())
# s1.addCourseGrade("Matematika",95)
# s1.addCourseGrade("IPA",90)
# s1.addCourseGrade("Bahasa Indonesia",85)
# s1.printGrades()
# print("Rata-Rata : ", s1.getAverageGrade())

# print()
# s2 = Student("Budi","Jakarta")
# print(s2.getName(),s2.getAddress())
# s2.addCourseGrade("Matematika",60)
# s2.addCourseGrade("IPA",70)
# s2.addCourseGrade("Bahasa Indonesia",100)
# s2.printGrades()
# print("Rata-Rata : ", s2.getAverageGrade())
 
print()

# Contoh OOP 3
class Employee: # Employee merupakan objek 
    def __init__(self,idEmp,firstName,lastName,salary): # merupakan list constructor method dari class employee 
        self.__idEmp = idEmp # Merepresentasikan interger dari ID pekerja
        self.__firstName = firstName # Merepresentasikan string dari Nama pekerja
        self.__lastName = lastName # Merepresentasikan string dari nama terakkhir pekerja
        self.__salary = salary # Merepresentasikan Float dari gaji pekejra
    
    def getID(self): # return dari ID employee
        return self.__idEmp
    
    def getFirstName(self): # return dari first name employee
        return self.__firstName
    
    def getLastName(self): # return dari last name employee
        return self.__lastName
    
    def getName(self): # return nama lengkap employee
        return self.__firstName + " " + self.__lastName
    
    def getSalary(self): # return dari gaji employee
        return self.__salary
    
    def setSalary(self,salary): # Set dari gaji employee
        self.__salary = salary
        
    def getAnnualSalary(self): # Return dari Annual gaji employee
        return self.__salary * 12
    
    def raiseSalary(self,percent): # Untuk meningkatkan gaji dari employee
        self.__salary += self.__salary * (percent/100) # Apabila presentase gaji dari employee sebesar 10%
        return self.__salary
    
    def __str__(self): # return dari representasi string nama employee
        return self.getName() + " " + str(self.getSalary())
        
# Class dari Manager 
class Manager(Employee): # Class dari Manager yang dimana mewarisi dari kelas karyawan 
    def __init__(self,idEmp,firstName,lastName,salary,bonus): # Merupakan Construcor method yang berada di Manager
        super().__init__(idEmp,firstName,lastName,salary)
        self.__bonus = bonus
        
    def setBranch(self, branch): # Merupakan branch set dari manager
        self.__branch = branch
    
    def getBranch(self, branch): # Merupakan return dari branch manager
        return self.__branch
    
    def getBonus(self, bonus): # Merupakan return dari bonus manager
        return self.__bonus
    
    def setBonus(self, bonus): # Merupakan set dari bonus manager
        self.__bonus = bonus
        
    def getAnnualSalary(self): # merupakan return dari manager annual salary yang dimana nanti ditotalkan dengan menambahkan bonus dari annual salary
        return super().getSalary() * 12 + self.__bonus
        

m1 = Manager(1234,"Syifa","Firman",5000,9000)
print(m1.getName())
print(m1.getAnnualSalary())

# e1 = Employee(1234,"Guntur","Budi",5000)
# print(e1.getSalary())
# print(e1.getAnnualSalary())
# print(e1.raiseSalary(50))
# print(e1.getAnnualSalary())
# print(e1)

print()

# Turunan People
class Person:
    def __init__(self, name, address):
        self.__name = name
        self.__address = address

    def getName(self):
        return self.__name

    def getAddress(self):
        return self.__address

    def setName(self, name):
        self.__name = name

    def __str__(self):
        return self.__name + " " + self.__address
        
class Student(Person):
    def __init__(self,name,address):
        super().__init__(name,address)
        self.__numCourses = 0
        self.__courses = []
        self.__grades = []        
    
    def addCourseGrade(self,course,grade):
        self.__courses.append(course)
        self.__grades.append(grade)
        self.__numCourses += 1
        
    def printGrades(self):
        for i in range(len(self.__courses)):
            print(self.__courses[i] +" : "+ str(self.__grades[i]))
            
    def getAverageGrade(self):
        sum = 0
        for n in self.__grades:
            sum += n
        return sum / len(self.__grades)
    
    def __str__(self):
        return "Siswa : " + super().getName() + " " +super().getAddress()

class Teacher(Person):
    def __init__(self,name,address):
        super().__init__(name,address)
        self.__numCourses = 0
        self.__courses = []
        
    def addCourse(self, course):
        if course not in self.__courses:
            self.__courses.append(course)
            return True
        else:
            return False
        
    def removeCourse(self, course):
        if course in self.__courses:
            i = self.__courses.index(course)
            del self.__courses[i]
            return True
        else:
            return False
        
    def printCourse(self):
        print(self.__courses)
        
    def __str__(self):
        return "Guru : " + super().getName() + " " +super().getAddress()

t1 = Teacher("Pak Bagas","Pati")
print(t1)

t1.addCourse("Fisika")
t1.addCourse("Seni Budaya")

t1.printCourse()
if(t1.removeCourse("Biologi")):
    print("Berhasil Dihapus")
else:
    print("Gagal Dihapus")
t1.printCourse()