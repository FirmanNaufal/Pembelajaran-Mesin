# Nama     : Firman Naufal Aryaputra
# NIM      : A11.2022.14181
# Kelompok : A11.4419

# Materi Mutable dan Immutable
a=20
print(a)
print(id(a))
a=30
print(a)
print(id(a))
a=[1,2,3,4]
print(a)
print(type(a))
print(id(a))
a[0]=0
print(a)
print(id(a))

print()

# Latihan Questions and Answers 
# Mengetahui apabila inputan angka lebih dari maka true apabila kurang dari 100 false
a=int(input("Sample input:"))
print("Expected output:", a>=100)

print()

# Operator Aritmatika 
x = 5 + 2 * 3 / (2 - 3) ** 2
print(x)

print()

# Struktur Pemilihan 
# 1. Statment if dalam satu kasus 

a = 5
b = 6

if a==b:
    print("oke")
    print("berhasil")
    print("Gagal")

print()

# 2.Statment if dalam dua kasus 

a = 5
b = 6

if a==b:
    print("oke")
    print("berhasil")
else:
    print("Gagal")

print()

# 3.Statment if dalam tiga
# Latihan 1 
bil = int(input("Masukan bilangan bulat: "))

if(bil%2 == 0):
    print("genap")
else:
    print("ganjil")

print() 

# 3 Statements if dalam 3 kasus atau lebih 
# Menentukan titik kuadran berdasarkan koordinat X dan koordinat Y 
print("Masukan nilai koordinat")
x = int(input("Nilai x: "))
y = int(input("Nilai y: "))

if x>0 and y>0:
    print("Koordinat ("+ str(x) + ","+ str(y) +") berada pada Kuadran I" )
elif x<0 and y>0:
    print("Koordinat ("+ str(x) + ","+ str(y) +") berada pada Kuadran II" )    
elif x<0 and y<0:
    print("Koordinat ("+ str(x) + ","+ str(y) +") berada pada Kuadran III" )
elif x>0 and y<0:
    print("Koordinat ("+ str(x) + ","+ str(y) +") berada pada Kuadran IV" )
else:
    pass

print()

# Latihan 3 
# Mencari inputan terbesar dalam 3 inputan 
num1 = int(input("First Number : "))
num2 = int(input("Second Number : "))
num3 = int(input("Third Number : "))

# set number1 sebagai yang terbesar
largestNum = num1

# deteksi apakah num2 > num1
if num2 > largestNum:
    largestNum = num2

# deteksi apakah num3 > num2
if num3 > largestNum:
    largestNum = num3

print("The largest number is: ", largestNum)

print()

# Latihan menfilter tentang jenis tanaman
plant = input("Please input a best plant = ")

if plant == "Spathiphyllum":
    print("Yes - Spathiphyllum is the best plant ever!")
elif plant == "spathiphyllum":
    print("No, I want a big Spathiphyllum!")
else:
    print("Spathiphyllum! Not "+ plant +"!")

print()

# Latihan tentang Tax Calculator
income = float(input("Enter the annual income: "))
tax = 0

if income<85528:
    tax = 0.18 * income - 556.2
else:
    tax = 14839 + 0.2 + 0.32 * (income - 85528)

tax = round(tax, 0)

if tax<0:
    tax = 0

print("The tax is:", tax, "thalers")

print()

# Latihan Astronomical reason, tentang tahun
year = int(input("Enter a year: "))

if year < 1582:
    print("Not within the Gregorian calender period")
else :
    if year % 4 !=0 :
        print("common year")
    elif year % 100 !=0:
        print ("leap year")
    elif year % 400 !=0:
        print("common year")
    else : 
        print("leap year")

print()

# Latihan Luaran Syantax
x = 5
y = 10
z = 8

print(x > y)
print(y > z)

print()

x, y, z = 5, 10, 8

print(x > z)
print((y - 5) == x)

print()

x, y, z = 5, 10, 8
x, y, z = z, y, x

print(x > z)
print((y - 5) == x)

print()

x = 10

if x == 10:
    print(x == 10)

if x > 5:
    print(x > 5)

if x < 10:
    print(x < 10)
else:
    print("else")

print()

x = "1"

if x == 1:
    print("one")
elif x == "1":
    if int(x) > 1:
        print("two")
    elif int(x) < 1:
        print("three")
    else:
        print("four")
        
if int(x) == 1:
    print("five")
else:
    print("six")

print()

x = 1
y = 1.0
z = "1"

if x == y:
    print("one")
if y == z:
    print("two")
    print(x)
    print(y)
elif x == y:
    print("three")
else:
    print("four")

print()

# Struktur Perulangan
# 1. Statment While

i = 1

while i<10:
    print(i**2)
    i = i+2

print()

# Program menghitung ganjil dan genap dalam rangkaian bilangan
# program that reads a sequence of numbers
# and counts how many numbers are even and how many are odd
# program terminates when zero is entered

oddNumbers = 0
evenNumbers = 0

# read the first number
number = int(input("Enter a number or type 0 to stop: "))

# 0 terminates execution
while number != 0:
    # check if the number is odd
    if number % 2 == 1:
        # increase the oddNumbers counter
        oddNumbers += 1
    else:
        # increase the evenNumbers counter
        evenNumbers += 1
    # read the next number
    number = int(input("Enter a number or type 0 to stop: "))

# print results
print("Odd numbers count:", oddNumbers)
print("Even numbers count:", evenNumbers)

print()

# Latihan SecretNumber
secretNum = 777

guessNum = 0

while guessNum != secretNum:
    guessNum = int(input("Input angka rahasia untuk berhenti : "))
    if guessNum != secretNum:
        print("Perulangan akan terus lanjut...")

print("Well done, Congrats...")

print()

# Statment for 
for i in range(5):
    print(i)
print()
for i in range(2,8):
    print(i)
print()
for i in range(2,18,3):
    print(i)
print()
for i in range(5):
    print(i)
else:
    print("masuk else")

print()

# Latihan 
import time

for i in range(5):
    print(i," Missisipi")
    time.sleep(1)

print()

# Statment Loncat 
# 3.1 Statement Break 

for i in range (11):
    print(i, end=" ")
    if i==7:
        print(" masih dalam for")
        break
print()
# break - example

print("The break instruction:")
for i in range(1, 6):
    if i == 3:
        break
    print("Inside the loop.", i)
print("Outside the loop.")
print()
largestNumber = -99999999
counter = 0

while True:
    number = int(input("Enter a number or type -1 to end program: "))
    if number == -1:
        break
    counter += 1
    if number > largestNumber:
        largestNumber = number

if counter != 0:
    print("The largest number is", largestNumber)
else:
    print("You haven't entered any number.")

print()

# 3.2 Statement Continue
for i in range (1,11):
    if i % 2 ==0:
        continue
    print(i, end="\t")
print()
for i in range (1,11):
    if i % 2 ==0:
        print(i)
print()
# continue - example

print("\nThe continue instruction:")
for i in range(1, 6):
    if i == 3:
        continue
    print("Inside the loop.", i)
print("Outside the loop.")
print()
largestNumber = -99999999
counter = 0

number = int(input("Enter a number or type -1 to end program: "))

while number != -1:
    if number == -1:
        continue
    counter += 1

    if number > largestNumber:
        largestNumber = number
    number = int(input("Enter a number or type -1 to end program: "))

if counter:
    print("The largest number is", largestNumber)
else:
    print("You haven't entered any number.")
print()

# Latihan break
while True:
    word = input("Masukkan Sebuah Kata untuk keluar dari perulangan = ")
    if(word=="chupacabra"):
        break
        
print("You've successfully left the loop")

print()

# Latihan Coountinue 
# Prompt the user to enter a word
# and assign it to the userWord variable.
userWord = input("Masukkan sebuah kata = ")

for letter in userWord:
    if letter.upper() in ['A','I','U','E','O']:
        continue
    
    print(letter.upper())
print()

# Latihan Blocks
blocks = int(input("Enter the number of blocks: "))
current_block = 0
height = 0
i=0

while i<blocks:
    height += 1
    current_block += height
    # print(i,current_block)
    if current_block> blocks:
        break

height -= 1

print("The height of the pyramid:", height)

print()

# Latihan Perulangnan
for i in range(1, 11):
    if i % 2 !=0:
        print(i,end=' ')

print()

x=1
while x<11:
    if x % 2 !=0:
        print(x,end=' ')
    x +=1

print()

for ch in "ratna.lestari@ugm.ac.id":
    if ch == "@":
        break
    print(ch,end=' ')

print()

for digit in "0165031806510":
    if digit == "0": 
        print ("x")
        continue
    print(digit)

print()

n = 3

while n > 0:
    print(n + 1)
    n -= 1
else:
    print(n) 

print()

n = range(4)

for num in n:
    print(num - 1)
else:
    print(num)

print()

for i in range(0, 6, 3):
    print(i)

print()

# Operasi Logika 

x = 1
y = 0

z = ((x == y) and (x == y)) or not(x == y)
print(not(z))

print()

# Lish dan Array 

var1=1
var2=2
var3=3
var4=4
var5=5

# Bandingkan

var=[1,2,3,4,5]
print(var) 

print()

# Indxing List

num = [2,3,4,5]
print(num)
print(num[2])
print(num[-1])

print()

# Indexing yang bernilai negatif
lst = [100,200,300,400,500]
print(lst[-1])
print(lst[-2])
print(lst[-5])

print("\n")
print(lst[4])
print(lst[3])
print(lst[0])

print()

# Slicing List 
lst=[4,1,6,9,10,8]

print(lst[0:3])
print(lst[1:3])

print(lst[-3:])
print(lst[:-3])

print(lst[:])

print()

# Slicing Swap 
num=[2,3,4]
num[0],num[1] = num[1], num[0]
print(num)

print()

# Bubble Short
myList = [8, 10, 6, 2, 4] # list to sort
swapped = True # it's a little fake - we need it to enter the while loop

while swapped:
    swapped = False # no swaps so far
    for i in range(len(myList) - 1):
        if myList[i] > myList[i + 1]:
            swapped = True # swap occured!
            myList[i], myList[i + 1] = myList[i + 1], myList[i]

print(myList)

print()

# Basic List 
hatList = [1, 2, 3, 4, 5]  # Ini adalah sebuah list yang sudah ada

# Step 1: Meminta pengguna untuk mengganti angka tengah dalam list
hatList[2] = int(input("Masukkan angka baru untuk mengganti angka tengah: "))

# Step 2: Menghapus elemen terakhir dari list
hatList.pop()

# Step 3: Mencetak panjang list yang sudah ada
print("Panjang list sekarang:", len(hatList))

print(hatList)

print()

# Menambah Elemen kedalam switch 

buah=["Salak","Mangga","Apel"]
buah.append("Jeruk")
print(buah)

print()

myList = [] # creating an empty list

for i in range(5):
    myList.append(i + 1)

print(myList)

print()

# Menggunakan metode insert 

buah = ["durian", "mangga", "apel"]
buah.insert(1,"kiwi")
print (buah)

print()

numbers = [111, 7, 2, 1]
print(len(numbers))
print(numbers)

###

numbers.append(4)

print(len(numbers))
print(numbers)

###

numbers.insert(0, 222)
print(len(numbers))
print(numbers)

#

print()

# Menghapus elemene ke dalam list 
num = [5,4,3,2,1]

del num[1]
print(num)

print ()

# Latihan 

# step 1
beatles = []
print("Step 1:", beatles)

# step 2
beatles.append("John Lennon")
beatles.append("Paul McCartney")
beatles.append("George Harrison")
print("Step 2:", beatles)

# step 3
for i in range(2):
    nama = input("Masukkan anggota baru = ")
    beatles.append(nama)
print("Step 3:", beatles)

# step 4
del beatles[-2:]
print("Step 4:", beatles)

# step 5
beatles.insert(0,"Ringo Starr")
print("Step 5:", beatles)


# testing list legth
print("The Fab", len(beatles))

print()

# Tambahan Method 
lst = [5, 1, 3, 2, 10]
print(lst)
lst.sort()
print(lst)
lst.reverse()
print(lst)

print()

vehiclesOne = ['car', 'bicycle', 'motor']
print(vehiclesOne) # outputs: ['car', 'bicycle', 'motor']

vehiclesTwo = vehiclesOne
del vehiclesOne[0] # deletes 'car'
print(vehiclesTwo) # outputs: ['bicycle', 'motor']
print(vehiclesOne)

print()

nama = ["Ana","Budi","Citra"]
kata_kunci = "Budi"

print(kata_kunci not in nama)

if kata_kunci in nama:
    print("Nama "+kata_kunci+" ada di list")

print()

for i in range(10):
    print(i, end=" ")
    
print()
for i in range(1,7):
    print(i, end=" ")
    
print()
for i in range(1,10,2):
    print(i, end=" ")
    
print()
for i in range(10,-10,-2):
    print(i, end=" ")

print()

for i in range(1, 11):
    if i%2 != 0:
        print(i, end=" ")

print()

x = 1
while x < 11:
    if x%2!=0:
        print(x, end=" ")
    x += 1

print()

for ch in "john.smith@pythoninstitute.org":
    if ch == "@":
        break
    print(ch,end="")

print()

for digit in "0165031806510":
    if digit == "0":
        print("x",end="")
        continue
    print(digit,end="")

print()

# List Method (Advance)
myList = [10, 1, 8, 3, 5]
total = 0

for i in range(len(myList)):
    total += myList[i]

print(total)

print()

#squares = [x ** 2 for x in range(10)]
#print (squares)

squares = []
for x in range(10):
    squares.append(x ** 2)
print(squares)

print()

twos = [2 ** i for i in range(8)]
print(twos)

print()

squares = []
odds = [x for x in squares if x % 2 != 0 ]
print (odds)

# sama dengan
odds = []
for x in squares:
    if x % 2 != 0:
        odds.append(x)
print (odds)

print()

# List Dua Dimensi Array

# Contoh List di dalam List
i = [2,3,4,5]
print(i[0])

k = [[3,5,1],[6,1,8],[9,10,4]]
print(k)
print(k[1])
print(k[1][0])
print(k[-1][-2:])

print()

EMPTY = "-"
board = []

for i in range(8):
    row = [EMPTY for i in range(8)] #membuat satu list
    board.append(row) #membuat 8 list
    print(row)
#print(board)

# Contoh list dari papan catur 

EMPTY = "-"
LIMALIMA = "LIMA"
ROOK = "ROOK"
KNIGHT = "KNIGHT"
PAWN = "PAWN"
board = []

for i in range(8):
    row = [EMPTY for i in range(8)]
    board.append(row)

#isi dengan ROOK
board[0][0] = ROOK
board[5][5] = LIMALIMA
board[0][7] = ROOK
board[7][0] = ROOK
board[7][7] = ROOK

#isi dengan KNIGHT
board[4][2] = KNIGHT

#isi dengan PAWN
board[3][4] = PAWN


print(board)
print("\n")

print()

# List MultiDimensi Array 

temps = [[0.0 for h in range(2)] for d in range(4)]
print (temps)

print()

temps = [[0.0 for h in range(24)] for d in range(31)]

# 1. the monthly average noon temperature
sum = 0.0

for day in temps:
    sum += day[11]

average = sum / 31

print("Average temperature at noon:", average)

# 2. the highest temperature during the whole month 
highest = -100.0

for day in temps:
    for temp in day:
        if temp > highest:
            highest = temp

print("The highest temperature was:", highest)

# 3.count the days when the temperature at noon was at least 20 ℃
hotDays = 0

for day in temps:
    if day[11] > 20.0:
        hotDays += 1

print(hotDays, "days were hot.")

print()

# Array 3 Dimensi 

# Kode penghasil Array
rooms = [[[False for r in range(2)] for f in range(3)] for t in range(4)]
print (rooms)

print()

# Contoh Imputan berdaasarkan kondisi 
rooms = [[[False for r in range(20)] for f in range(15)] for t in range(3)]

rooms[1][9][13] = True
rooms[0][4][1] = False

print(rooms)

vacancy = 0

for roomNumber in range(20):
    if not rooms[2][14][roomNumber]:
        vacancy += 1
        
print(roomNumber)