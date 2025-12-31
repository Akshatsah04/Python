import time
age = int(input("enter your age: "))

if age>=18:
    print("you are an adult")
else:
    for i in range(100 , 120 ,2):
     print(f"you are not an adult {i}")
     time.sleep(1)
