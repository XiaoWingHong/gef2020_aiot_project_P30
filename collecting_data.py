import time

times = 50
for i in range(times - 1):
    print("Times left: " + str(times - i - 1))
    time.sleep(0)
    exec(open("testing_and_making_data.py").read())