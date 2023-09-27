from functools import lru_cache
import time
import timeit
  

  
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


@lru_cache(maxsize=128)
def fibonacci_lru(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)
#print(timeit.timeit('fibonacci_lru(35)', globals=globals(), number=1))

avg_time_nocache = []
avg_time_cache = []
num_iterations = 10
for i in range(0,num_iterations-1):
  time2 = timeit.timeit('fibonacci_lru(35)', globals=globals(), number=1)
  time1= timeit.timeit('fibonacci(35)', globals=globals(), number=1)
  avg_time_nocache.append(time1)
  avg_time_cache.append(time2)

 
print("Average time for")
print("NO CACHE: " + str(sum(avg_time_nocache)/len(avg_time_nocache)))
print("CACHE: " + str(sum(avg_time_cache)/len(avg_time_cache)))