print ("Always executed")
import numpy as np
truePos_count = 0
falsePos_count = 0

#precision1 = float(truePos_count) /(float(truePos_count) + float(falsePos_count))
#print(precision1)
precision2 = truePos_count / np.float32(truePos_count + falsePos_count)
print(precision2)












  
        
if __name__ == "__main__":
    print ("Executed when invoked directly")
else:
    print ("Executed when imported")

