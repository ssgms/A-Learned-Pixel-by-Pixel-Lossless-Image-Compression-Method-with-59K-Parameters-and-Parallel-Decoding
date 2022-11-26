import os
import json
import numpy as np
import matplotlib.pyplot as plt

def parse_json(fname):
    with open(fname, 'r') as f:
        try:
            data = json.load(f)
            array = []
            for d in data:
                array.append([data[d]['dist'], data[d]['rate']])
            array = np.array(array)
            return array[:,0], array[:,1]
            
        except ValueError:
            print("INVALID JSON file format..")
            exit(-1)

exps =  [f for f in os.listdir('.') 
                    if os.path.isdir(f)]
for folder in exps:
    fname = os.path.join(folder, 'results.json')
    y, x = parse_json(fname)
    plt.scatter(x,y)

plt.ylabel('rate')
plt.xlabel('dist')
plt.legend(exps)
plt.show()
