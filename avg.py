with open('result-loadws', 'r') as o:
   d = o.read().split('\n')[:-1]
   print(sum([float(x) for x in d])/len(d), 'load-ws')

with open('result-ws', 'r') as o:
   d = o.read().split('\n')[:-1]
   print(sum([float(x) for x in d])/len(d), 'std-ws')

