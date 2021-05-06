import pickle 
import numpy

def unpickle(path):
    infile = open(path, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    print(new_dict)

unpickle("./cc18_results/results_cv/adult_results_dict.pkl")
