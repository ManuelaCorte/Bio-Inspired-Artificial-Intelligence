# !/usr/bin/python
import sys
import pickle
from .exercise_maze import eval

if __name__ == "__main__":
    fname = sys.argv[1]
    inputs = pickle.load(open(fname, "rb"))
    eval(inputs[0], inputs[1], inputs[2], {}, True)
