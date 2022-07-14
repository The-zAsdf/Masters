import random as rd
import math as m
import csv

def writeh(h, L, ext):
    file = open("input/h{}.txt".format(ext), "w")
    for i in range(L):
        file.write(f"{h[i]}\n")
    file.close()

# Interprets the input
def readInput(char, input):
    for entry in input:
        if entry[0] == char:
            return float(entry[1])
    raise Exception("ERROR: Input not found")
    quit(-1)

# Reads input from file
def getInput(fileName):
    file = open(fileName, "r")
    csvreader = csv.reader(file, delimiter =' ')
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()
    return rows

def hr(phi,W,i,th=0):
    return W*m.cos(2.0*i*m.pi/phi+th)

def hBR(W, L):
    BR = (3.0+m.sqrt(13.0))/2.0
    rd.seed()
    h = []
    for i in range(L):
        h.append(hr(BR,W,i))
    writeh(h,L,"")

def hSR(W, L):
    SR = (1.0+m.sqrt(2.0))
    rd.seed()
    h = []
    for i in range(L):
        h.append(hr(SR,W,i))
    writeh(h,L,"")

def hGR(W, L):
    GR = (1.0+m.sqrt(5.0))/2.0
    rd.seed()
    h = []
    for i in range(L):
        h.append(hr(GR,W,i))
    writeh(h,L,"")

def main():
    rd.seed()

    rows = getInput("input.txt")
    W = readInput('W', rows)
    L = int(readInput('N', rows))

    BR = (3.0+m.sqrt(13.0))/2.0
    SR = (1.0+m.sqrt(2.0))
    GR = (1.0+m.sqrt(5.0))/2.0

    h_BR = []
    h_SR = []
    h_GR = []
    for i in range(L):
        h_BR.append(hr(BR,W,i))
        h_SR.append(hr(SR,W,i))
        h_GR.append(hr(GR,W,i))

    writeh(h_BR,L,"_BR")
    writeh(h_SR,L,"_SR")
    writeh(h_GR,L,"_GR")

if __name__ == "__main__":
    main()
