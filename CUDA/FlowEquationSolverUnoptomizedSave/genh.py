import random as rd
import math as m
import csv

def writeh(h, L):
    file = open("input/h.txt", "w")
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

def hr(phi, W):
    return W*m.cosh(2.0*m.pi/phi)*m.cos(2.0*m.pi*rd.random())

def inith(W,L,phi):
    h = []
    for i in range(L):
        h.append(hr(phi,W))

    writeh(h,L)

def hBR(W,L):
    inith(W,L,(3.0+m.sqrt(13.0))/2.0)

def hSR(W,L):
    inith(W,L,(1.0+m.sqrt(2.0)))

def hGR(W,L):
    inith(W,L,(1.0+m.sqrt(5.0))/2.0)

def main():
    rd.seed()

    rows = getInput("input.txt")
    W = readInput('W', rows)
    L = int(readInput('N', rows))

    BR = (3.0+m.sqrt(13.0))/2.0
    SR = (1.0+m.sqrt(2.0))
    GR = (1.0+m.sqrt(5.0))/2.0

    h = []
    for i in range(L):
        h.append(hr(BR,W))
        # h.append(hr(SR,W))
        # h.append(hr(GR,W))

    writeh(h,L)

if __name__ == "__main__":
    main()
