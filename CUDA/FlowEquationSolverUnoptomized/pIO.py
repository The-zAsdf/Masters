import csv

# Read the runtime of FES from file
def getRuntime():
    file = open("data/time.txt", "r")
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()
    return rows[0][0]

# Interprets array of input
def readInputReturnEntries(char, input):
    for entry in input:
        if entry[0] == char:
            return entry
    raise Exception("ERROR: Input not found")
    quit(-1)

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

def getDFES(L):
    # Read from file, get D(l = inf) entries
    file = open("data/D.txt", "r")
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    c = -1
    index = -1
    count = 1
    # find the index of the last entries
    for i in range(1,len(rows)):
        if int(rows[i][0]) > c:
            c = int(rows[i][0])
            index = count
        count += 1
    # store D values in q array
    q = []
    for i in range(L):
        q.append([])
        for j in range(L):
            q[i].append(float(rows[index+i*L+j][4]))
    # Reverse the array
    # for i in range(L):
    #     q[i] = q[i][::-1]
    # q = q[::-1]
    return q

def gethFES(L):
    # Read from file, get h(l = inf) entries
    file = open("data/h.txt", "r")
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    c = -1
    index = -1
    count = 1
    # find the index of the last entries
    for i in range(1,len(rows)):
        if int(rows[i][0]) > c:
            c = int(rows[i][0])
            index = count
        count += 1

    # store h values in q array
    q = []
    for i in range(L):
        q.append(float(rows[index+i][3]))

    # return reversed list, corresponds to ED evals
    return q
