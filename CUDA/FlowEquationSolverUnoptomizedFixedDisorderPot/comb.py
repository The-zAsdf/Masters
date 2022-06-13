
def getInput(fileName):
    file = open(fileName, "r")
    csvreader = csv.reader(file, delimiter =' ')
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()
    return rows

def createComb():
    rows = getInput("comb.txt")
    N = 0
    ind = []
    c = 0
    for row in rows:
        if (len(row) > 2):
            N += 1
            ind.append(c)
        c += 1

    # N-Dimensional list with combinations from indices ind
