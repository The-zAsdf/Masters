import pIO
from FES import runFESarg


def createCombArgs(fileName):
    rows = pIO.getInput(fileName)
    NL = []
    f1L = []
    f2L = []
    JL = []
    for row in rows:
        if (row[0] == 'N'):
            N = len(row)-1
            for i in range(1, len(row)):
                NL.append(float(row[i]))

        elif (row[0] == 'f1'):
            f1 = len(row)-1
            for i in range(1, len(row)):
                f1L.append(float(row[i]))

        elif (row[0] == 'f2'):
            f2 = len(row)-1
            for i in range(1, len(row)):
                f2L.append(float(row[i]))

        elif (row[0] == 'J'):
            J = len(row)-1
            for i in range(1, len(row)):
                JL.append(float(row[i]))

    e = float(pIO.readInput('e', rows))
    h = float(pIO.readInput('h', rows))
    S = float(pIO.readInput('S', rows))
    c = float(pIO.readInput('c', rows))

    # print("NL({}): ".format(N),NL)
    # print("f1L({}):".format(f1),f1L)
    # print("f2L({}):".format(f2),f2L)
    # print("JL({}): ".format(J),JL)
    comb = []
    T = N*f1*f2*J
    Np = f1*f2*J
    f1p = f2*J
    f2p = J
    for i in range(0,T):
        comb.append([])

    for i in range(0,N):
        for j in range(0,f1):
            for k in range(0,f2):
                for q in range(0,J):
                    comb[i*Np+j*f1p+k*f2p+q].append(f1L[j]*JL[q])
                    comb[i*Np+j*f1p+k*f2p+q].append(JL[q])
                    comb[i*Np+j*f1p+k*f2p+q].append(f2L[k]*JL[q])
                    comb[i*Np+j*f1p+k*f2p+q].append(h)
                    comb[i*Np+j*f1p+k*f2p+q].append(S)
                    comb[i*Np+j*f1p+k*f2p+q].append(NL[i])
                    comb[i*Np+j*f1p+k*f2p+q].append(e)
                    comb[i*Np+j*f1p+k*f2p+q].append(c)

    # print(comb)
    return comb

def runManyFES():
    input = "comb.txt"
    output = "data/combOutput.txt"

    args = createCombArgs(input)
    file = open(output, 'w')
    file.write("W/J,D/J,L,Invar,Log(Invar)\n")
    file.close()
    for arg in args:
        runFESarg(arg,output)

if __name__ == '__main__':
    runManyFES()
