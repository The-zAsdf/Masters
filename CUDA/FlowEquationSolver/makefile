all: clean
	nvcc -o main PRBM.cu err.cu IO.cu main.cu measureTime.cu distribution.cu erfinv.cu -run

clean:
	del "main"
	del "data\*.txt"
