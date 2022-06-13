all: clean
	nvcc -o main interacting.cu err.cu IO.cu main.cu measureTime.cu distribution.cu erfinv.cu

clean:
	del "main"
	del "data\time.txt"
