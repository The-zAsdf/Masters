all: clean
	nvcc -o main PRBM.cu err.cu IO.cu main.cu measureTime.cu -run

clean:
	del "main"
	del "data\*.txt"
