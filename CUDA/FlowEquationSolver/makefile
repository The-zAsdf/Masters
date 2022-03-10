all: clean
	nvcc -o main PRBM.cu err.cu IO.c main.c -run

clean:
	del "main"
	del "data\*.txt"
