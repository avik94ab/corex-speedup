import multiprocessing
import os, time


def print_cube(num):
    """
    function to print cube of given num
    """
    print("Cube: {}".format(num * num * num))


def print_square(num):
    """
    function to print square of given num
    """
    print("Square: {}".format(num * num))

if __name__ == "__main__":

    processes = []
    for i in range(os.cpu_count()):
        processes.append(multiprocessing.Process(target=print_square, args=(10, )))

    print("Here")

    start_time = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print(time.time() - start_time)

    start_time = time.time()
    for i in range(4):
        print_square(10)
    print(time.time() - start_time)







