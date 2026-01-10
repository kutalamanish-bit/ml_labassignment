import random
import statistics
import numpy as np

# 1. Count pairs with sum = 10
def count_pairs_with_sum_10(lst):
    count = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] + lst[j] == 10:
                count += 1
    return count


# 2. Range of real numbers
def range_of_real_numbers():
    n = int(input("Enter number of elements (>3): "))
    numbers = []

    for i in range(n):
        numbers.append(float(input(f"Enter element {i+1}: ")))

    return max(numbers) - min(numbers)


# 3. Matrix power
def matrix_operation():
    A = np.array([[1, 2], [3, 4]])
    m = int(input("Enter matrix power: "))
    return np.linalg.matrix_power(A, m)


# 4. Highest occurring character
def highest_count_of_char(s):
    max_char = s[0]
    max_count = 0

    for ch in s:
        if s.count(ch) > max_count:
            max_count = s.count(ch)
            max_char = ch

    return max_char, max_count


# 5. Mean, Median, Mode
def statistics_of_random_numbers():
    numbers = []

    for i in range(25):
        numbers.append(random.randint(1, 10))

    mean = sum(numbers) / len(numbers)
    median = statistics.median(numbers)
    mode = statistics.mode(numbers)

    return numbers, mean, median, mode


# MAIN FUNCTION
def main():
    print("\n--- SIMPLE LAB PROGRAM ---\n")

    # 1
    lst = list(map(int, input("Enter numbers: ").split()))
    print("Pairs with sum 10:", count_pairs_with_sum_10(lst))

    # 2
    print("\nRange:", range_of_real_numbers())

    # 3
    print("\nMatrix Power Result:\n", matrix_operation())

    # 4
    s = input("\nEnter a string: ")
    ch, cnt = highest_count_of_char(s)
    print("Highest occurring character:", ch, "Count:", cnt)

    # 5
    nums, mean, median, mode = statistics_of_random_numbers()
    print("\nRandom Numbers:", nums)
    print("Mean:", mean)
    print("Median:", median)
    print("Mode:", mode)


# PROGRAM START
main()
