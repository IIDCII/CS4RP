import random
import string

def load_random_data(length, num_sequences):
    seq = []
    for _ in range(num_sequences):
        tokens = ''.join(random.choices(string.ascii_lowercase + ' ', k=length))
        seq.append(tokens)
    return seq

s = load_random_data(100,10)

print (s)