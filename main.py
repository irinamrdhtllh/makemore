words = open("names.txt", "r").read().splitlines()

count_bigram = {}
for word in words:
    chs = ["<S>"] + list(word) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        count_bigram[bigram] = count_bigram.get(bigram, 0) + 1

print(count_bigram)
