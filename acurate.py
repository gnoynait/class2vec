answer = []
result = []
with open('answer.dat', 'r') as f:
    for line in f:
        answer.append(line.strip())
with open('result.dat', 'r') as f:
    for line in f:
        result.append(line.strip())
count = 0
ig = 0
for a, r in zip(answer, result):
    if r == 'IGNORE':
        ig += 1
        continue
    if a == r:
        count += 1
print count * 1.0 / (len(answer) - ig)
