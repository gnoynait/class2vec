import math
def cosine(v1, v2):
	r1 = 0
	r2 = 0
	p = 0
	for a, b in zip(v1, v2):
		p += a * b
		r1 += a * a
		r2 += b * b
	return p / (math.sqrt(r1) * math.sqrt(r2))


def norm(v):
	s = 0
	for a in v:
		s += a * a
	return math.sqrt(s)

def load_vector(file_name):
	f = open(file_name)
	table = {}
	for line in f:
		row = line.split()
		if len(row) < 3:
			continue
		table[row[0]] = [float(a) for a in row[1:]]
	f.close()
	return table

def load_records(file_name):
    f = open(file_name)
    table = {}
    for line in f:
        row = line.split()
        if len(row) < 5:
            continue
        if not row[0] in table:
            table[row[0]] = []
        table[row[0]].append(row[1:])
    f.close()
    return table
	
