import sys

mode = sys.argv[1]
if mode == 'classification':
    mode = 'TreeNet classification'
else:
    mode = 'TreeNet regression'
print mode

for i in range(20):
    file_name = '{}.out'.format(i)
    reader = open(file_name, 'r')

    content = []
    flag = -1
    for line in reader.readlines():
        line = line.strip()
        if mode in line:
            # Either this line is "TreeNet classification" or "TreeNet regression"
            # Then read and rewrite
            flag = 0
        if flag >= 0:
            flag += 1
            content.append(line)
        if flag >= 18:
            break

    out = open(file_name, 'w')
    for i in range(17):
        print >> out, content[i]
    out.flush()
    out.close()