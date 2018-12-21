import sys


def format_convert(file_name):
    ndim = 0
    with open(file_name, 'r') as f:
        data = f.readline()
        output = []
        while data != '':
            data = data.split()
            for feature in data[1:]:
                ndim = max(ndim, int(feature.split(':')[0]))
            data = f.readline()

    with open(file_name, 'r') as f:
        with open(file_name + "_converted", 'w') as g:
            data = f.readline()
            while data != '':
                d = dict()
                data = data.split()
                output = data[0]
                for feature in data[1:]:
                    ft = feature.split(':')
                    d[int(ft[0])] = ft[1]
                for dim in range(1, ndim + 1):
                    output = output + ' ' + str(dim) + ':' + str(d.get(dim, 0))
                g.writelines(output + '\n')
                data = f.readline()


if __name__ == '__main__':

    #print('Invalid Argument! Inputthe data file ')
    print(sys.argv)
    for file_name in sys.argv[1:]:
        format_convert(file_name)
