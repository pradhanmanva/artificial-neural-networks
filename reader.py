import csv


def change_csv(path):
    with open(path) as csv_file:
        # reading code is from https://docs.python.org/3/library/csv.html#csv.Sniffer
        dialect = csv.Sniffer().sniff(csv_file.read(1024))
        csv_file.seek(0)
        reader = csv.reader(csv_file, dialect)
        output = [];
        row_size = 0;  # tracking number of rows with this one
        for row in reader:
            # only going to check row size the first time around
            if row_size == 0:
                row_size = len(row)

            # we will create a list for all values except last and convert to tuple
            list_except_last = []
            for i in range(row_size):
                list_except_last.append(row[i - 1])

            # convert our list to tuple
            tuple_except_last = tuple(list_except_last)

            # add the final element
            complete_tuple = (tuple_except_last, (row[row_size - 1]))

            output.append(complete_tuple)
    return output
