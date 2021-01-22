import mmap
import sys
import getopt

def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    answer = "".join([ch for ch in ins if ch in (ascii_letters)])
    answer = answer[1:]
    answer = answer[:-1]
    answer = answer[:-1]
    return answer


def run_count(filename):

    counts = dict()
    i = 1
    f = open(filename, 'rb')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    data = m.readline()

    while data:


        if i==2:
            data = ExtractAlphanumeric(str(data))
            counts[data]=1
        else:
            if (i+2) % 4==0:
                data = ExtractAlphanumeric(str(data))
                if data in counts.keys():
                    counts[data]+=1
                else:
                    counts[data]=1
        data = m.readline()
        i += 1
    return counts



def save_dict_to_file(dic, outputfile):
    f = open(outputfile,'w')
    for key,value in dic.items():
        f.write(key + " " + str(value) + "\n")
    f.close()


def main(argv):
    inputfile = ""
    outputfile = ""

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    counts = run_count(inputfile)
    save_dict_to_file(counts, outputfile)




if __name__ == "__main__":
   main(sys.argv[1:])