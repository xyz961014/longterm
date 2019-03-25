import argparse
import ipdb

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_start", type=int, default=1, help="start of pi in decimal part")
    parser.add_argument("--period_end", type=int, default=100, help="end of pi in decimal part")
    parser.add_argument("--size", type=float, default=1.0, help="size of pseudo data (MB)")
    parser.add_argument("--save", type=str, default="pseudo_data.txt", help="name of pseudo data")
    return parser.parse_args(args)


def getpi(end, start=1):
    N1 = end + 10
    b = 10 ** N1
    x1 = b * 4 // 5
    x2 = b // -239
    he = x1 + x2
    end *= 2
    for i in range(3, end, 2):
        x1 //= -25
        x2 //= -57121
        x = (x1 + x2) // i
        he += x
    pi = he * 4
    pi //= 10 ** 10
    pistring = str(pi)[start:]
    return pistring


def main(args):
    pi_period = getpi(args.period_end, args.period_start)
    length = int(1048576 * args.size) // 2 
    with open(args.save, "w") as f:
        i = 0
        l = []
        for char in pi_period:
            f.write(char + " ")
            i += 1
            l.append(int(char))
        while i < length:
            num = sum(l) % 10
            f.write(str(num) + " ")
            l.append(num)
            del l[0]
            i += 1

if __name__ == "__main__":
    args = parse_args()
    main(args)

