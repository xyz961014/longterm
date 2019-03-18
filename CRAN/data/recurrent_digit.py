import argparse
import ipdb

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=100, help="cycling period, namelu length of pi in decimal part")
    parser.add_argument("--size", type=float, default=1.0, help="size of pseudo data (MB)")
    parser.add_argument("--save", type=str, default="pseudo_data.txt", help="name of pseudo data")
    return parser.parse_args(args)


def getpi(N):
    N1 = N + 10
    b = 10 ** N1
    x1 = b * 4 // 5
    x2 = b // -239
    he = x1 + x2
    N *= 2
    for i in range(3, N, 2):
        x1 //= -25
        x2 //= -57121
        x = (x1 + x2) // i
        he += x
    pi = he * 4
    pi //= 10 ** 10
    pistring = str(pi)[1:]
    return pistring


def main(args):
    pi_period = getpi(args.period)
    num_of_period = int(1048576 * args.size) // (2 * len(pi_period))
    with open(args.save, "w") as f:
        for i in range(num_of_period):
            for char in pi_period:
                f.write(char + " ")

if __name__ == "__main__":
    args = parse_args()
    main(args)

