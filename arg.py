class ARG:
    def argmax(self, f):
        argument = []
        val = float('-inf')
        for num in f:
            if f[num] > val:
                argument = [num]
                val = f[num]
            elif f[num] == val:
                argument.append(num)
        return argument

    def argmin(self,f):
        argument = []
        val = float('inf')
        for num in f:
            if f[num] < val:
                argument = [num]
                val = f[num]
            elif f[num] == val:
                argument.append(num)
        return argument

if __name__ == '__main__':
    from collections import defaultdict
    from random import seed, randint

    seed(27)
    f = defaultdict(int)
    for i in range(randint(3,35)):
        f[randint(-1000,1000)] = randint(-15000,15000)
    
    print(f,'\n')
    val = ARG()
    print(f'The argument max is {val.argmax(f)}')
    print(f'The argument min is {val.argmin(f)}')

