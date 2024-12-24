from msp.msp import *
import numpy as np

import time

gen = CGEngine.create_engine()

source = "++++++++++[>+++++++>" \
         "++++++++++>+++>+<<<<" \
         "-]>++.>+.+++++++..++" \
         "+.>++.<<++++++++++++" \
         "+++.>.+++.------.---" \
         "-----.>+.>."

print(source)

def putchar(n : int) -> int: pass
def getchar() -> int: pass

def bf():
    program = []
    jmp = 0
    stack = []
    for command in source:
        if   command == '+': program.append('data[ptr] = data[ptr] + 1')
        elif command == "-": program.append('data[ptr] = data[ptr] - 1')
        elif command == ">": program.append('ptr = ptr + 1')
        elif command == "<": program.append('ptr = ptr - 1')
        elif command == ".": program.append('_ = putchar(data[ptr])')
        elif command == ",": program.append('data[ptr] = getchar()')
        elif command == "[":
            before, after = f"pre_{jmp}", f"post_{jmp}"
            stack.append((before, after))
            jmp += 1
            program.append("\n".join([
                f"label('{before}')",
                f"if data[ptr] == 0: goto('{after}')",
            ]))
        elif command == "]":
            before, after = stack.pop()
            program.append("\n".join([
                f"goto('{before}')",
                f"label('{after}')",
            ]))
    # print("\n".join(program))
    return q("\n".join(program))
    
    return body

times = []
@gen.compile(env = globals(), debug = False, external = {putchar, getchar}, add_time = times)
def run(data : Int[256]) -> None:
    ptr = 0
    splice(bf())
    return

print(f"Compile time: {times}")
arr = np.zeros(30000, dtype=np.int32)

start = time.perf_counter()
run(arr)
end = time.perf_counter()
print(f"Run time: {end - start}")
