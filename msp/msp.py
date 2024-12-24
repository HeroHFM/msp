from .engine import LLVMEngine, NDArray, Int, Bool, Float

import ast

# Quote
def q(expr):
    res = ast.parse(expr, mode='exec')
    # print(ast.dump(res, indent=2))
    return res

# Code generation engine
class CGEngine:
    backends = ['llvmlite']

    def __init__(self): raise Exception("Use create_engine to instantialize a backend.")

    @staticmethod
    def create_engine(backend = backends[0]):
        assert backend == "llvmlite", f"Currently supported backends: {backends}"

        return LLVMEngine()
