from .convert import GenerateIR
from .array import NDArray

from ctypes import POINTER, CFUNCTYPE, c_int, c_float, c_bool
from llvmlite import binding as llvm
import numpy as np
import inspect
import time
import ast

Int   = NDArray(int)
Bool  = NDArray(bool)
Float = NDArray(float)

# Ctypes conversion
def get_ctypes_sig(tree, env = {}):

    ctype_map = { 'int' : c_int, 'float' : c_float, 'bool' : c_bool, None : None}

    fn = tree.body[0]

    assert isinstance(fn.returns, ast.Name) or isinstance(fn.returns, ast.Constant)
    sig = [ctype_map[fn.returns.id if isinstance(fn.returns, ast.Name) else fn.returns.value]]
    
    for arg in fn.args.args:
        arg_type = arg.annotation

        if isinstance(arg_type, ast.Name):
            sig.append(ctype_map[arg_type.id])
        elif isinstance(arg_type, ast.Subscript):
            arg_eval = eval(compile(ast.Expression(arg_type), '', 'eval'), env)
            assert isinstance(arg_eval, NDArray)

            sig.append(POINTER(ctype_map[arg_eval.kind.__name__]))
        else:
            raise Exception(f"Unknown annotation in type sig: {arg_type}")

    return sig

class LLVMEngine:

    def __init__(self):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        self.engine = self.get_engine()

    def get_engine(self):
        return llvm.create_mcjit_compiler(
            llvm.parse_assembly(""),
            llvm.Target.from_default_triple().create_target_machine()
        )

    def optimize(self, module_source):
        # materialize a LLVM module
        mod = llvm.parse_assembly(str(module_source))
        
        # create optimizer
        pm = llvm.create_module_pass_manager()
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 3  # -O3
        pmb.populate(pm)
        
        # optimize
        pm.run(mod)
    
        return mod

    def compile_module(self, mod):
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

    def compile(self, env = {}, debug = True, external = set(), add_time = []):

        def _compile(fn):
            start = time.perf_counter_ns()
            code = inspect.getsource(fn)
            tree = ast.parse(code)

            mod = GenerateIR(env = env, external = external).visit(tree)

            if debug:
                print("# UNOPTIMIZED CODE")
                print(mod)
        
            mod = self.optimize(mod)
            self.compile_module(mod)
        
            sig = get_ctypes_sig(tree, env = env)
        
            if debug:
                print("# OPTIMIZED CODE")
                print(mod)

            cfunc = CFUNCTYPE(*sig)(self.engine.get_function_address(fn.__name__))
            
            def inner(*args):
                marshalled_Args = []
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        np_map = { 'int32' : c_int, 'float32' : c_float, 'bool' : c_bool}
                        assert arg.dtype.name in np_map
                        arg = arg.ctypes.data_as(POINTER(np_map[arg.dtype.name]))
                    marshalled_Args.append(arg)
                return cfunc(*marshalled_Args)

            end = time.perf_counter_ns()

            add_time.append(end - start)

            return inner

        return _compile
