from .array import NDArray

from llvmlite import ir
import inspect
import ast

class SpliceRewrite(ast.NodeTransformer):
    def __init__(self, name_map):
        super().__init__()
        self.name_map = name_map
        
    def visit_Name(self, node):
        return self.name_map[node.id] if node.id in self.name_map else node

class GenerateIR(ast.NodeVisitor):

    def __init__(self, env, external):
        super().__init__()
        self.labels = {}
        self.stack = {}
        self.env = env

        self.ext_types = {}
        self.ext = {}

        for func in external:
            node = ast.parse(inspect.getsource(func)).body[0]
            self.ext_types[func.__name__] = ir.FunctionType(
                self.node_to_irtype(node.returns),
                list(map(lambda x: self.node_to_irtype(x.annotation), node.args.args))
            )

    # Types

    def typename_to_irtype(self, name):
        return {
            'int'   : ir.IntType(32),
            'bool'  : ir.IntType(1),
            'float' : ir.FloatType(),
            None    : ir.VoidType()
        }[name]

    def node_to_irtype(self, node):
        if   isinstance(node, ast.Name): return self.typename_to_irtype(node.id)
        elif isinstance(node, ast.Constant): return self.typename_to_irtype(node.value)
        elif isinstance(node, ast.Subscript):
            arg_eval = eval(compile(ast.Expression(node), '', 'eval'), self.env)
            # print(arg_eval, type(arg_eval), type(NDArray))
            assert isinstance(arg_eval, NDArray)

            # Determine type of array type

            assert len(arg_eval.dims) > 0
            rdims = reversed(arg_eval.dims)
            t = ir.ArrayType(self.typename_to_irtype(arg_eval.kind.__name__), next(rdims))
            while True:
                try: t = ir.ArrayType(t, next(rdims))
                except StopIteration: break

            return t.as_pointer()
        
        raise Exception(f"Unsupported type: {node}")

    # Splicing

    def run_under_splice(self, node):
        # This will always return an AST, which can subsequently be spliced in

        # Rewrite args
        # print(ast.dump(ast.parse(inspect.getsource(self.env[node.func.id])), indent=2))
        fn_args = [a.arg for a in ast.parse(inspect.getsource(self.env[node.func.id])).body[0].args.args]
        name_map = { fn_arg : arg for arg, fn_arg in zip(node.args, fn_args)}
        
        # Evaluate
        node = ast.fix_missing_locations(ast.Expression(node))
        # print(ast.dump(node, indent=2), name_map)
        # print(ast.dump(node, indent=2))
        result = eval(compile(node, '', 'eval'), self.env)
        
        # Result of evaluating a splice must be an AST
        assert isinstance(result, ast.AST)
    
        # Rewrite variables
        transformed = SpliceRewrite(name_map).visit(result)
        
        return transformed

    # Visitors

    def visit_all(self, nodes):
        for node in nodes: self.visit(node)

    def generic_visit(self, node):
        raise Exception(f"Unsupported node type: {node}")

    def visit_Expr(self, node):
        self.visit(node.value)
    
    def visit_Module(self, node):
        self.module = ir.Module()
        for k, v in self.ext_types.items():
            self.ext[k] = ir.Function(self.module, v, k)

        self.visit_all(node.body)
        return self.module

    #def visit_Interactive(self, node):
    #    visit_all(node.body)

    def visit_Call(self, node):
        # Eval node, then splice
        if node.func.id == "splice":
            assert len(node.args) == 1, "Splices are univariate"
            evaluated = self.run_under_splice(node.args[0]).body
            if len(evaluated) == 1 and isinstance(evaluated[0], ast.Expr):
                return self.visit(evaluated[0].value)
            else:
                return self.visit_all(evaluated)
        elif node.func.id == "label":
            assert len(node.args) == 1, "Labels are univariate"
            assert isinstance(node.args[0], ast.Constant)
            label = node.args[0].value
            if self.fn_builder.block.is_terminated:
                new = self.fn_builder.append_basic_block()
                self.fn_builder.position_at_start(new)
            new = self.fn_builder.append_basic_block()
            if label not in self.labels:
                self.labels[node.args[0].value] = new
            else:
                # Backlink block
                with self.fn_builder.goto_block(self.labels[label]):
                    self.fn_builder.branch(new)
            self.fn_builder.branch(new)
            self.fn_builder.position_at_start(new)
        elif node.func.id == "goto":
            assert len(node.args) == 1, "Labels are univariate"
            assert isinstance(node.args[0], ast.Constant)
            label = node.args[0].value
            if label not in self.labels:
                new = self.fn_builder.append_basic_block()
                self.labels[label] = new
            self.fn_builder.branch(self.labels[label])
            new = self.fn_builder.append_basic_block()
            self.fn_builder.position_at_start(new)
        elif node.func.id in self.ext:
            return self.fn_builder.call(self.ext[node.func.id], [self.visit(n) for n in node.args])
        else:
            assert False, "Arbitrary function calls not implemented"

    def visit_If(self, node):
        body   = self.fn_builder.append_basic_block()
        orelse = self.fn_builder.append_basic_block()
        after  = self.fn_builder.append_basic_block()

        test = self.visit(node.test)
        
        self.fn_builder.position_at_start(body)
        self.visit_all(node.body)
        self.fn_builder.branch(after)

        self.fn_builder.position_at_start(orelse)
        self.visit_all(node.orelse)
        self.fn_builder.branch(after)

        self.fn_builder.position_after(test)
        self.fn_builder.cbranch(test, body, orelse)

        self.fn_builder.position_at_start(after)

    def visit_While(self, node):
        test   = self.fn_builder.append_basic_block()
        body   = self.fn_builder.append_basic_block()
        after  = self.fn_builder.append_basic_block()

        self.fn_builder.branch(test)
        
        self.fn_builder.position_at_start(body)
        self.visit_all(node.body)
        self.fn_builder.branch(test)

        self.fn_builder.position_at_start(test)
        cond = self.visit(node.test)
        self.fn_builder.cbranch(cond, body, after)
        
        self.fn_builder.position_at_start(after)

    def visit_FunctionDef(self, node):
        self.function = ir.Function(
            self.module,
            ir.FunctionType(
                self.node_to_irtype(node.returns),
                list(map(lambda x: self.node_to_irtype(x.annotation), node.args.args))
            ),
            node.name
        )

        for i in range(len(node.args.args)): self.function.args[i].name = node.args.args[i].arg

        self.fn_builder = ir.IRBuilder(self.function.append_basic_block('entry'))

        self.visit_all(node.body)
        return self.function

    def visit_Return(self, node):

        return self.fn_builder.ret(self.visit(node.value)) \
            if node.value is not None else self.fn_builder.ret_void()

    def visit_BinOp(self, node):
        return self.visit(node.op)(self.visit(node.left), self.visit(node.right))

    def visit_Add(self, node):
        def _add(a, b):
            assert a.type == b.type
            if   isinstance(a.type, ir.IntType):   return self.fn_builder.add(a, b)
            elif isinstance(a.type, ir.FloatType): return self.fn_builder.fadd(a, b)
            else: raise Exception(f"Unknown type: {a.type}") 

        return _add

    def visit_Sub(self, node):
        return lambda a, b : self.fn_builder.sub(a, b)

    def visit_Mult(self, node):
        def _mul(a, b):
            assert a.type == b.type
            if   isinstance(a.type, ir.IntType):   return self.fn_builder.mul(a, b)
            elif isinstance(a.type, ir.FloatType): return self.fn_builder.fmul(a, b)
            else: raise Exception(f"Unknown type: {a.type}") 

        return _mul

    def visit_Compare(self, node):
        assert len(node.ops) == 1, "Only one comparison supported"
        mapping = {ast.Eq  : "==",
                   ast.LtE : "<=",
                   ast.GtE : ">=",
                   ast.Lt  : "<",
                   ast.Gt  : ">"}
        return self.fn_builder.icmp_signed(mapping[type(node.ops[0])], self.visit(node.left), self.visit(node.comparators[0]))

    def visit_Subscript(self, node):
        
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        # print(ast.dump(node, indent=2))
        idx = [ir.Constant(ir.IntType(32), 0)] + [self.visit(n) for n in indices]
        ptr = self.fn_builder.gep(self.visit(node.value), idx)

        if isinstance(node.ctx, ast.Load):
            return self.fn_builder.load(ptr)
        elif isinstance(node.ctx, ast.Store):
            return ptr
        else:
            raise Exception(f"Unknown context: {cfx}")
    
    def visit_Constant(self, node):
        return ir.Constant(self.typename_to_irtype(type(node.value).__name__), node.value)

    def visit_Assign(self, node):
        value = self.visit(node.value)
        assert len(node.targets) == 1
        target = node.targets[0]

        if isinstance(target, ast.Name):
            if target.id not in self.stack:
                self.stack[target.id] = self.fn_builder.alloca(value.type, name=target.id)
            ptr =  self.stack[target.id]
        elif isinstance(target, ast.Subscript):
            ptr = self.visit(target)
        else:
            raise Exception(f"Unknown assignment target {target}")

        self.fn_builder.store(value, ptr)

    def visit_Name(self, node):

        for arg in self.function.args:
            if arg.name == node.id:
                return arg
        
        if node.id in self.stack:
            return self.fn_builder.load(self.stack[node.id])

        assert False, "Could not resolve variable: " + node.id
