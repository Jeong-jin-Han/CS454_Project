import ast
import copy

# ------------------------------
# Data structure
# ------------------------------
class FunctionInfo:
    def __init__(self, name: str, args: list[str], node: ast.FunctionDef):
        self.name = name
        self.args = args
        self.args_dim = len(args)
        self.node = node

        # Extract constants
        extractor = ConstantExtractor()
        extractor.visit(node)
        self.consts = extractor.total_constants
        self.min_const = min(extractor.total_constants)
        self.max_const = max(extractor.total_constants)

        self.branches = []

    def __repr__(self):
        return f"Function {self.name} with {self.args_dim} arg(s): {self.args}"

class BranchInfo:
    def __init__(self, node: ast.AST, subject: ast.expr = None, match_lineno: int = None):
        self.node = node
        self.subject = subject
        self.match_lineno = match_lineno

    def __repr__(self):
        node_type = type(self.node).__name__
        lineno = getattr(self.node, 'lineno', 'N/A')
        return f"Branch {node_type} at lineno={self.match_lineno if self.match_lineno is not None else lineno}"

# ------------------------------
# Preprocess
# ------------------------------
class MatchTransformer(ast.NodeTransformer):
    '''
    A class that Transforms match-case statements into if-elif-else chains for simplicity.
    '''
    def _build_test_expression(self, match_node: ast.Match, match_case: ast.match_case) -> ast.expr:
        '''
        Builds a test (condition) expression for a match-case statement.
        '''
        def build_pattern_expr(subject: ast.expr, pattern: ast.pattern) -> ast.expr:
            match pattern:
                case ast.MatchValue(value): 
                    # case 1: -> if subject == 1
                    return ast.Compare(left=subject, ops=[ast.Eq()], comparators=[value])
                
                case ast.MatchSingleton(value):
                    # case True: -> if subject is True
                    return ast.Compare(left=subject, ops=[ast.Is()], comparators=[ast.Constant(value)])

                case ast.MatchSequence(patterns):
                    # case [a, b]: -> isinstance(subject, (list, tuple)) and (each pattern's expression)
                    is_sequence = ast.Call(
                        func=ast.Name(id='isinstance', ctx=ast.Load()),
                        args=[subject, ast.Tuple(elts=[ast.Name(id='list', ctx=ast.Load()), ast.Name(id='tuple', ctx=ast.Load())], ctx=ast.Load())],
                        keywords=[]
                    )

                    exprs = [is_sequence]
                    for i, subpattern in enumerate(patterns):
                        if isinstance(subpattern, ast.MatchStar): break # Star pattern always matches no matter what is left

                        subject_index = ast.Subscript(
                            value=subject,
                            slice=ast.Constant(value=i),
                            ctx=ast.Load()
                        )
                        exprs.append(build_pattern_expr(subject_index, subpattern))

                    return ast.BoolOp(op=ast.And(), values=exprs)

                case ast.MatchMapping(keys, patterns, _):
                    # case {'k': v}: -> isinstance(subject, dict) and (each key-value set's expression)
                    is_dict = ast.Call(
                        func=ast.Name(id='isinstance', ctx=ast.Load()),
                        args=[subject, ast.Name(id='dict', ctx=ast.Load())],
                        keywords=[]
                    )
                    
                    exprs = [is_dict]
                    for key, subpattern in zip(keys, patterns):
                        key_exist_check = ast.Compare(
                            left=key,
                            ops=[ast.In()],
                            comparators=[subject]
                        )
                        
                        subject_index = ast.Subscript(
                            value=subject,
                            slice=key,
                            ctx=ast.Load()
                        )
                        pattern_match_check = build_pattern_expr(subject_index, subpattern)

                        exprs.extend([key_exist_check, pattern_match_check])

                    return ast.BoolOp(op=ast.And(), values=exprs)

                case ast.MatchAs(pattern):
                    # case x: or case _ as x: or case <pattern> as x:
                    if pattern is None:
                        # Wildcard or name binding always matches.
                        return ast.Constant(value=True)
                    else:
                        return build_pattern_expr(subject, pattern)

                case ast.MatchOr(patterns):
                    # case 1 | 2: -> (subject == 1) or (subject == 2)
                    if not patterns: return ast.Constant(value=False)

                    or_exprs = [build_pattern_expr(subject, p) for p in patterns]
                    return ast.BoolOp(op=ast.Or(), values=or_exprs)
                
                case ast.MatchStar():
                    # MatchStar cannot appear at the top level of a case.
                    return ast.Constant(value=False)
                
                case _:
                    return ast.Constant(value=True)
        
        pattern_expr = build_pattern_expr(match_node.subject, match_case.pattern)
        if match_case.guard:
            return ast.BoolOp(
                op=ast.And(),
                values=[pattern_expr, match_case.guard]
            )
        else:
            return pattern_expr

    def visit_Match(self, node: ast.Match) -> ast.If:
        self.generic_visit(node)  # First, visit all child nodes

        if not node.cases:
            return None  # No cases to transform

        # Start building the if-elif-else chain
        first_case = node.cases[0]
        test_expr = self._build_test_expression(node, first_case)
        if_node = ast.If(
            test=test_expr,
            body=first_case.body,
            orelse=[]
        )

        current_if = if_node
        for case in node.cases[1:]:
            if isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None:
                # Handle wildcard matches at the end
                current_if.orelse = case.body
                break

            test_expr = self._build_test_expression(node, case)
            new_if = ast.If(
                test=test_expr,
                body=case.body,
                orelse=[]
            )

            current_if.orelse = [new_if]
            current_if = new_if

        return if_node

class IfExpTransformer(ast.NodeTransformer):
    """
    Hoist any ast.IfExp found anywhere into a preceding if/else that assigns
    into a generated temporary variable, then replace the original IfExp with
    that temporary variable.
    """
    def __init__(self):
        super().__init__()
        self.counter = 0

    def _new_temp(self):
        self.counter += 1
        return f"__ifexp_repl_var_{self.counter}"

    # Entry points for blocks that contain statements
    def visit_Module(self, node):
        node.body = self._process_stmt_list(node.body)
        return node

    def visit_FunctionDef(self, node):
        node.body = self._process_stmt_list(node.body)
        return node

    def visit_AsyncFunctionDef(self, node):
        node.body = self._process_stmt_list(node.body)
        return node

    def visit_ClassDef(self, node):
        node.body = self._process_stmt_list(node.body)
        return node

    # Process a list of statements: hoist inside each stmt and inject hoisted stmts
    def _process_stmt_list(self, stmts):
        new_body = []
        for stmt in stmts:
            hoisted, new_stmt = self._process_stmt(stmt)
            new_body.extend(hoisted)
            new_body.append(new_stmt)
        return new_body

    # Process a single statement: transform expressions within and collect hoisted stmts
    def _process_stmt(self, stmt):
        hoisted = []
        for field, value in list(ast.iter_fields(stmt)):
            # single expression field
            if isinstance(value, ast.expr):
                h, new_expr = self._transform_expr(value)
                hoisted.extend(h)
                setattr(stmt, field, new_expr)

            # lists can be lists of stmts (e.g. body) or lists of exprs (e.g. targets, keywords)
            elif isinstance(value, list):
                if len(value) > 0 and all(isinstance(item, ast.stmt) for item in value):
                    # statement-list: fully process it (recursively)
                    processed = self._process_stmt_list(value)
                    setattr(stmt, field, processed)
                else:
                    # expression-list (e.g., comparators, args)
                    new_list = []
                    for item in value:
                        if isinstance(item, ast.expr):
                            h, new_item = self._transform_expr(item)
                            hoisted.extend(h)
                            new_list.append(new_item)
                        else:
                            new_list.append(item)
                    setattr(stmt, field, new_list)
        return hoisted, stmt

    # Transform an expression and return (hoisted_statements, new_expression)
    def _transform_expr(self, expr):
        if expr is None:
            return [], expr

        # direct IfExp: hoist it
        if isinstance(expr, ast.IfExp):
            return self._hoist_ifexp(expr)

        # special-case Compare to avoid in-place aliasing issues and maintain order
        if isinstance(expr, ast.Compare):
            hoisted = []
            left_h, left_new = self._transform_expr(expr.left)
            hoisted.extend(left_h)

            new_comparators = []
            for comp in expr.comparators:
                comp_h, comp_new = self._transform_expr(comp)
                hoisted.extend(comp_h)
                new_comparators.append(comp_new)

            new_compare = ast.Compare(
                left=left_new,
                ops=copy.deepcopy(expr.ops),
                comparators=new_comparators
            )
            return hoisted, new_compare

        # generic recursion for other expression types
        hoisted = []
        for field, value in list(ast.iter_fields(expr)):
            if isinstance(value, ast.expr):
                h, new_val = self._transform_expr(value)
                hoisted.extend(h)
                setattr(expr, field, new_val)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, ast.expr):
                        h, new_item = self._transform_expr(item)
                        hoisted.extend(h)
                        new_list.append(new_item)
                    else:
                        new_list.append(item)
                setattr(expr, field, new_list)
        return hoisted, expr

    # Actual hoisting of an IfExp -> produces a list of statements and a tmp name
    def _hoist_ifexp(self, ifexp: ast.IfExp):
        # recursively transform inner pieces first (this returns hoisted code + new expr)
        test_h, test = self._transform_expr(ifexp.test)
        body_h, body = self._transform_expr(ifexp.body)
        orelse_h, orelse = self._transform_expr(ifexp.orelse)

        tmp_name = self._new_temp()
        tmp_store = ast.Name(id=tmp_name, ctx=ast.Store())
        tmp_load = ast.Name(id=tmp_name, ctx=ast.Load())

        # deepcopy the expressions used as assigned values to avoid aliasing bugs
        assign_then = ast.Assign(targets=[tmp_store], value=copy.deepcopy(body))
        assign_else = ast.Assign(targets=[tmp_store], value=copy.deepcopy(orelse))

        # we put any hoisted code from body/orelse inside the respective branch,
        # and hoisted code from test before the If.
        new_if = ast.If(
            test=copy.deepcopy(test),
            body=[*body_h, assign_then],
            orelse=[*orelse_h, assign_else]
        )

        total_hoisted = [*test_h, new_if]
        return total_hoisted, tmp_load

# ------------------------------
# AST walkers / instrumenter
# ------------------------------
class Traveler(ast.NodeVisitor):
    def __init__(self):
        self.branches: dict[str, dict[int, BranchInfo]] = {}
        self.functions: list[FunctionInfo] = []
        self.current_function = None
        self.current_function_branches = None
        self.parent_map = {}
        self.parent_stack = []

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.current_function_branches = {}
        self.generic_visit(node)

        args = [arg.arg for arg in node.args.args]
        self.functions.append(FunctionInfo(name=node.name, args=args, node=node))
        self.functions[-1].branches.extend(list(self.current_function_branches.values()))
        self.branches[self.current_function] = self.current_function_branches
        self.current_function = None
        self.current_function_branches = None

    def _register_branch(self, node, subject=None, match_lineno=None, case_pattern_lineno=None):
        if self.current_function_branches is not None:
            lineno = getattr(node, "lineno", None)
            # for match cases a case pattern lineno may be provided
            key = case_pattern_lineno if case_pattern_lineno is not None else lineno
            self.current_function_branches[key] = BranchInfo(node, subject, match_lineno)
            if self.parent_stack:
                self.parent_map[key] = self.parent_stack[-1]

    def visit_If(self, node):
        self._register_branch(node)
        self.parent_stack.append(node.lineno)
        self.generic_visit(node)
        self.parent_stack.pop()

    def visit_While(self, node):
        self._register_branch(node)
        self.parent_stack.append(node.lineno)
        self.generic_visit(node)
        self.parent_stack.pop()

    def visit_For(self, node):
        self._register_branch(node)
        self.parent_stack.append(node.lineno)
        self.generic_visit(node)
        self.parent_stack.pop()

    def visit_Match(self, node):
        subject_node = node.subject
        match_lineno = node.lineno
        for case in node.cases:
            # case.pattern may have lineno on the pattern node
            pattern_lineno = getattr(case.pattern, "lineno", None)
            key_lineno = pattern_lineno if pattern_lineno is not None else case.lineno if hasattr(case, "lineno") else match_lineno
            self.current_function_branches[key_lineno] = BranchInfo(case, subject_node, match_lineno)
            # visit body of case
            for item in case.body:
                self.visit(item)

class Record:
    def __init__(self):
        self.records = {}
        self.trace = []
    
    def write(self, lineno, vars):
        self.records[lineno] = vars
        if lineno not in self.trace:
            self.trace.append(lineno)

    def get_records(self, lineno):
        return self.records.get(lineno)
    
    def get_trace(self):
        return self.trace
    
    def clear(self):
        self.records = {}
        self.trace = []

class Instrumenter(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)

        vars = set()
        for n in ast.walk(node.test):
            if isinstance(n, ast.Name):
                vars.add(n.id)
        
        call_record_func = ast.Attribute(value=ast.Name(id='_record', ctx=ast.Load()), attr='write', ctx=ast.Load())
        branch_lineno = ast.Constant(value=node.lineno)
        vars_dict = ast.Dict(keys=[ast.Constant(value=v) for v in vars], values=[ast.Name(id=v, ctx=ast.Load()) for v in vars])
        call_record = ast.Call(func=call_record_func, args=[branch_lineno, vars_dict], keywords=[])

        new_node = [ast.Expr(value=call_record), node]
        return new_node
    
    def visit_While(self, node):
        self.generic_visit(node)

        vars = set()
        for n in ast.walk(node.test):
            if isinstance(n, ast.Name):
                vars.add(n.id)
        
        call_record_func = ast.Attribute(value=ast.Name(id='_record', ctx=ast.Load()), attr='write', ctx=ast.Load())
        branch_lineno = ast.Constant(value=node.lineno)
        vars_dict = ast.Dict(keys=[ast.Constant(value=v) for v in vars], values=[ast.Name(id=v, ctx=ast.Load()) for v in vars])
        call_record = ast.Call(func=call_record_func, args=[branch_lineno, vars_dict], keywords=[])

        new_node = [ast.Expr(value=call_record), node]
        return new_node
    
    def visit_For(self, node):
        self.generic_visit(node)

        vars = set()
        for n in ast.walk(node.iter):
            if isinstance(n, ast.Name):
                vars.add(n.id)
        
        call_record_func = ast.Attribute(value=ast.Name(id='_record', ctx=ast.Load()), attr='write', ctx=ast.Load())
        branch_lineno = ast.Constant(value=node.lineno)
        vars_dict = ast.Dict(keys=[ast.Constant(value=v) for v in vars], values=[ast.Name(id=v, ctx=ast.Load()) for v in vars])
        call_record = ast.Call(func=call_record_func, args=[branch_lineno, vars_dict], keywords=[])

        new_node = [ast.Expr(value=call_record), node]
        return new_node
    
    def visit_Match(self, node):
        raise AssertionError("Match should converted in the previous step")
    
class ConstantExtractor(ast.NodeVisitor):
    def __init__(self):
        self.constants = {}
        self.total_constants = set()
    def visit_Compare(self, node):
        if isinstance(node, ast.Compare):
            operands = [node.left] + node.comparators
            for i, op in enumerate(node.ops):
                left_operand = operands[i]
                right_operand = operands[i + 1]

                if isinstance(op, (ast.In, ast.NotIn)):
                    if isinstance(left_operand, ast.Name):
                        var_name = left_operand.id
                        for sub in ast.walk(right_operand):
                            if isinstance(sub, ast.Constant) and isinstance(sub.value, int):
                                self.constants.setdefault(var_name, set()).add(sub.value)
                                self.total_constants.add(sub.value)
                    continue

                var_name = None
                constant_value = None

                if isinstance(left_operand, ast.Name) and isinstance(right_operand, ast.Constant):
                    var_name = left_operand.id
                    constant_value = right_operand.value
                elif isinstance(left_operand, ast.Constant) and isinstance(right_operand, ast.Name):
                    var_name = right_operand.id
                    constant_value = left_operand.value
                
                if var_name is not None and isinstance(constant_value, (int)):
                    self.constants.setdefault(var_name, set()).add(constant_value)
                    self.total_constants.add(constant_value)
        self.generic_visit(node)

    def visit_MatchValue(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int)):
            self.total_constants.add(node.value.value)
        self.generic_visit(node)

# ------------------------------
# Main Fitness Calculator
# ------------------------------
class FitnessManager():
    def __init__(self, code: str):
        # Count
        self.evals = 0

        # Record
        self._record = Record()

        # Preprocess
        raw_tree = ast.parse(code)
        preprocessed_tree = IfExpTransformer().visit(MatchTransformer().visit(raw_tree))
        ast.fix_missing_locations(preprocessed_tree)
        tree = ast.parse(ast.unparse(preprocessed_tree))
        
        # Instrument / Travel
        self._instrumenter = Instrumenter()
        instrumented_tree = self._instrumenter.visit(tree)
        ast.fix_missing_locations(instrumented_tree)

        # FIXME: for debug
        # print(ast.unparse(instrumented_tree)) 
        

        self._traveler = Traveler()
        self._traveler.visit(tree)

        self._instrumented_obj = compile(instrumented_tree, filename="<instrumented>", mode="exec")
        self._namespace = {"_record": self._record}
        exec(self._instrumented_obj, self._namespace)

        # function_testcases = {}
        # for func in self._traveler.functions:
        #     func_name = func["name"]
        #     func_args_len = len(func["args"])
        #     target_func = self._namespace[func_name]
        #     branches = self._traveler.branches.get(func_name, {})
        #     func_node = func["node"]

        #     print(f" '{func_name}' test ")

        #     parents = []

        #     parents_from_constants = []
        #     base_parent = parents[0] if parents else [random.randint(min_const, max_const) for _ in range(func_args_len)]

        #     for var, consts in constants.items():
        #         if var in func["args"]:
        #             arg_index = func["args"].index(var)
        #             for c in consts:
        #                 for offset in [-1, 0, 1]:
        #                     new_arg_val = c + offset
        #                     new_parent = base_parent[:]
        #                     new_parent[arg_index] = new_arg_val
        #                     parents_from_constants.append(new_parent)

        #     consts_per_arg = {}
        #     for var, consts in constants.items():
        #         if var in func["args"]:
        #             arg_index = func["args"].index(var)
        #             consts_per_arg[arg_index] = list(consts)
        #     consts_lists = []

        #     for i in range(func_args_len):
        #         if i in consts_per_arg:
        #             consts_lists.append(consts_per_arg[i])
        #         else:
        #             consts_lists.append([random.randint(min_const, max_const)])

        #     strong_parents =  list(itertools.product(*consts_lists))
        #     if len(strong_parents) > 100:
        #         strong_parents = random.sample(strong_parents, 100)
        #     parents_from_constants.extend(strong_parents)

            
        #     print("  parents from constants:", parents_from_constants)
            
        #     target_branches = []
        #     for lineno, branch_info in branches.items():
        #         target_branches.append((lineno, branch_info, True))
        #         target_branches.append((lineno, branch_info, False))
            
        #     test_cases = []
        #     for lineno, branch_info, target_outcome in target_branches:
        #         print(f"  branch at line {lineno} to be {'taken' if target_outcome else 'not taken'}")
        #         branch_node = branch_info['node']
        #         subject_node = branch_info['subject']
        #         match_lineno = branch_info.get('match_lineno')

        #         actual_vars = {n.id for n in ast.walk(branch_node) if isinstance(n, ast.Name)}
        #         actual_indices = [i for i, arg in enumerate(func["args"]) if arg in actual_vars]
        #         if len(actual_indices) == 0:
        #             actual_indices = list(range(func_args_len))

        #         total_parents = parents + parents_from_constants
        #         if len(constants) == 0:
        #             total_parents = parents
        #         total_parents = total_parents + [[random.randint(min_const, max_const) for _ in range(func_args_len)] for _ in range(5)]
        #         found_case = False
        #         for child in total_parents:
        #             result = hill_climbing(target_func, branch_node, target_outcome, namespace, actual_indices, list(child), branches, traveler.parent_map, subject_node, match_lineno)
        #             if result is not None:
        #                 print(f"   found test case: {result}")
        #                 test_cases.append(result)
        #                 parents.append(result)
        #                 found_case = True
        #                 break
        #         if not found_case:
        #             random_child = [random.randint(min_const, max_const) for _ in range(func_args_len)]
        #             result = hill_climbing(target_func, branch_node, target_outcome, namespace, actual_indices, random_child, branches, traveler.parent_map, subject_node, match_lineno)
        #             if result is not None:
        #                 print(f"   found test case: {result}")
        #                 test_cases.append(result)
        #                 parents.append(result)
        #             else:
        #                 print(f"   no test case found")
            
        #     function_testcases[func_name] = test_cases

    def get_functions(self) -> list[FunctionInfo]:
        return self._traveler.functions
    
    
    # -----------------
    # Calculate fitness
    # -----------------
    def _normalize(self, dist):
        return 1 - (1.001 ** (-dist))
    
    def _get_parents_list(self, branch_lineno, parent_map):
        path = [branch_lineno]
        current = branch_lineno
        while current in parent_map:
            parent = parent_map[current]
            path.append(parent)
            current = parent
        return path

    def _get_approach_level(self, target_lineno, last_executed_lineno, parent_map):
        target_parents = self._get_parents_list(target_lineno, parent_map)
        last_executed_parents = self._get_parents_list(last_executed_lineno, parent_map)

        common_parents = None
        for p in target_parents:
            if p in last_executed_parents:
                common_parents = p
                break
        if common_parents is None:
            return len(target_parents) + len(last_executed_parents)
        else:
            return target_parents.index(common_parents) + last_executed_parents.index(common_parents)
        
    def _target_in_node(self, target_lineno, nodes):
        for node in nodes:
            for subnode in ast.walk(node):
                if hasattr(subnode, 'lineno') and subnode.lineno == target_lineno:
                    return node
        return None

    def _calculate_expr_distance(self, expr, vars, target_outcome, namespace):
        if isinstance(expr, ast.BoolOp):
            if isinstance(expr.op, ast.And):
                if target_outcome:
                    total_distance = 0
                    for value in expr.values:
                        distance = self._calculate_expr_distance(value, vars, target_outcome, namespace)
                        total_distance += distance
                    return total_distance
                else:
                    distances = [self._calculate_expr_distance(value, vars, target_outcome, namespace) for value in expr.values]
                    return min(distances)
            elif isinstance(expr.op, ast.Or):
                if target_outcome:
                    distances = [self._calculate_expr_distance(value, vars, target_outcome, namespace) for value in expr.values]
                    return min(distances)
                else:
                    total_distance = 0
                    for value in expr.values:
                        distance = self._calculate_expr_distance(value, vars, target_outcome, namespace)
                        total_distance += distance
                    return total_distance
        elif isinstance(expr, ast.Compare):
            left = expr.left
            left_val = eval(compile(ast.Expression(body=left), filename="<string>", mode="eval"), namespace, vars)
            right = expr.comparators[0]
            if isinstance(expr.ops[0], ast.In):
                right_collection = eval(compile(ast.Expression(body=right), filename="<string>", mode="eval"), namespace, vars)
                
                if not isinstance(right_collection, (list, set, tuple)):
                    return float('inf')
                if target_outcome:
                    if not right_collection:
                        return 1
                    return min(abs(left_val - item) for item in right_collection)
                else:
                    return 1
                
            right_val = eval(compile(ast.Expression(body=right), filename="<string>", mode="eval"), namespace, vars)
            
            if isinstance(expr.ops[0], ast.Eq):
                return abs(left_val - right_val) if target_outcome else 1
            elif isinstance(expr.ops[0], ast.NotEq):
                return 1 if target_outcome else abs(left_val - right_val)
            elif isinstance(expr.ops[0], ast.Lt):
                return left_val - right_val + 1 if target_outcome else right_val - left_val
            elif isinstance(expr.ops[0], ast.LtE):
                return left_val - right_val if target_outcome else right_val - left_val + 1
            elif isinstance(expr.ops[0], ast.Gt):
                return right_val - left_val + 1 if target_outcome else left_val - right_val
            elif isinstance(expr.ops[0], ast.GtE):
                return right_val - left_val if target_outcome else left_val - right_val + 1
        elif isinstance(expr, ast.IfExp):
            test_node = expr.test
            test_result = eval(compile(ast.Expression(body=test_node), filename="<string>", mode="eval"), namespace, vars)
            if test_result:
                return self._calculate_expr_distance(expr.body, vars, target_outcome, namespace)
            else:
                return self._calculate_expr_distance(expr.orelse, vars, target_outcome, namespace)
        
        return float('inf')

    def _calculate_branch_distance(self, branch_node, vars, target_outcome, namespace):
        if isinstance(branch_node, ast.If) or isinstance(branch_node, ast.While):
            test = ast.Expression(body=branch_node.test)
            test_obj = compile(test, filename="<string>", mode="eval")
            test_result = eval(test_obj, namespace, vars)

            if test_result and target_outcome:
                return 0
            elif not test_result and not target_outcome:
                return 0
            
            return self._calculate_expr_distance(branch_node.test, vars, target_outcome, namespace)
                
        elif isinstance(branch_node, ast.For):
            iter_node = branch_node.iter
            iter = eval(compile(ast.Expression(body=iter_node), filename="<string>", mode="eval"), namespace, vars)

            if target_outcome and len(iter) == 0:
                return 1
            elif not target_outcome and len(iter) > 0:
                return len(iter)
            else:
                return 0

        print("Unknown branch node type for distance calculation")
        return float('inf')
    
    def _calculate_fitness(self, target_branch_node, trace, log, parent_map, target_outcome, namespace, branches):
        self.evals += 1

        if isinstance(target_branch_node, ast.match_case):
            target_lineno = target_branch_node.pattern.lineno
        else:
            target_lineno = target_branch_node.lineno

        if target_lineno in trace:
            approach_level = 0
        else:
            if not trace:
                return float('inf')
            last_executed_lineno = trace[-1]

            approach_level = self._get_approach_level(target_lineno, last_executed_lineno, parent_map)

        branch_distance = float('inf')
        if approach_level == 0:
            if target_lineno in log:
                branch_distance = self._calculate_branch_distance(target_branch_node, log[target_lineno], target_outcome, namespace)
        else:
            blocking_branch_lineno = trace[-1]
            if blocking_branch_lineno in log and blocking_branch_lineno in branches:
                blocking_branch_node = branches[blocking_branch_lineno]
                blocking_branch_outcome = None
                if isinstance(blocking_branch_node, ast.If) or isinstance(blocking_branch_node, ast.While):
                    if self._target_in_node(target_lineno, blocking_branch_node.body):
                        blocking_branch_outcome = True
                    elif self._target_in_node(target_lineno, blocking_branch_node.orelse):
                        blocking_branch_outcome = False
                if blocking_branch_outcome is not None:
                    branch_distance = self._calculate_branch_distance(blocking_branch_node, log[blocking_branch_lineno], blocking_branch_outcome, namespace)
                else:
                    branch_distance = 1

        if branch_distance == float('inf'):
            return float('inf')
        return approach_level + self._normalize(branch_distance)

    def calculate_fitness(self, target_function: FunctionInfo, target_branch: BranchInfo, target_outcome: bool, candidate: list[int]) -> float:
        """
        Run func(*candidate_args) with fresh record and compute fitness.
        Returns float (fitness) or float('inf').
        """
        if len(candidate) != target_function.args_dim:
            raise ValueError("Input dimension and the function dimension does not match")

        func = self._namespace[target_function.name]

        # clear record then run
        self._record.clear()
        try:
            func(*candidate)
        except Exception:
            # if runtime error occurs we still may have some recorded trace; leave it
            pass

        trace = self._record.get_trace()
        log = self._record.records

        fitness = self._calculate_fitness(target_branch.node, trace, log, self._traveler.parent_map, target_outcome, self._namespace, self._traveler.branches.get(target_function.name, {}))
        assert fitness >= 0
        return fitness
