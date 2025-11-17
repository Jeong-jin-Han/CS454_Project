# sbst_core.py
import ast
import random
import itertools
import types
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
        # store last recorded vars for that lineno and maintain trace order
        self.records[lineno] = vars
        if lineno not in self.trace:
            self.trace.append(lineno)

    def get_records(self, lineno):
        return self.records.get(lineno)

    def get_trace(self):
        return list(self.trace)

    def clear(self):
        self.records = {}
        self.trace = []

class Instrumenter(ast.NodeTransformer):
    """
    Instruments If/While/For and transforms Match into nested Ifs (like original).
    For each branch it creates: _record.write(lineno, {varname: varvalue, ...})
    """
    def visit_If(self, node):
        self.generic_visit(node)
        vars = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        call_record = self._make_record_call(node.lineno, vars)
        return [ast.Expr(value=call_record), node]

    def visit_While(self, node):
        self.generic_visit(node)
        vars = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        call_record = self._make_record_call(node.lineno, vars)
        return [ast.Expr(value=call_record), node]

    def visit_For(self, node):
        self.generic_visit(node)
        # collect names from the iterator expression
        vars = {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)}
        call_record = self._make_record_call(node.lineno, vars)
        return [ast.Expr(value=call_record), node]

    def _make_record_call(self, lineno, vars_set):
        # build: _record.write(lineno, {'a': a, 'b': b})
        record_attr = ast.Attribute(value=ast.Name(id='_record', ctx=ast.Load()), attr='write', ctx=ast.Load())
        branch_lineno = ast.Constant(value=lineno)
        keys = [ast.Constant(value=v) for v in vars_set]
        vals = [ast.Name(id=v, ctx=ast.Load()) for v in vars_set]
        vars_dict = ast.Dict(keys=keys, values=vals)
        return ast.Call(func=record_attr, args=[branch_lineno, vars_dict], keywords=[])

    def visit_Match(self, node):
        # Transform match into nested Ifs for instrumentation + keep semantics for common cases
        subject_node = node.subject
        final_else = [ast.Pass()]

        # build reversed chain to preserve order
        for case in reversed(node.cases):
            pattern = case.pattern
            guard = case.guard
            body = case.body
            condition = None
            assignments = []

            if isinstance(pattern, ast.MatchValue):
                condition = ast.Compare(left=subject_node, ops=[ast.Eq()], comparators=[pattern.value])
            elif isinstance(pattern, ast.MatchOr):
                element_list = [p.value for p in pattern.patterns if isinstance(p, ast.MatchValue)]
                condition = ast.Compare(left=subject_node, ops=[ast.In()], comparators=[ast.List(elts=element_list, ctx=ast.Load())])
            elif isinstance(pattern, ast.MatchAs):
                condition = ast.Constant(value=True)
                if pattern.name is not None:
                    assignments.append(ast.Assign(targets=[ast.Name(id=pattern.name, ctx=ast.Store())], value=subject_node))

            if condition is None:
                continue
            if guard:
                condition = ast.BoolOp(op=ast.And(), values=[condition, guard])

            new_body = assignments + body
            new_node = ast.If(test=condition, body=new_body, orelse=final_else)
            ast.copy_location(new_node, pattern)
            final_else = [new_node]

        if final_else and isinstance(final_else[0], ast.If):
            return self.visit(final_else[0])
        else:
            return ast.Pass()

# ------------------------------
# constant extractor
# ------------------------------
class ConstantExtractor(ast.NodeVisitor):
    def __init__(self):
        self.constants = {}
        self.total_constants = set()

    def visit_Compare(self, node):
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

            if var_name is not None and isinstance(constant_value, int):
                self.constants.setdefault(var_name, set()).add(constant_value)
                self.total_constants.add(constant_value)
        self.generic_visit(node)

    def visit_MatchValue(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            self.total_constants.add(node.value.value)
        self.generic_visit(node)

# ------------------------------
# helpers: approach level, parents, normalization
# ------------------------------
def normalize_distance(distance):
    return 1 - (1.001 ** (-distance))

def get_parents_list(branch_lineno, parent_map):
    path = [branch_lineno]
    current = branch_lineno
    while current in parent_map:
        parent = parent_map[current]
        path.append(parent)
        current = parent
    return path

def get_approach_level(target_lineno, last_executed_lineno, parent_map):
    target_parents = get_parents_list(target_lineno, parent_map)
    last_parents = get_parents_list(last_executed_lineno, parent_map)

    common_parent = None
    for p in target_parents:
        if p in last_parents:
            common_parent = p
            break
    if common_parent is None:
        return len(target_parents) + len(last_parents)
    else:
        # distance from target to common + last to common
        return target_parents.index(common_parent) + last_parents.index(common_parent)

def target_in_node(target_lineno, nodes):
    for node in nodes:
        for subnode in ast.walk(node):
            if hasattr(subnode, 'lineno') and subnode.lineno == target_lineno:
                return node
    return None

# ------------------------------
# branch/expr distance calculations
# ------------------------------
def calculate_expr_distance(expr, vars_map, target_outcome, namespace):
    # many expressions supported previously (BoolOp, Compare, IfExp)
    # fallback: infinite distance if can't reason
    if isinstance(expr, ast.BoolOp):
        # handle AND/OR
        if isinstance(expr.op, ast.And):
            if target_outcome:
                total = 0
                for v in expr.values:
                    total += calculate_expr_distance(v, vars_map, target_outcome, namespace)
                return total
            else:
                distances = [calculate_expr_distance(v, vars_map, target_outcome, namespace) for v in expr.values]
                return min(distances)
        elif isinstance(expr.op, ast.Or):
            if target_outcome:
                distances = [calculate_expr_distance(v, vars_map, target_outcome, namespace) for v in expr.values]
                return min(distances)
            else:
                total = 0
                for v in expr.values:
                    total += calculate_expr_distance(v, vars_map, target_outcome, namespace)
                return total

    elif isinstance(expr, ast.Compare):
        left = expr.left
        left_val = eval(compile(ast.Expression(body=left), filename="<string>", mode="eval"), namespace, vars_map)
        right = expr.comparators[0]

        # 'in' handling
        if isinstance(expr.ops[0], ast.In):
            right_collection = eval(compile(ast.Expression(body=right), filename="<string>", mode="eval"), namespace, vars_map)
            if not isinstance(right_collection, (list, tuple, set)):
                return float('inf')
            if target_outcome:
                if not right_collection:
                    return 1
                return min(abs(left_val - item) for item in right_collection)
            else:
                return 1

        right_val = eval(compile(ast.Expression(body=right), filename="<string>", mode="eval"), namespace, vars_map)

        op = expr.ops[0]
        if isinstance(op, ast.Eq):
            return abs(left_val - right_val) if target_outcome else 1
        elif isinstance(op, ast.NotEq):
            return 1 if target_outcome else abs(left_val - right_val)
        elif isinstance(op, ast.Lt):
            return left_val - right_val + 1 if target_outcome else right_val - left_val
        elif isinstance(op, ast.LtE):
            return left_val - right_val if target_outcome else right_val - left_val + 1
        elif isinstance(op, ast.Gt):
            return right_val - left_val + 1 if target_outcome else left_val - right_val
        elif isinstance(op, ast.GtE):
            return right_val - left_val if target_outcome else left_val - right_val + 1

    elif isinstance(expr, ast.IfExp):
        test_result = eval(compile(ast.Expression(body=expr.test), filename="<string>", mode="eval"), namespace, vars_map)
        if test_result:
            return calculate_expr_distance(expr.body, vars_map, target_outcome, namespace)
        else:
            return calculate_expr_distance(expr.orelse, vars_map, target_outcome, namespace)

    return float('inf')

def calculate_branch_distance(branch_node, recorded_vars, target_outcome, namespace, subject_node=None):
    # branch_node may be ast.If/ast.While/ast.For or ast.match_case node (ast.match_case)
    if isinstance(branch_node, ast.If) or isinstance(branch_node, ast.While):
        test_result = eval(compile(ast.Expression(body=branch_node.test), "<string>", "eval"), namespace, recorded_vars)

        if test_result is True and target_outcome:
            return 0
        if test_result is False and not target_outcome:
            return 0

        # fallback: compute distance on expression
        return calculate_expr_distance(branch_node.test, recorded_vars, target_outcome, namespace)

    elif isinstance(branch_node, ast.For):
        try:
            iter_val = eval(compile(ast.Expression(body=branch_node.iter), "<string>", "eval"), namespace, recorded_vars)
        except Exception:
            return float('inf')
        if target_outcome and len(iter_val) == 0:
            return 1
        elif (not target_outcome) and len(iter_val) > 0:
            return len(iter_val)
        else:
            return 0

    # match/case handling — original used ast.match_case type
    if type(branch_node).__name__ == "match_case" or getattr(branch_node, "_fields", None) and hasattr(branch_node, "pattern"):
        # pattern distance: try to evaluate subject and pattern values
        if subject_node is None:
            return float('inf')
        try:
            subject_val = eval(compile(ast.Expression(body=subject_node), "<string>", "eval"), namespace, recorded_vars)
        except Exception:
            return float('inf')

        pattern = branch_node.pattern
        guard = branch_node.guard
        pattern_distance = float('inf')

        if isinstance(pattern, ast.MatchValue):
            try:
                match_val = eval(compile(ast.Expression(body=pattern.value), "<string>", "eval"), namespace, recorded_vars)
                pattern_distance = abs(subject_val - match_val)
            except Exception:
                pattern_distance = float('inf')
        elif isinstance(pattern, ast.MatchAs):
            pattern_distance = 0
        else:
            pattern_distance = float('inf')

        guard_distance = 0
        if guard:
            guard_distance = calculate_expr_distance(guard, recorded_vars, True, namespace)

        total = pattern_distance + guard_distance
        return total if target_outcome else (1 if total == 0 else 0)

    return float('inf')

# ------------------------------
# Fitness calculator: main interface
# ------------------------------
class FitnessCalculator:
    def __init__(self, traveler: Traveler, record: Record, namespace: dict):
        self.traveler = traveler
        self._record = record
        self.namespace = namespace  # namespace where functions are loaded
        # keep a count if desired
        self.evals = 0

    def calculate_fitness(self, target_branch_node, trace, log, parent_map, target_outcome, subject_node):
        """
        returns approach_level + normalized(branch_distance)
        returns float('inf') if not computable
        """
        self.evals += 1

        # determine target lineno
        if isinstance(target_branch_node, ast.match_case):
            target_lineno = target_branch_node.pattern.lineno
        else:
            target_lineno = target_branch_node.lineno

        if target_lineno in trace:
            approach_level = 0
        else:
            if not trace:
                return float('inf')
            last_executed = trace[-1]
            approach_level = get_approach_level(target_lineno, last_executed, parent_map)

        branch_distance = float('inf')
        if approach_level == 0:
            # we reached the branch statement at runtime; use recorded vars to compute branch distance
            if target_lineno in log:
                branch_distance = calculate_branch_distance(target_branch_node, log[target_lineno], target_outcome, self.namespace, subject_node)
        else:
            # compute based on blocking branch (last executed)
            if not trace:
                return float('inf')
            blocking_lineno = trace[-1]
            if blocking_lineno in log and blocking_lineno in self.traveler.branches.get(self._find_function_for_lineno(blocking_lineno), {}):
                blocking_branch_node = self.traveler.branches[self._find_function_for_lineno(blocking_lineno)][blocking_lineno].node
                blocking_branch_outcome = None
                # check where target lives relative to blocking branch
                if isinstance(blocking_branch_node, ast.If) or isinstance(blocking_branch_node, ast.While):
                    if target_in_node(target_lineno, blocking_branch_node.body):
                        blocking_branch_outcome = True
                    elif target_in_node(target_lineno, blocking_branch_node.orelse):
                        blocking_branch_outcome = False
                if blocking_branch_outcome is not None:
                    branch_distance = calculate_branch_distance(blocking_branch_node, log[blocking_lineno], blocking_branch_outcome, self.namespace)
                else:
                    branch_distance = 1

        if branch_distance == float('inf'):
            return float('inf')
        return approach_level + normalize_distance(branch_distance)

    def _find_function_for_lineno(self, lineno):
        # find function name that contains the branch lineno
        for func in self.traveler.functions:
            # search AST subtree for lineno
            for n in ast.walk(func.node):
                if hasattr(n, "lineno") and n.lineno == lineno:
                    return func.name
        # fallback: None
        return None

    def fitness_for_candidate(self, func, candidate_args, target_branch_node, target_outcome, subject_node, parent_map):
        """
        Run func(*candidate_args) with fresh record and compute fitness.
        Returns float (fitness) or float('inf').
        """
        # clear record then run
        self._record.clear()
        try:
            func(*candidate_args)
        except Exception:
            # if runtime error occurs we still may have some recorded trace; leave it
            pass

        trace = self._record.get_trace()
        log = self._record.records
        return self.calculate_fitness(target_branch_node, trace, log, parent_map, target_outcome, subject_node)

# ------------------------------
# Instrument + load helper
# ------------------------------
def instrument_and_load(source_code: str):
    """
    Instrument source_code (string). Returns:
      (namespace, traveler, record, instrumented_tree)
    Where namespace contains the instrumented functions and _record set to record instance.
    """
    tree = ast.parse(source_code)
    instrumenter = Instrumenter()
    instrumented_tree = instrumenter.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(instrumented_tree)

    traveler = Traveler()
    traveler.visit(tree)

    record = Record()
    namespace = {"_record": record}
    compiled = compile(instrumented_tree, filename="<instrumented>", mode="exec")
    exec(compiled, namespace)

    for func in traveler.functions:
        func_name = func.name
        func_node = func.node

        print(f" '{func_name}' test ")

        extractor = ConstantExtractor()
        extractor.visit(func_node)


    return namespace, traveler, record, instrumented_tree

# ------------------------------
# hill_climbing wrapper that uses FitnessCalculator
# ------------------------------
def hill_climbing_search(func, initial_args, target_branch_node, target_outcome, fitness_calc: FitnessCalculator,
                         actual_indices, parent_map, subject_node=None, match_lineno=None, max_iters=1000):
    """
    Try to find args that make target_branch_node evaluate to target_outcome.
    Uses the hill-climbing logic adapted from original script.
    Returns candidate args or None.
    """
    current_args = list(initial_args)
    # run once to get fitness
    fitness = fitness_calc.fitness_for_candidate(func, current_args, target_branch_node, target_outcome, subject_node, parent_map)
    if fitness == 0:
        return current_args

    update_direction = {}
    direction_correct = {}
    update_rate = {}

    for _ in range(max_iters):
        new_args = current_args[:]
        num_to_modify = 1
        if random.random() < 0.2:
            max_modify = len(actual_indices)
            if max_modify > 1:
                num_to_modify = random.randint(2, max_modify)
        index_to_modify = random.sample(actual_indices, num_to_modify)

        for index in index_to_modify:
            update_direction[index] = update_direction.get(index, 0)
            direction = 0
            direction_correct[index] = direction_correct.get(index, False)
            update_rate[index] = update_rate.get(index, 10)
            if update_direction[index] != 0:
                direction = update_direction[index]
            else:
                direction = random.choice([-1, 1])

            update_size = random.randint(1, max(1, int(round(update_rate[index]))))
            if fitness != float('inf'):
                update_by = max(1, int(round(fitness / 10)))
                max_update_size = min(update_size, update_by)
                update_size = random.randint(1, max(1, max_update_size))
            new_args[index] += direction * update_size

        fitness_calc._record.clear()
        try:
            func(*new_args)
        except Exception:
            pass
        new_fitness = fitness_calc.calculate_fitness(target_branch_node, fitness_calc._record.get_trace(),
                                                     fitness_calc._record.records, parent_map, target_outcome, subject_node)

        if new_fitness == 0:
            return new_args

        if new_fitness < fitness:
            current_args = new_args
            fitness = new_fitness
            for index in index_to_modify:
                update_direction[index] = direction
                if direction_correct[index]:
                    update_rate[index] *= 1.2
                direction_correct[index] = True
        else:
            for index in index_to_modify:
                update_direction[index] = 0
                update_rate[index] = 10
                direction_correct[index] = False

    return None
