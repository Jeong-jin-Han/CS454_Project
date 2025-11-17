import argparse, ast, os, random, traceback, math
from typing import Any, Dict, List, Tuple
import ast, random
from typing import Any, Dict, Tuple, List
# from sbst import run_instrumented_code # TODO: 외부 모듈과 연결

# --- # Temporary code to substitue for sbst module --- #
class ConditiontoTriple(ast.NodeTransformer):
    """
    Convert a condition expression to a triple (value, d_true, d_false)
    """

    def __init__(self):
        pass

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Compare):
            return self.visit_Compare(node)
        elif isinstance(node, ast.BoolOp):
            return self.visit_BoolOp(node)
        elif isinstance(node, ast.UnaryOp):
            return self.visit_UnaryOp(node)
        elif isinstance(node, (ast.Name, ast.Constant)):
            return node
        elif isinstance(node, ast.IfExp):
            return self.visit_IfExp(node)
        else:
            return node  # Return the node as is if not a condition

    def visit_Compare(
        self, node: ast.Compare
    ):  # <, <=, >, >=, ==, !=, in, not in -> __cmp
        self.generic_visit(node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single comparisons are supported")
        op = node.ops[0]
        left = node.left
        right = node.comparators[0]

        op_map = {  # candidates of op
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }

        if isinstance(op, ast.In):
            return ast.Call(
                func=ast.Name(id="__in_cmp", ctx=ast.Load()),
                args=[
                    left,
                    right,
                    ast.Constant(value="True"),  # True = in
                ],  # ex. [a, [1,2,3], 'in']
                keywords=[],
            )
        elif isinstance(op, ast.NotIn):
            return ast.Call(
                func=ast.Name(id="__in_cmp", ctx=ast.Load()),
                args=[left, right, ast.Constant(value=False)],  # False = not in
                keywords=[],
            )
        else:
            op_name = None
            for k, v in op_map.items():
                if isinstance(op, k):
                    op_name = v
                    break
            if op_name is None:
                raise NotImplementedError(f"Operator {type(op)} not supported")
            return ast.Call(
                func=ast.Name(id="__cmp", ctx=ast.Load()),
                args=[
                    node.left,
                    node.comparators[0],
                    ast.Constant(value=op_name),
                ],  # ex. [a, b, '<']
                keywords=[],
            )

    def visit_BoolOp(self, node: ast.BoolOp):  # and, or -> __bool_and, __bool_or
        # Handle multiple values in BoolOp
        # ex. a and (b or c)
        self.generic_visit(node)
        args = node.values
        func_name = "and" if isinstance(node.op, ast.And) else "or"
        # Wrap the frist two args
        stmt = ast.Call(
            func=ast.Name(id=f"__bool_{func_name}", ctx=ast.Load()),
            args=[args[0], args[1]],
            keywords=[],
        )
        # If more than 2 args, wrap them iteratively
        for arg in args[2:]:
            stmt = ast.Call(
                func=ast.Name(id=f"__bool_{func_name}", ctx=ast.Load()),
                args=[stmt, arg],
                keywords=[],
            )
        return stmt

    def visit_UnaryOp(self, node: ast.UnaryOp):  # 'not' -> __bool_not
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.Call(
                func=ast.Name(id="__bool_not", ctx=ast.Load()),
                args=[node.operand],
                keywords=[],
            )
        return node

    def visit_IfExp(self, node: ast.IfExp):
        self.generic_visit(node)
        if isinstance(node.body, ast.Constant) and isinstance(
            node.orelse, ast.Constant
        ):
            if node.body.value == True and node.orelse.value == False:
                return node.test
        cond = self.visit(node.test)
        then = self._to_triple_node(node.body)
        els = self._to_triple_node(node.orelse)
        return ast.Call(
            func=ast.Name(id="__ifexp", ctx=ast.Load()),
            args=[cond, then, els],
            keywords=[],
        )

    def _to_triple_node(self, node: ast.AST) -> ast.AST:
        v = self.visit(node)
        if isinstance(v, ast.Constant) and isinstance(v.value, bool):
            return ast.Call(
                func=ast.Name(id="__const", ctx=ast.Load()),
                args=[ast.Constant(value=v.value)],
                keywords=[],
            )
        return v


class Instrumenter(ast.NodeTransformer):
    """
    Wrap If and loop statements (for, while) with instrumentation code.
    __sbse_eval(<triple>, node_id).value
    """

    def __init__(self):
        self.next_id = 0
        self.node_info = (
            {}
        )  # node_id -> (line_no, kind, parent_nid, required_taken(bool))
        self.context = []  # stack of [nid, taken(bool)] # To pass info to children

    def _wrap_test(self, test: ast.AST, node_id: int) -> ast.AST:
        triple = ConditiontoTriple().visit(test)

        triple = ast.copy_location(triple, test)
        triple = ast.fix_missing_locations(triple)

        call = ast.Call(
            func=ast.Name(id="__sbse_eval", ctx=ast.Load()),
            args=[triple, ast.Constant(value=node_id)],
            keywords=[],
        )

        sub = ast.Subscript(
            value=call, slice=ast.Constant(value="value"), ctx=ast.Load()
        )

        sub = ast.copy_location(sub, test)
        sub = ast.fix_missing_locations(sub)

        return sub

    def visit_If(self, node: ast.If):
        # Assign branch id & wrap test
        nid = self.next_id
        parent_nid, required_taken = self.context[-1] if self.context else (None, None)
        self.node_info[nid] = (
            getattr(node, "lineno"),
            "if",
            parent_nid,
            required_taken,
        )  # Record line number and kind
        node.test = self._wrap_test(node.test, nid)
        self.next_id += 1

        # Visit body/orelse # Count node number starting from the children
        self.context.append((nid, True))  # If True branch
        node.body = [self.visit(n) for n in node.body]
        self.context.pop()
        self.context.append((nid, False))  # If False branch
        node.orelse = [self.visit(n) for n in node.orelse]
        self.context.pop()

        return node

    def visit_While(self, node: ast.While):
        # Assign branch id & wrap test
        nid = self.next_id
        parent_nid, required_taken = self.context[-1] if self.context else (None, None)
        self.node_info[nid] = (
            getattr(node, "lineno"),
            "while",
            parent_nid,
            required_taken,
        )  # Record line number and kind
        node.test = self._wrap_test(node.test, nid)
        self.next_id += 1

        # Visit body/orelse # Count node number starting from the children
        self.context.append((nid, True))  # If True branch
        node.body = [self.visit(n) for n in node.body]
        self.context.pop()
        self.context.append((nid, False))  # If False branch
        node.orelse = [self.visit(n) for n in node.orelse]
        self.context.pop()

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # 함수 몸체(body)만 방문
        node.body = [self.visit(n) for n in node.body]
        return node

    def visit_For(self, node: ast.For):
        nid = self.next_id
        parent_nid, required_taken = self.context[-1] if self.context else (None, None)
        self.node_info[nid] = (
            getattr(node, "lineno"),
            "for",
            parent_nid,
            required_taken,
        )
        self.next_id += 1

        node.iter = ast.Call(
            func=ast.Name(id="__for_eval", ctx=ast.Load()),
            args=[ast.Constant(value=nid), node.iter],
            keywords=[],
        )

        self.context.append((nid, True))
        node.body = [self.visit(n) for n in node.body]
        self.context.pop()

        return node


# (Instrumentation) Injection step : instrument the code to be tested
# code -> parse -> instrument -> code
# input: code
# output: (node_id, taken, d_true, d_false)


def inject_instrumentation(tree: ast.AST):
    """
    Inject instrumentation code into the AST and return the modified AST.
    Args:
        tree (ast.AST): the original AST

    Returns:
        ast.AST: new tree with instrumentation injected
        node_info: node_id -> (line_no, kind, parent_nid, required_taken)
    """
    inst = Instrumenter()
    new_tree = inst.visit(tree)
    ast.fix_missing_locations(new_tree)  # Fix missing locations after modifications
    return new_tree, inst.node_info


# (Instrumentation) Measurement step : run the instrumented code with test input
# code, input -> run -> output, branch distance
class Recorder:
    """
    Record the branch distance during execution.
    """

    def __init__(self):
        self.records: List[Tuple[int, bool, float, float]] = (
            []
        )  # (node_id, taken, d_true, d_false)

    def record(self, node_id: int, taken: bool, d_true: float, d_false: float):
        self.records.append((node_id, taken, d_true, d_false))

    def record_break(self, node_id: int):
        # For break statement, we record it as taken=True, d_true=0, d_false=inf
        self.records.append((node_id, True, 0.0, 1.0))


# Rules defined for computing branch distance
def __cmp(a, b, op: str) -> Dict[str, float]:
    if op == "<":
        d_true = max(0, a - b + 1)
        d_false = max(0, b - a)
    elif op == "<=":
        d_true = max(0, a - b)
        d_false = max(0, b - a + 1)
    elif op == ">":
        d_true = max(0, b - a + 1)
        d_false = max(0, a - b)
    elif op == ">=":
        d_true = max(0, b - a)
        d_false = max(0, a - b + 1)
    elif op == "==":
        d_true = abs(a - b)
        d_false = 1 if a != b else 0
    elif op == "!=":
        d_true = 1 if a != b else 0
        d_false = abs(a - b)
    else:
        raise NotImplementedError(f"Operator {op} not supported")
    return {"value": d_true == 0, "d_true": d_true, "d_false": d_false}


def __in_cmp(a, container, is_in: bool):
    cond = a in container
    if not isinstance(container, (list, set, tuple)):
        cond = False
        dist = 1.0
    elif container:
        dist = (
            min(abs(a - x) for x in container if isinstance(x, (int, float)))
            if not cond
            else 0.0
        )
    else:
        dist = 1.0

    if is_in:
        d_true = dist if not cond else 0.0
        d_false = 0.0 if not cond else dist
        return {"value": cond, "d_true": d_true, "d_false": d_false}
    else:
        d_true = 0.0 if not cond else dist
        d_false = dist if not cond else 0.0
        return {"value": not cond, "d_true": d_true, "d_false": d_false}


def __bool_and(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    d_true = a["d_true"] + b["d_true"]
    d_false = min(a["d_false"], b["d_false"])
    return {"value": a["value"] and b["value"], "d_true": d_true, "d_false": d_false}


def __bool_or(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    d_true = min(a["d_true"], b["d_true"])
    d_false = a["d_false"] + b["d_false"]
    return {"value": a["value"] or b["value"], "d_true": d_true, "d_false": d_false}


def __bool_not(a: Dict[str, float]) -> Dict[str, float]:
    return {"value": not a["value"], "d_true": a["d_false"], "d_false": a["d_true"]}


def __const(x):
    return {"value": x, "d_true": 0.0, "d_false": 0.0}


def __ifexp(cond, true_val, false_val):
    if cond["value"]:
        v = true_val
    else:
        v = false_val
    # Wrap v into triple if not already
    if isinstance(v, bool):
        return {"value": v, "d_true": 0.0, "d_false": 0.0}
    elif isinstance(v, dict):
        return v
    else:
        return {"value": v, "d_true": 0.0, "d_false": 0.0}


__recorder = Recorder()  # __: Not used by outside, only used internally


def __sbse_eval(triple: Dict[str, float], node_id: int) -> Dict[str, Any]:
    """
    Evaluate the condition and record the branch distance.
    """
    taken = triple["value"]
    d_true = triple["d_true"]
    d_false = triple["d_false"]
    __recorder.record(node_id, taken, d_true, d_false)
    return {"value": taken, "d_true": d_true, "d_false": d_false}


def run_instrumented_code(tree: ast.AST, test_input) -> dict:
    """
    Run the instrumented code with the given test input and return the output and branch distance.
    Args:
        tree (ast.AST): the instrumented AST
        test_input (Tuple[str, Tuple]): tuple of function name, and its arguments

    Returns:
        dict: the standard output and branch distance records
    """
    __recorder.records = []  # Clear record
    # Execute the instrumented code
    env = {
        "__cmp": __cmp,
        "__in_cmp": __in_cmp,
        "__bool_and": __bool_and,
        "__bool_or": __bool_or,
        "__bool_not": __bool_not,
        "__const": __const,
        "__ifexp": __ifexp,
        "__sbse_eval": __sbse_eval,
        "__for_eval": __for_eval,
        "__recorder": __recorder,
        "__name__": "sbst_target_module", # To avoid issues with __main__
    }

    code = compile(tree, filename="<ast>", mode="exec")
    exec(code, env)

    func_name, args = test_input
    if not isinstance(args, tuple):
        args = (args,)
    try:
        result = env[func_name](*args)
        return {"result": result, "records": __recorder.records}
    except Exception as e:
        print(f"[instrumentation] Exception during run: {e.__class__.__name__}: {e}")
        # If exception occurs, return None result and empty records
        return {"result": None, "records": []}


def __for_eval(node_id: int, iterable):
    try:
        seq = list(iterable)  # materialize once
        taken = len(seq) > 0
        d_true = 0.0 if taken else 1.0  # If loop executes at least once
        d_false = 1.0 if taken else 0.0
    except Exception as e:
        print(f"Exception in for loop node {node_id}: {e}")
        seq = []
        taken = False
        d_true = d_false = 1.0

    __recorder.record(node_id, taken, d_true, d_false)
    return seq

def fitness(
    args: Tuple[int, ...],
    tree: ast.AST,
    func_name: str,
    target: Tuple[int, str, bool],
    node_info: Dict[
        int, Tuple[int, str, Any, bool]
    ],
):
    """
    fitness = approach_level + normalize(branch_distance)
    - approach level: parent-child distance left after reaching the closest common ancestor
    - branch distance: minimum distance to get to the target branch from the closest common ancestor (d_true or d_false)
    normalize = x / (x + 1)
    
    Args:
        args (Tuple[int, ...]): input arguments to the function
        tree (ast.AST)
        func_name (str): the function name to test
        target (Tuple[int, str, bool]): target branch's (node_id, kind, desired_taken)
        node_info (Dict[int, Tuple[int, str, Any, bool]]): node_id -> (line_no, kind, parent_nid, required_taken)
    Returns:
        (fitness_value, records)
        records: [(node_id, taken, d_true, d_false), ...]
    """
    # 1. Run instrumented code to get execution records
    out = run_instrumented_code(tree, (func_name, args))
    records: List[Tuple[int, bool, float, float]] = out["records"] or []
    
    # 2. Build target chain (root -> ... -> target)
    target_nid, _kind, desired_taken = target
    target_chain: List[Tuple[int, bool]] = []  # [(nid, required_taken), ...]  root -> ...
    if target_nid in node_info:
        # Start from target node and go up to root to build target_chain
        target_chain.append((target_nid, desired_taken))
        curr = target_nid
        while curr in node_info:
            parent = node_info[curr][2]  # parent_nid
            req = node_info[curr][3]     # required_taken at parent to go to curr
            if parent is None:
                break
            target_chain.append((parent, req))
            curr = parent
        target_chain.reverse()  # root -> ... -> target
    else:
        # If cannot find target node, return worst fitness
        return float("inf"), records
    
    # 3. Get the first seen of each node to make execution path
    first_seen = {}
    exec_path: List[Tuple[int, bool]] = []
    for nid, taken, *_ in records:
        if nid not in first_seen:
            first_seen[nid] = taken
            exec_path.append((nid, taken))
            
    target_nids = {nid for nid, _ in target_chain}
    exec_path_filtered = [(nid, taken) for nid, taken in exec_path if nid in target_nids]
            
    # 4. Get divided points and get approach level
    approach_level = 0
    divided = False
    divided_at_idx = -1
    for i, (need_nid, need_taken) in enumerate(target_chain):
        if i >= len(exec_path_filtered):
            # If exec_path is shorter than target_chain
            # The rest is all not covered
            divided = True
            divided_at_idx = min(i, len(exec_path_filtered) - 1) if exec_path_filtered else 0
            approach_level += len(target_chain) - i
            break
        got_nid, got_taken = exec_path_filtered[i]
        if (got_nid != need_nid) or (got_taken != need_taken):
            divided = True
            divided_at_idx = i
            # The rest is all not covered
            approach_level += len(target_chain) - i
            break
    
    # 5. Get branch distance
    if not divided:
        # Perfect coverage
        branch_distance = 0.0
    else:
        # node_id of the closest common ancestor
        split_nid, split_required = target_chain[divided_at_idx]
        # Find the first record for split_nid
        branch_distance = float("inf")
        for nid, taken, d_true, d_false in records:
            if nid == split_nid:
                branch_distance = d_true if split_required else d_false
                break
        if branch_distance == float("inf"):
            branch_distance = 0.0  # Fallback, should not happen
            
    # 6. Normalize and compute final fitness
    def normalize(x: float) -> float:
        return x / (x + 1.0)
    
    fit = approach_level + normalize(branch_distance)
    return fit, records

def get_random(
    num_args: int,
    value_range: Tuple[int, int],
) -> Tuple[int, ...]:
    # Get random arguments
    return tuple(random.randint(value_range[0], value_range[1]) for _ in range(num_args))

def ga(
    tree: ast.AST,
    func_name: str,
    targets: List[Tuple[int, str, bool]],
    target: Tuple[int, str, bool],  # current objective
    num_args: int,
    node_info: Dict[
        int, Tuple[int, str, Any, bool]
    ],
    pop_size: int = 60,
    max_gen: int = 100,
    tournament_k: int = 3,
    elite_ratio: float = 0.1,
    gene_mut_p: float = None,   # If None, set to 1/num_args
    value_range: Tuple[int, int] = (-100, 100),
    ensure_mutation: bool = True,
    mutation_step_choices: Tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
    rng: random.Random = random,
):
    """
    Return: (best_ind, best_fit, remaining_targets, covered_cases)
    """
    # --- Helper functions ---
    if gene_mut_p is None:
        gene_mut_p = 1.0 / max(1, num_args)
        
    def clip(v: int) -> int:
        return max(value_range[0], min(value_range[1], v))

    def init_individual() -> Tuple[int, ...]:
        return tuple(rng.randint(*value_range) for _ in range(num_args))

    def evaluate(ind: Tuple[int, ...]):
        f, records = fitness(ind, tree, func_name, target, node_info)
        return f, records

    def selection(pop, fits, k=tournament_k):
        idxs = rng.sample(range(len(pop)), k)
        best_idx = min(idxs, key=lambda i: fits[i][0])
        return pop[best_idx]

    def crossover(p1: Tuple[int, ...], p2: Tuple[int, ...]) -> Tuple[int, ...]:
        # uniform crossover
        return tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(num_args))

    def mutate(ind: Tuple[int, ...]) -> Tuple[int, ...]:
        # integer perturbation; per-gene mutation with probability gene_mut_p
        vec = list(ind)
        any_flip = False
        for i in range(num_args):
            if rng.random() < gene_mut_p:
                vec[i] = clip(vec[i] + rng.choice(mutation_step_choices))
                any_flip = True
        if ensure_mutation and not any_flip:
            i = rng.randrange(num_args)
            vec[i] = clip(vec[i] + rng.choice(mutation_step_choices))
        return tuple(vec)

    def dedup(pop):
        seen, out = set(), []
        for ind in pop:
            if ind not in seen:
                out.append(ind)
                seen.add(ind)
        return out

    # Initialization
    population = [init_individual() for _ in range(pop_size)]
    fit_cache = [evaluate(ind) for ind in population]

    covered_cases = []  # [(args_tuple, covered_target)]
    best_ind, best_fit = None, float("inf")
    print(f"[GA] Starting GA for target {target} with population size {pop_size}, max generations {max_gen}")
    for gen in range(max_gen):
        print(f"[GA] Generation {gen}, best fitness so far: {best_fit:.4f}")
        # Sort (ascending: smaller fitness is better)
        ranked = sorted(zip(population, fit_cache), key=lambda x: x[1][0])
        elites_n = max(1, int(pop_size * elite_ratio))
        elites = [ind for ind, _ in ranked[:elites_n]]

        # Early stopping: current target covered
        if ranked[0][1][0] == 0.0:
            best_ind, best_fit = ranked[0][0], 0.0
            break

        # Side coverage detection: if other targets are covered in the execution path, save/remove them
        for ind, (fval, records) in ranked:
            if not records:
                continue
            seen = set()
            for rec_nid, rec_taken, *_ in records:
                key = (rec_nid, rec_taken)
                if key in seen:
                    continue
                seen.add(key)
                for other_target in list(targets[1:]):
                    if other_target[0] == rec_nid and other_target[2] == rec_taken:
                        covered_cases.append((ind, other_target))
                        targets.remove(other_target)
                        print(f"[GA] Side coverage: target {other_target} covered by args {ind}")

        # Next generation creation
        next_gen = elites[:]
        while len(next_gen) < pop_size:
            p1 = selection(population, fit_cache)
            p2 = selection(population, fit_cache)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)

        next_gen = dedup(next_gen)
        # Next generation evaluation
        population = next_gen
        fit_cache = [evaluate(ind) for ind in population]

        # Update best solution
        for ind, (fval, _rec) in zip(population, fit_cache):
            if fval < best_fit:
                best_fit = fval
                best_ind = ind

        print(f"[GA] End of Generation {gen}, best fitness: {best_fit:.4f}")
        
        # Early stopping: break loop if optimal solution found
        if best_fit == 0.0:
            break

    # Return best solution even if target is not covered; targets remain as is
    return best_ind, best_fit, targets, covered_cases


def ga_search_all(
    tree: ast.AST,
    func_name: str,
    targets: List[Tuple[int, str, bool]],
    num_args: int,
    node_info: Dict[int, Tuple[int, str, Any, bool]],
    **ga_kwargs,
):
    """
    Cover leaf targets sequentially using GA.
    If other targets are covered in the process, record and remove them immediately.
    Returns: tests = [(args_tuple, target_tuple), ...]
    """
    tests: List[Tuple[Tuple[int, ...], Tuple[int, str, bool]]] = []

    while targets:
        current_target = targets[0]
        best_args, fit, targets, covered_cases = ga(
            tree=tree,
            func_name=func_name,
            targets=targets.copy(),
            target=current_target,
            num_args=num_args,
            node_info=node_info,
            **ga_kwargs,
        )

        tests.append((best_args, current_target))
        if current_target in targets:
            # Preventing infinite loop: Even if not found, proceed to the next target
            targets.remove(current_target)
        if fit == 0.0:
            print(f"[GA] Target {current_target} covered with args {best_args}, remaining targets: {len(targets)}")
        else:
            print(f"[GA] Target {current_target} NOT covered, Best fit={fit:.4f}, args={best_args}, remaining targets: {len(targets)}")
        for args_tuple, covered_t in covered_cases:
            tests.append((args_tuple, covered_t))
        print(f"[GA] Covered {len(covered_cases)} additional targets in this run")
        print(f"[GA] Remaining targets to cover: {len(targets)}")

    return tests

# file -> instrument AST -> ga_search_all -> output tests
def main(target_path: str):
    # 1. Read code
    with open(target_path, "r") as f:
        code = f.read()

    # 2. Parse & instrument
    tree = ast.parse(code)
    tree, node_info = inject_instrumentation(
        tree
    )  # Here, mark the parent-child relationship
    print(f"[+] Instrumentation complete for {target_path}")
    
    # 3. Extract function definitions
    func_defs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]  # example1.py -> foo, bar
    num_args = {func.name: len(func.args.args) for func in func_defs}
    print(f"[+] Found {len(func_defs)} function(s): {list(num_args.keys())}")
    tests: Dict[str, List[Tuple[Tuple[int, ...], Tuple[int, str, bool]]]] = {}

    # 4. For each function, find branch leaves and run GA search
    for func in func_defs:
        func_name = func.name
        tests[func_name] = []
        
        # 4-1. Find all branch nodes (if / while / for)
        func_node_ids = [
            nid
            for nid, (lineno, kind, parent, required) in node_info.items()
            if ast.get_source_segment(code, func)
            and func.lineno <= lineno <= func.end_lineno
        ]
        branch_nodes = [
            (nid, node_info[nid])
            for nid in func_node_ids
            if node_info[nid][1] in ("if", "while", "for") # TODO: is pattern matching handled?
        ]
        
        # 4-2. Find all target leaves
        targets = []
        # print_targets = []
        for nid, (lineno, kind, parent, required) in branch_nodes:
            true_leaf, false_leaf = True, True
            for other_nid, (_, _, other_parent, other_required) in branch_nodes:
                if other_parent == nid:  # If current node is parent of other node
                    if other_required == True:
                        true_leaf = False
                    if other_required == False:
                        false_leaf = False
            if true_leaf:
                targets.append((nid, kind, True))
            if false_leaf:
                targets.append((nid, kind, False))

        # 4-3. GA search for all targets
        print(f"\n[Function] {func_name} — {len(targets)} target(s) to cover")
        if not targets:
            continue
        
        tests_found = ga_search_all(
            tree,
            func_name,
            targets,
            num_args[func_name],
            node_info,
        )
        tests[func_name].extend(tests_found)

    # 5. Print and write test files
    any_found = False
    for func_name, test in tests.items():
        if not test:
            continue
        any_found = True
        for args_tuple, (nid, kind, desired) in test:
            print(f"    nid={nid} ({kind}), desired={desired} -> args={args_tuple}")
    if not any_found:
        print("No test cases found.")

    # 6. Write generated tests
    target_module = os.path.basename(target_path).removesuffix(".py")

    for func_name, test in tests.items():
        test_file_path = os.path.join(
            os.path.dirname(target_path), f"test_{func_name}.py"
        )

        lines = [f"import {target_module}", ""]

        for idx, (args_tuple, _target_info) in enumerate(test, start=1):
            args_str = ", ".join(repr(a) for a in args_tuple)
            lines.append(f"def test_{func_name}_{idx}():")
            lines.append(f"    {target_module}.{func_name}({args_str})")
            lines.append("")

        with open(test_file_path, "w") as f:
            f.write("\n".join(lines))

        print("Test suite generated in", test_file_path)
        
    return tests

tests = main("examples/example7.py")
print(tests)