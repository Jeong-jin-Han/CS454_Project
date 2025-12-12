import argparse
import ast
import os
import random, inspect, itertools
import operator as _op
from dataclasses import dataclass
from collections import defaultdict
import time, math
from operator import length_hint
from functools import reduce
from typing import Tuple, Optional, Callable, Dict, Any
from contextlib import contextmanager
import copy
import bisect

_INT_BINOPS = {
    ast.Add: _op.add,   ast.Sub: _op.sub,   ast.Mult: _op.mul,
    ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod, ast.Pow: _op.pow,
    ast.LShift: _op.lshift, ast.RShift:_op.rshift,
    ast.BitAnd: _op.and_, ast.BitOr: _op.or_, ast.BitXor: _op.xor,
}

VERBOSE = False

def debug_print(msg):
    if VERBOSE:
        print(msg)

def _is_int_const(n):
    return isinstance(n, ast.Constant) and isinstance(n.value, int)

# int Const Folding for better seeds candidates, and for better auto parameter tuning
class IntConstFolder(ast.NodeTransformer):
    def __init__(self):
        self.env = {} # name -> int value
    
    def visit_BinOp(self, node: ast.BinOp):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if _is_int_const(node.left) and _is_int_const(node.right):
            fn = _INT_BINOPS.get(type(node.op))
            if fn:
                try:
                    v = fn(node.left.value, node.right.value)
                    return ast.copy_location(ast.Constant(v), node)
                except Exception:
                    pass
        return node
    
    def visit_UnaryOp(self, node):
        node.operand = self.visit(node.operand)
        # +C / -C / ~C
        if _is_int_const(node.operand):
            if isinstance(node.op, ast.UAdd):
                return node.operand
            if isinstance(node.op, ast.USub):
                return ast.copy_location(ast.Constant(-node.operand.value), node)
            if isinstance(node.op, ast.Invert):
                return ast.copy_location(ast.Constant(~node.operand.value), node)
        return node
    
    def visit_Name(self, node: ast.Name):
        return node
    
    def visit_Assign(self, node: ast.Assign):
        node.value = self.visit(node.value)
        rhs_is_const = _is_int_const(node.value)
        
        # only propagate to the simple name
        for target in node.targets:
            if isinstance(target, ast.Name) and rhs_is_const:
                self.env[target.id] = node.value.value
            elif isinstance(target, ast.Name):
                self.env.pop(target.id, None)
            else:
                # clear env (cut the propagation)
                self.env.clear()
        return node
    
    def visit_If(self, node):
        # for conservative way, reset the env
        node.test = self.visit(node.test)
        old = self.env.copy()
        node.body = [self.visit(n) for n in node.body] 
        self.env = old.copy()
        node.orelse = [self.visit(n) for n in node.orelse]
        self.env.clear()
        return node
    
    def visit_For(self, node: ast.For):
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        old = self.env.copy()
        node.body = [self.visit(n) for n in node.body] 
        self.env = old.copy()
        node.orelse = [self.visit(n) for n in node.orelse]
        self.env.clear()
        return node
    
    visit_While = visit_If

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old = self.env
        self.env = {}
        node.body = [self.visit(n) for n in node.body]
        self.env = old
        return node
    
    def visit_Expr(self, node: ast.Expr):
        node.value = self.visit(node.value)
        return node
    
    def visit_Return(self, node: ast.Return):
        if node.value: node.value = self.visit(node.value)
        return node


_OP = {">": _op.gt, ">=": _op.ge, "<": _op.lt, "<=": _op.le, "==": _op.eq, "!=": _op.ne}

# branch criterion tag
LT0, LE0, EQ0 = "lt0", "le0", "eq0"

def raw_f(a: int, b:int, op: str):
    if op == ">":    return (b - a) + 1, LT0    # a > b iff b-a < 0 -> f = b-a+1
    if op == ">=":   return (b - a) + 1, LE0    # a >= b iff b-a ≤ 0 -> f = b-a+1
    if op == "<":    return (a - b) + 1, LT0    # a < b iff a-b < 0 -> f = a-b+1
    if op == "<=":   return (a - b) + 1, LE0    # a <= b iff a-b ≤ 0 -> f = a-b+1
    if op == "==":   return abs(a - b), EQ0      
    if op == "!=":   return -abs(a - b), LT0    
    raise ValueError(op)  # invalidate op

def b_from_raw(f: float, criterion: str, eps: float = 1.0):
    # based on the raw_f result, we can obtain the BD(branch distance)
    if criterion == LT0:
        return 0.0 if f < 0 else (f + eps)
        # return 0.0 if f < 0 else f
        # interface value (f = 0), so we need to penalty for this case.
    if criterion == LE0:
        return 0.0 if f <= 0 else f
    if criterion == EQ0:
        return abs(f)
    raise ValueError(criterion)

def negate(op: str) -> str:
    return {">":"<=", ">=":"<", "<":">=", "<=":">", "==":"!=", "!=":"=="}[op]

def bd(a: int, b:int, op:str, want_true: bool, eps: float = 1.0) -> float:
    use_op = op if want_true else negate(op)
    f, crit = raw_f(a, b, use_op)
    return b_from_raw(f, crit, eps)

def normalise(d: float) -> float:
    return 1.0 - (1.001 ** (-d))

@dataclass
class B:
    value: bool
    d_true: float
    d_false: float
    def __bool__(self): return self.value
  
class BranchProbe:
    def __init__(self): self.records = {}
    def clear(self): 
        self.records.clear()

    @staticmethod
    def _to_scalar(v):
        if isinstance(v, B): return int(bool(v))
        if isinstance(v, bool): return int(v)
        return v
    
    @staticmethod
    def _to_B(x):
        if isinstance(x, B):
            return x
        bx = bool(x)
        d_true = 0.0 if bx else 1.0
        d_false = 1.0 if bx else 0.0
        return B(bx, d_true, d_false)
    
    def compare(self, a, c, op: str) -> B:
        def _is_01_bool(x):
            return x in (0, 1, True, False)

        # --- Case 1: (B 객체) == / != (0/1/bool)
        if isinstance(a, B) and _is_01_bool(c) and op in ("==", "!="):
            want = bool(c)
            out = (bool(a) == want) if op == "==" else (bool(a) != want)
            if op == "==":
                d_true  = a.d_true  if want else a.d_false
                d_false = a.d_false if want else a.d_true
            else:
                d_true  = a.d_false if want else a.d_true
                d_false = a.d_true  if want else a.d_false

            return B(out, d_true, d_false)
        
        # --- Case 2: (0/1/bool) == / != (B 객체)
        if isinstance(c, B) and _is_01_bool(a) and op in ("==", "!="):
            want = bool(a) 
            out = (want == bool(c)) if op == "==" else (want != bool(c))

            if op == "==":
                d_true  = c.d_true  if want else c.d_false
                d_false = c.d_false if want else c.d_true
            else:
                d_true  = c.d_false if want else c.d_true
                d_false = c.d_true  if want else c.d_false

            return B(out, d_true, d_false)
        
        # --- Case 3: 튜플/리스트 비교
        if isinstance(a, (tuple, list)) and isinstance(c, (tuple, list)):
            if op in ("==", "!="):
                if len(a) != len(c):
                    out = (op == "!=")
                    d_true  = float("inf") if op == "==" else 0.0
                    d_false = 0.0          if op == "==" else float("inf")
                    return B(out, d_true, d_false)


                sc = self._to_scalar
                a_vals = tuple(sc(x) for x in a)
                c_vals = tuple(sc(x) for x in c)

                # 스칼라가 아닌 원소가 섞여있으면 그냥 파이썬 연산자 적용
                if not all(isinstance(x, (int, bool)) for x in a_vals + c_vals):
                    out = _OP[op](a, c)
                    return B(out, 0.0 if out else 1.0, 1.0 if out else 0.0)

                # 스칼라만 있으면 요소-wise 차이를 이용해 branch distance 계산
                equal_now = (a_vals == c_vals)
                if op == "==":
                    out = equal_now
                    d_true  = 0.0 if equal_now else float(sum(abs(int(ai) - int(ci)) for ai, ci in zip(a_vals, c_vals)))
                    d_false = 1.0 if equal_now else 0.0
                    return B(out, d_true, d_false)
                else:  # "!="
                    out = (not equal_now)
                    d_true  = 0.0 if (not equal_now) else 1.0
                    d_false = 0.0 if equal_now else float(sum(abs(int(ai) - int(ci)) for ai, ci in zip(a_vals, c_vals)))
                    return B(out, d_true, d_false)
            # 리스트/튜플이지만 다른 연산자 (<, > 등)
            # 파이썬 기본 비교 결과에 따라 B 리턴
            out = _OP[op](a, c)
            return B(out, 0.0 if out else 1.0, 1.0 if out else 0.0)

        # --- Case 4: 기본 스칼라 비교 (int, float, bool 등)
        a_s =self._to_scalar(a)
        c_s = self._to_scalar(c)
        out = _OP[op](a_s, c_s)
        return B(out, bd(a_s, c_s, op, True), bd(a_s, c_s, op, False))
    
    def record_If(self, b: B, bid: int) -> bool:
        b = self._to_B(b)
        rec = self.records.get(bid)
        if rec is None:
            self.records[bid] = {
                "outcome": bool(b),
                "d_true": b.d_true,
                "d_false": b.d_false,
                "seen_true": bool(b), # Observed True branch
                "seen_false": (not bool(b)), # Observed False branch
            }
        else:
            rec["outcome"] = bool(b)
            rec["d_true"]  = min(rec["d_true"],  b.d_true)
            rec["d_false"] = min(rec["d_false"], b.d_false)
            if bool(b):
                rec["seen_true"] = True
            else:
                rec["seen_false"] = True
        return bool(b)
    
    def record_While(self, b: B, bid: int) -> bool:
        b = self._to_B(b)

        # Unlike the record function, it is recorded only once and not multiple times.
        # In the case of a while loop, if it starts with a True condition, it naturally becomes False as it exits the while loop.
        if bid not in self.records:
            self.records[bid] = {
                "outcome": bool(b),
                "d_true": b.d_true,
                "d_false": b.d_false,
                "seen_true": bool(b),
                "seen_false": (not bool(b)), 
            }
        return bool(b)
    
    def bool_and(self, L: B, R: B) -> B:
        L = self._to_B(L); R = self._to_B(R)
        out = bool(L) and bool(R)
        # L and R == T iff L == T and R == T
        # L and R == F iff L == F or R == F
        d_true = L.d_true + R.d_true
        d_false = min(L.d_false, R.d_false)
        return B(out, d_true, d_false)
    
    def bool_or(self, L: B, R: B) -> B:
        L = self._to_B(L); R = self._to_B(R)
        out = bool(L) or bool(R)
        # L or R == T iff L == T or R == T
        # L or R == F iff L == F and R == F
        d_true = min(L.d_true, R.d_true)
        d_false = L.d_false + R.d_false
        return B(out, d_true, d_false)
    
    def bool_not(self, b: B) -> B:
        b = self._to_B(b)
        out = not bool(b)
        d_true = b.d_false
        d_false = b.d_true
        return B(out, d_true, d_false)
    
    def _record_iter_entry(self, bid:int, entered: bool, len_hint):
        # entered = True : 본문이 한 번이라도 실행
        # entered = False : 본문 미실행

        already_recorded = bid in self.records
        if already_recorded: return

        iter_has_unkonwn_length = len_hint is None
        if iter_has_unkonwn_length:
            d_true = 0.0 if entered else 1.0
            d_false = 1.0 if entered else 0.0

        else:
            n = max(0, int(len_hint))
            if entered:
                d_true, d_false = 0.0, n
            else:
                d_true, d_false = 1.0, 0.0
        
        self.records[bid] = {
            "outcome": bool(entered),
            "d_true": d_true,
            "d_false": d_false,
            "seen_true": bool(entered),
            "seen_false": (not bool(entered)),
        }

    def record_For(self, iterable, bid: int, minlen):
        
        # --- len_hint 결정: minlen이 있으면 우선 사용, 없으면 length_hint 시도 ---
        len_hint = minlen
        if len_hint is None:
            try:
                lh = length_hint(iterable, -1)
                len_hint = None if lh < 0 else lh
            except Exception:
                len_hint = None

        # iterable한지 확인하기
        try:
            iter(iterable)
        except TypeError:
            self._record_iter_entry(bid, False, None)
            return iter(())

        def iter_gen():
            yielded = False
            for entry in iterable:
                if not yielded:
                    yielded = True
                    self._record_iter_entry(bid, True, len_hint)
                yield entry
            if not yielded:
                self._record_iter_entry(bid, False, len_hint)

        return iter_gen()
    
    def membership(self, a, coll) -> B:
        a_s = self._to_scalar(a)

        # --- helper: 이터러블 원소를 int로 변환 (bool -> 0/1, int만 허용)
        def _intify_iter(it):
            for x in it:
                xv = self._to_scalar(x)
                if isinstance(xv, bool):
                    xv = int(xv)
                if isinstance(xv, int):
                    yield int(xv)

        # --- helper: 정렬된 int 집합을 연속 구간으로 압축
        # e.g.) [1,2,3,5,6] -> [(1,3),(5,6)]
        def _compress_sorted_ints_to_intervals(S):
            if not S:
                return []
            intervals = []
            L = R = S[0]
            for v in S[1:]:
                if v == R + 1:
                    R = v
                else:
                    intervals.append((L, R))
                    L = R = v
            intervals.append((L, R))
            return intervals

        # --- helper: coll 객체를 interval 리스트로 변환
        # - range, list, set 등 다양한 형태 지원
        def _to_intervals(obj):
            if isinstance(obj, range):
                st, en, step = obj.start, obj.stop, obj.step
                if step == 0:
                    return []  # 안전장치

                if step > 0:
                    # 빈 range: start >= stop
                    if st >= en:
                        return []
                    if step == 1:
                        # e.g.) range(3,7) -> [(3,6)]
                        return [(st, en - 1)]
                    else:
                        # e.g.) range(1,10,2) -> [1,3,5,7,9] -> [(1,1),(3,3),(5,5),(7,7),(9,9)]
                        return _compress_sorted_ints_to_intervals(sorted(set(_intify_iter(obj))))
                else:  # step < 0
                    # 빈 range: start <= stop
                    if st <= en:
                        return []
                    if step == -1:
                        # e.g.) range(5,1,-1) -> [5,4,3,2] -> [(2,5)]
                        return [(en + 1, st)]
                    else:
                        # e.g.) range(10,0,-3) -> [10,7,4,1] -> [(1,1),(4,4),(7,7),(10,10)]
                        return _compress_sorted_ints_to_intervals(sorted(set(_intify_iter(obj))))
            else:
                # list/set 등 일반 이터러블
                try:
                    S = sorted(set(_intify_iter(obj)))
                except Exception:
                    return []
                return _compress_sorted_ints_to_intervals(S)

        # --- 집합을 intervals로 변환
        intervals = _to_intervals(coll)
        if not intervals:
            # 빈 집합 -> 무조건 False, d_true=1 (들어가기 위해 최소 이동 1 필요), d_false=0
            # e.g.) membership(5, []) -> B(False,1,0)
            return B(False, 1.0, 0.0)

        starts = [L for (L, _) in intervals]   # 모든 구간의 시작점만 리스트로 모음
        idx = bisect.bisect_right(starts, a_s) - 1

        # --- 실제 포함 여부 확인
        in_now = False
        containing = None
        if 0 <= idx < len(intervals):
            L, R = intervals[idx]
            if L <= a_s <= R:
                in_now = True
                containing = (L, R)

        # --- branch distance 계산
        if in_now:
            # 집합 안에 이미 있음 -> d_true=0
            # e.g.) membership(4, [3,4,5]) -> B(True,0,d_false=?)
            d_true = 0.0
        else:
            # 집합에 없음 -> d_true = 집합까지의 최소 거리
            cand = []
            if 0 <= idx < len(intervals):
                L, R = intervals[idx]
                if a_s > R:
                    cand.append(a_s - R)
            if idx + 1 < len(intervals):
                L2, _R2 = intervals[idx + 1]
                if a_s < L2:
                    cand.append(L2 - a_s)
            if not cand: # 경계 지점에 대한 것
                if a_s < intervals[0][0]:
                    cand.append(intervals[0][0] - a_s)
                else:
                    cand.append(a_s - intervals[-1][1])
            # e.g.) membership(10, [3,4,5]) -> d_true=5 (10에서 5까지 거리)
            d_true = float(min(cand))

        if not in_now:
            # 이미 밖에 있음 -> d_false=0
            # e.g.) membership(10, [3,4,5]) -> B(False,5,0)
            d_false = 0.0
        else:
            # 집합 안에 있음 -> d_false= 경계로부터 빠져나가기 최소 거리
            L, R = containing
            # e.g.) membership(4, [3,4,5]) -> d_false=min(4-3+1,5-4+1)=1
            d_false = float(min(a_s - L + 1, R - a_s + 1))
            # a_s - (L-1)
            # (R + 1) - a_s
            # (1) 2, 3, 4, 5 (6) | x = 3
            # 2, 3 칸

        return B(in_now, d_true, d_false)

    


__probe = BranchProbe()

# =====================
# AST transformer (include allocate the bid)
# =====================
def _as_int_const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = int(node.operand.value)
        return -v if isinstance(node.op, ast.USub) else v
    return None


class _NameReplacer(ast.NodeTransformer):
    def __init__(self, old: str, new: str):
        self.old, self.new = old, new
    def visit_Name(self, n: ast.Name):
        if n.id == self.old:
            return ast.copy_location(ast.Name(id=self.new, ctx=n.ctx), n)
        return n


class BoolToProbe(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self._next = 1 # for allocation the bid (branch id)
        self.if_root_bids = []
        self.if_guards = {}
        # 각 분기(bid)에 도달하기 위해 반드시 만족해야 하는 guard 조건들
        # 구조: {target_bid: [(guard_bid, want_true), ...]}

        self._guard_stack = []
        self._func_stack = []
        
        self.bid_to_func = {}
        # 구조: {bid: func_name}

        self.func_to_bids = defaultdict(list)
        # 구조: {func_name: [bid1, bid2, ...]}

        self._name_to_nonempty_guard = {}
        # 특정 변수 이름이 "nonempty"임을 보장하는 guard 조건 기록
        # 구조: {var_name: (cond_bid, want_true_for_nonempty)}
        
        self.loop_minlen = {}
        # 루프의 최소 반복 횟수 정보
        # 구조: {bid: 최소 반복 횟수}, 값이 없으면 0

        self.while_always_true = {}
        # while True: 같은 항상 참인 while 루프 추적
        # 구조: {bid: True} if the while condition is always True

        self._cond_stack = [False]
        # 현재 AST 탐색 위치가 조건식(test) 안인지 표시하는 플래그 스택
        # e.g.) if문의 test 안일 때 True, 바깥일 때 False

        self._def_env_stack = [dict()]
        # SSA 흉내: "이 변수는 어떤 순수 표현식으로 정의되었는가"를 추적
        # 구조: [{name: pure_expr_ast, ...}]

        self._eq_env_stack = [dict()]
        # 변수에 대해 "특정 상수값일 때 어떤 조건식이 성립하는가"를 추적
        # 구조: [{name: {const_value: cond_ast, ...}}]

    def _alloc(self):
        i = self._next; self._next += 1; return i
    
    @contextmanager
    def _enter_cond(self):
        self._cond_stack.append(True)
        try:
            yield
        finally:
            self._cond_stack.pop()

    @contextmanager
    def _suspend_cond(self):
        self._cond_stack.append(False)
        try:
            yield
        finally:
            self._cond_stack.pop()

    def _visit_in_cond(self, expr: ast.AST) -> ast.AST:
        with self._enter_cond():
            return self.visit(expr)
        
    def _in_cond(self) -> bool:
        return self._cond_stack[-1]
    
    def _eqenv(self):
        # 현재 eq_env (상수 값 -> 조건식 매핑) 조회
        return self._eq_env_stack[-1]

    def _env(self) -> dict:
        # 현재 def_env (변수 -> 순수 표현식 매핑) 조회
        return self._def_env_stack[-1]
    
    def _push_env(self):
        self._def_env_stack.append(dict(self._env()))
        self._eq_env_stack.append({k: dict(v) for k, v in self._eqenv().items()})

    def _pop_env(self):
        self._def_env_stack.pop()
        self._eq_env_stack.pop()

    def _is_pure_expr(self, node: ast.AST) -> bool:
        # pure 표현식 판별기
        if isinstance(node, (ast.Constant, ast.Name)):
            return True
        if isinstance(node, ast.UnaryOp):
            return self._is_pure_expr(node.operand)
        if isinstance(node, ast.BinOp):
            return self._is_pure_expr(node.left) and self._is_pure_expr(node.right)
        if isinstance(node, ast.BoolOp):
            return all(self._is_pure_expr(v) for v in node.values)
        if isinstance(node, ast.Compare):
            return self._is_pure_expr(node.left) and all(self._is_pure_expr(c) for c in node.comparators)
        if isinstance(node, ast.IfExp):
            return self._is_pure_expr(node.test) and self._is_pure_expr(node.body) and self._is_pure_expr(node.orelse)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return all(self._is_pure_expr(e) for e in node.elts)
        return False
  
    def _const_truth(self, node):
        # 주어진 AST 노드가 '상수적으로' 참/거짓인지 판정한다.
        # - 판정 가능: True/False (bool) 반환
        # - 판정 불가/불명: None 반환
        # 1) 리터럴 상수: int/str/bool/None 등 -> 파이썬 truthiness 규칙으로 평가
        if isinstance(node, ast.Constant):
            return bool(node.value)

        # 2) 비교식: 모든 피연산자가 상수면 평가 (a < b < c … 체인 비교도 지원)
        if isinstance(node, ast.Compare):
            def _val(n):
                return n.value if isinstance(n, ast.Constant) else None

            left = _val(node.left)
            if left is None:
                return None
            vals = [left]
            for c in node.comparators:
                v = _val(c)
                if v is None:
                    return None
                vals.append(v)

            # 파이썬 비교 연산자 매핑 (is/is not, in/not in 포함)
            ops = {
                ast.Eq: _op.eq, ast.NotEq: _op.ne,
                ast.Lt: _op.lt, ast.LtE: _op.le,
                ast.Gt: _op.gt, ast.GtE: _op.ge,
                ast.Is: _op.is_, ast.IsNot: _op.is_not,
                ast.In: lambda a, b: a in b,
                ast.NotIn: lambda a, b: a not in b,
            }
            ok = True
            for i, o in enumerate(node.ops):
                fn = ops.get(type(o))
                # 지원하지 않는 비교자거나, 비교 결과가 False면 전체 False
                if fn is None or not fn(vals[i], vals[i + 1]):
                    ok = False
                    break
            return ok  # True/False

        # 3) not 상수: 피연산자의 상수 참/거짓이 알려지면 뒤집어서 반환
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            t = self._const_truth(node.operand)
            return (not t) if isinstance(t, bool) else None

        # 4) and/or 의 상수 조합 간소화
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                all_true = True
                for v in node.values:
                    t = self._const_truth(v)
                    if t is False:
                        return False
                    if t is not True:
                        all_true = False
                return True if all_true else None
            if isinstance(node.op, ast.Or):
                any_true = False
                all_known_false = True
                for v in node.values:
                    t = self._const_truth(v)
                    if t is True:
                        return True
                    if t is not False:
                        all_known_false = False
                return False if all_known_false else None

        # 5) 컬렉션 리터럴의 truthiness
        #   - Tuple/List/Set: 모든 요소가 상수일 때만 길이로 판단
        #   - Dict: 키/값이 전부 상수(키는 None 허용)일 때만 길이로 판단

        if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
            # 비어있으면 False, 아니면 True (단, 요소가 전부 상수일 때만)
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                if all(isinstance(e, ast.Constant) for e in node.elts):
                    return bool(len(node.elts))
            else:  # Dict
                if all((k is None or isinstance(k, ast.Constant)) for k in node.keys) and \
                all(isinstance(v, ast.Constant) for v in node.values):
                    return bool(len(node.keys))

        return None
    
    def _lift_eq_in_expr(self, expr: ast.AST) -> ast.AST:
        env_eq = self._eqenv()

        class _Lift(ast.NodeTransformer):
            def visit_Compare(self, node: ast.Compare):
                # 비교식 내부 먼저 처리(안전)
                self.generic_visit(node)

                # 단순 이항 비교만 대상 (x ? y)
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    L, R = node.left, node.comparators[0]
                    op = node.ops[0]

                    def _as_simplified(var_node, const_node, is_eq):
                        # Name==IntConst 패턴만 cond로 치환
                        if not isinstance(var_node, ast.Name):
                            return None
                        K = _as_int_const(const_node)   # 정수 상수로 폴딩 가능해야 함
                        if K is None:
                            return None
                        cond = env_eq.get(var_node.id, {}).get(int(K))
                        if cond is None:
                            return None
                        # == -> cond,  != -> not cond
                        return cond if is_eq else ast.UnaryOp(op=ast.Not(), operand=ast.copy_location(cond, cond))

                    # x == K  또는  K == x
                    if isinstance(op, ast.Eq):
                        simp = _as_simplified(L, R, True) or _as_simplified(R, L, True)
                        if simp is not None:
                            return ast.copy_location(simp, node)

                    # x != K  또는  K != x
                    if isinstance(op, ast.NotEq):
                        simp = _as_simplified(L, R, False) or _as_simplified(R, L, False)
                        if simp is not None:
                            return ast.copy_location(simp, node)

                return node

        # 안전한 재구성 후 변환 (unparse->parse로 좌표 보정)
        rebuilt = ast.fix_missing_locations(ast.parse(ast.unparse(expr)).body[0].value)
        lifted = _Lift().visit(rebuilt)
        return lifted

    def _lift_rel_in_expr_for_collect(self, expr: ast.AST) -> ast.AST:
        env_eq = self._eqenv()
        op_map = {ast.Gt:">", ast.GtE:">=", ast.Lt:"<", ast.LtE:"<="}

        class LiftRel(ast.NodeTransformer):
            def visit_Compare(self, node: ast.Compare):
                self.generic_visit(node)
                # 단순 이항 비교만 대상
                if len(node.ops)!=1 or len(node.comparators)!=1:
                    return node
                op = node.ops[0].__class__
                if op not in op_map:
                    return node
                op_str = op_map[op]

                def _as_int_const_local(e):
                    # 정수 상수로 폴딩 시도 (안되면 None)
                    try:
                        v = IntConstFolder().visit(ast.fix_missing_locations(ast.copy_location(e,e)))
                        if isinstance(v, ast.Constant) and isinstance(v.value,int):
                            return v.value
                    except:
                        return None
                    return None

                def _ok(v,k,op_str):
                    # v op k 를 만족하는지 검사
                    return (op_str==">" and v>k) or (op_str==">=" and v>=k) or \
                        (op_str=="<" and v<k) or (op_str=="<=" and v<=k)

                # x (rel) K
                if isinstance(node.left, ast.Name):
                    K = _as_int_const_local(node.comparators[0])
                    if K is not None:
                        eqmap = env_eq.get(node.left.id, {})
                        chosen = [cond for vv,cond in eqmap.items()
                                if isinstance(vv,int) and _ok(vv,K,op_str)]
                        if chosen:
                            # cond가 여러 개인 경우 OR로 결합
                            return chosen[0] if len(chosen)==1 else ast.BoolOp(op=ast.Or(), values=chosen)

                # K (rel) x  ->  x (rev_rel) K 로 바꿔 동일 처리
                if isinstance(node.comparators[0], ast.Name):
                    K = _as_int_const_local(node.left)
                    if K is not None:
                        rev = {">":"<","<":">",">=":"<=","<=":">="}
                        eqmap = env_eq.get(node.comparators[0].id, {})
                        chosen = [cond for vv,cond in eqmap.items()
                                if isinstance(vv,int) and _ok(vv,K,rev[op_str])]
                        if chosen:
                            return chosen[0] if len(chosen)==1 else ast.BoolOp(op=ast.Or(), values=chosen)
                return node

        return LiftRel().visit(ast.fix_missing_locations(ast.parse(ast.unparse(expr)).body[0].value))

    def _simplify_for_collect(self, expr: ast.AST) -> ast.AST:
        # 안전한 재구성 (위치 정보 보정)
        e = ast.fix_missing_locations(ast.parse(ast.unparse(expr)).body[0].value)

        # 1) == / != 를 eqenv cond로 치환
        e = self._lift_eq_in_expr(e)

        # 2) </≤/≥/> 를 eqenv cond들의 OR로 치환
        e = self._lift_rel_in_expr_for_collect(e)

        class Boolify(ast.NodeTransformer):
            # (True if T else False) -> T
            # (False if T else True) -> not T
            # 또한 (IfExp == K)/(IfExp != K)에서 body/orelse가 정수 상수일 때
            def visit_IfExp(self, node: ast.IfExp):
                # 재귀 처리
                node.test = self.visit(node.test)
                node.body = self.visit(node.body)
                node.orelse = self.visit(node.orelse)
                # (True if T else False) / (False if T else True)
                if isinstance(node.body, ast.Constant) and isinstance(node.orelse, ast.Constant):
                    tb, fb = node.body.value, node.orelse.value
                    if isinstance(tb, bool) and isinstance(fb, bool):
                        if tb is True and fb is False:
                            return ast.copy_location(node.test, node)
                        if tb is False and fb is True:
                            return ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=node.test), node)
                return node

            def visit_Compare(self, node: ast.Compare):
                # IfExp == K / IfExp != K 접기 (양팔 Int 상수 가능할 때)
                self.generic_visit(node)
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    op = node.ops[0]
                    if isinstance(op, (ast.Eq, ast.NotEq)):
                        L, R = node.left, node.comparators[0]

                        def _as_int_const(e):
                            try:
                                v = IntConstFolder().visit(
                                    ast.fix_missing_locations(ast.copy_location(e, e))
                                )
                                if isinstance(v, ast.Constant) and isinstance(v.value, int):
                                    return v.value
                            except Exception:
                                pass
                            return None

                        def _fold_ifexp_cmp(ife: ast.IfExp, other):
                            # (IfExp == K)/(IfExp != K)에서 body/orelse가 상수면 test로 환원
                            K = _as_int_const(other)
                            if K is None:
                                return None
                            C1 = _as_int_const(ife.body)
                            C2 = _as_int_const(ife.orelse)
                            if C1 is None or C2 is None:
                                return None
                            t = ife.test
                            is_eq = isinstance(op, ast.Eq)
                            b1 = (C1 == K); b2 = (C2 == K)
                            if b1 and not b2:
                                return t if is_eq else ast.UnaryOp(op=ast.Not(), operand=t)
                            if (not b1) and b2:
                                return ast.UnaryOp(op=ast.Not(), operand=t) if is_eq else t
                            # 둘 다 동일 값 -> 상수 부울
                            val = (b1 == True)
                            return ast.Constant(val if is_eq else (not val))

                        # IfExp가 좌/우에 있는 경우 모두 처리
                        if isinstance(L, ast.IfExp):
                            folded = _fold_ifexp_cmp(L, R)
                            if folded is not None:
                                return ast.copy_location(folded, node)
                        if isinstance(R, ast.IfExp):
                            folded = _fold_ifexp_cmp(R, L)
                            if folded is not None:
                                return ast.copy_location(folded, node)
                return node

        # 3) 경량 Boolify 적용
        e2 = Boolify().visit(e)
        return ast.fix_missing_locations(e2)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._func_stack.append(node.name)
        self._push_env()
        node = self.generic_visit(node) # internel node visit
        self._pop_env()
        self._func_stack.pop()
        return node 

    def visit_If(self, node: ast.If):
        # 1. 조건식을 표준화(==/!= lifting, </≤/≥/> lifting, Boolify) # DEBUG
        # test_for_collect  = self._simplify_for_collect(node.test)
        # debug_print(f"[DEBUG]\n{ast.unparse(test_for_collect)}")

        node.test = self._visit_in_cond(node.test) # Transform internal comparison/BoolOp into B/Probecalls

        # 2. allocate the new bid(branch id) and record the meta data
        bid = self._alloc()


        self.if_root_bids.append(bid)
        self.if_guards[bid] = list(self._guard_stack) # current context guard 
        cur_func = self._func_stack[-1] if self._func_stack else "<module>"

        self.bid_to_func[bid] = cur_func
        self.func_to_bids[cur_func].append(bid)


        # 3. Transform the final branch point to call __probe.record() to record results/distances/observations
        node.test = ast.Call(
            func = ast.Attribute(
                value = ast.Name(id ="__probe", ctx = ast.Load()),
                attr = "record_If", ctx=ast.Load()),
            args=[node.test, ast.Constant(bid)], keywords=[]
        )

        # 4. The 'then' block is visited with the guard that this 'if' was True pushed onto the stack.
        self._guard_stack.append((bid, True))
        self._push_env()
        node.body = [self.visit(n) for n in node.body]
        self._pop_env()
        self._guard_stack.pop()

        # 5. The 'else' block is visited with the guard that this 'if' was False pushed onto the stack.
        self._guard_stack.append((bid, False))
        self._push_env()
        node.orelse = [self.visit(n) for n in node.orelse]
        self._pop_env()
        self._guard_stack.pop()

        return node
  
    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)

        op_map = {"Gt": ">", "GtE": ">=", "Lt": "<", "LtE": "<=", "Eq": "==", "NotEq": "!="}
        rev_rel = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "==": "==", "!=": "!="}

        # === 비교 원자 생 ===
        def _mk_cmp(L, R, op_str):
            return ast.Call(
                func = ast.Attribute(value=ast.Name(id = "__probe", ctx=ast.Load()),
                                    attr="compare", ctx=ast.Load()),
                args = [L, R, ast.Constant(op_str)],
                keywords=[]
            )
        
        # eqenv로부터 부등식 lifting:
        #  x (rel) K   ->  OR_{v op K} cond(x==v)
        #  cond 들이 여러 개면 OR로 조립, 하나도 없으면 False (도달 불가 판단)
        def _lift_rel_from_eqenv_Compare(var_node, const_node, op_str):
            if not isinstance(var_node, ast.Name):
                return None
            K = _as_int_const(const_node)
            if K is None:
                return None

            eqmap = self._eqenv().get(var_node.id, {})
            if not eqmap:
                return None

            def _ok(v, k, op):
                if op == ">":  return v >  k
                if op == ">=": return v >= k
                if op == "<":  return v <  k
                if op == "<=": return v <= k
                return False

            chosen = []
            for v, cond_ast in eqmap.items():
                if isinstance(v, int) and _ok(v, K, op_str):
                    # 위치정보 유지
                    chosen.append(ast.fix_missing_locations(ast.copy_location(cond_ast, cond_ast)))

            if not chosen:
                # eqenv 지식상 절대 참이 될 수 없음 -> False (조건 컨텍스트에서 안전)
                return ast.Constant(False)

            # OR로 결합: __probe.bool_or(a, b)
            out = chosen[0]
            for c in chosen[1:]:
                out = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="__probe", ctx=ast.Load()),
                                    attr="bool_or", ctx=ast.Load()),
                    args=[out, c], keywords=[]
                )
            return out

        def lift_eq_from_name_const_Compare(var_node, const_node, op_str):
            if not isinstance(var_node, ast.Name):
                return None
            K = _as_int_const(const_node)
            if K is None:
                return None
            eqmap = self._eqenv().get(var_node.id, {})
            cond = eqmap.get(int(K))
            if cond is None:
                return None
            return cond if op_str == "==" else ast.Call(
                func=ast.Attribute(value=ast.Name(id="__probe", ctx=ast.Load()),
                                attr="bool_not", ctx=ast.Load()),
                args=[cond], keywords=[]
            )        

        # -----------------------------------------
        # 1. Single comparison: a op b form (1 operator, 1 comparator)
        # e.g., x < 10, y in {1,2,3}, (x%2) == 0
        if len(node.ops) == 1 and len(node.comparators) == 1:
            # 파편 꺼내기
            op_obj = node.ops[0]
            right = node.comparators[0]
            left = node.left
            # (1-a) membership 처리: in / not in -> __probe.membership 위임
            # - 전개(OR/AND) 대신 런타임 membership이 직접 거리(d_true/d_false)를 계산
            # - 큰/비연속/동적 컨테이너에도 스케일 좋음
            if isinstance(op_obj, ast.In) or isinstance(op_obj, ast.NotIn):
                # __probe.membership(left, right) : B 반환
                memb_call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="__probe", ctx=ast.Load()),
                                    attr="membership", ctx=ast.Load()),
                    args=[left, right], keywords=[]
                )
                # not in -> bool_not(membership(...))
                return memb_call if isinstance(op_obj, ast.In) else ast.Call(
                    func=ast.Attribute(value=ast.Name(id="__probe", ctx=ast.Load()),
                                    attr="bool_not", ctx=ast.Load()),
                    args=[memb_call], keywords=[]
                )

            # (1-b) 일반 비교자
            op_name = type(op_obj).__name__
            if op_name in op_map:
                op_str = op_map[op_name]

                # == / != : eqenv 리프팅 우선
                #  x==K  -> cond(x==K)
                #  x!=K  -> not cond(x==K)
                if op_str in ("==", "!="):

                    # 좌/우 교차 시도 (x==K 또는 K==x)
                    simp = lift_eq_from_name_const_Compare(left, right, op_str) or \
                        lift_eq_from_name_const_Compare(right, left, op_str)
                    if simp is not None:
                        simp = ast.copy_location(simp, node)
                        # 단순화 결과 내부에도 비교가 있을 수 있으니 '조건 컨텍스트'로 재방문하여 추가 변환 허용
                        return self._visit_in_cond(simp) if self._in_cond() else self.visit(simp)

                    # 리프팅 실패 -> 원자 비교로 폴백
                    return _mk_cmp(left, right, op_str)

                # >, >=, <, <= : eqenv 부등식 리프팅
                if op_str in (">", ">=", "<", "<="):
                    simp = _lift_rel_from_eqenv_Compare(left, right, op_str)
                    if simp is None:
                        # K (rel) x 형태면 역관계로 뒤집어 재시도:  K < x  ->  x > K
                        simp = _lift_rel_from_eqenv_Compare(right, left, rev_rel[op_str])
                    if simp is not None:
                        simp = ast.copy_location(simp, node)
                        return self._visit_in_cond(simp) if self._in_cond() else self.visit(simp)

                    # 리프팅 실패 -> 원자 비교로 폴백
                    return _mk_cmp(left, right, op_str)

            # 지원 밖(알 수 없는 비교자) -> 원본 유지
            return node

        # 2. chain compariosn: a < b <c 
        # a < b < c
        # left = a, ops = [Lt, Lt], comparators = [b, c]
        current_left = node.left
        current = None

        for op, right in zip(node.ops, node.comparators):
            op_name = type(op).__name__
            if op_name not in op_map:
                return node
            
            op_str = op_map[op_name]

            tmp = _mk_cmp(current_left, right, op_str)

            current = tmp if current is None else ast.Call(
                func = ast.Attribute(
                    value=ast.Name(id="__probe", ctx = ast.Load()),
                    attr="bool_and", ctx=ast.Load()
                ),
                args = [current, tmp],
                keywords=[]
            )

            current_left = right

        return current
    
    def visit_BoolOp(self, node: ast.BoolOp):
        # 중요한 역할을 함
        # 이것이 없어도 작동하는데 문제는 없지만, 대신 거리 계산이 어려워진다는 문제점 존재
        # A and B and C -> __probe.bool_and(__probe.bool_and(A,B), C)
        self.generic_visit(node)
        op_attr = "bool_and" if isinstance(node.op, ast.And) else "bool_or"
        cur = node.values[0]
        for nxt in node.values[1:]:
            cur = ast.Call(func=ast.Attribute(value=ast.Name(id="__probe", ctx=ast.Load()),
                                              attr=op_attr, ctx=ast.Load()),
                           args=[cur, nxt], keywords=[])
        return cur
    
    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)

        if isinstance(node.op, ast.Not):
            opnd = node.operand

            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="__probe", ctx=ast.Load()),
                        attr="bool_not",
                        ctx=ast.Load(),
                    ),
                    args=[opnd],
                    keywords=[],
                ),
                node,
            )
        return node

    def visit_While(self, node: ast.While):
        # Check if the while condition is always True (e.g., while True:)
        original_test = node.test
        is_always_true = (self._const_truth(original_test) is True)
        
        node.test = self._visit_in_cond(node.test)
        bid = self._alloc()

        self.if_root_bids.append(bid)
        self.if_guards[bid] = list(self._guard_stack)
        cur_func = self._func_stack[-1] if self._func_stack else "<module>"
        self.bid_to_func[bid] = cur_func
        self.func_to_bids[cur_func].append(bid)

        # Mark if this is a while-True loop (False branch unreachable)
        if is_always_true:
            self.while_always_true[bid] = True
            debug_print(f"Detected while True: bid={bid}")

        node.test = ast.Call(
            func = ast.Attribute(value=ast.Name(id="__probe", ctx=ast.Load()),
                                attr="record_While", ctx=ast.Load()),
            args = [node.test, ast.Constant(bid)],
            keywords = []
        )

        self._guard_stack.append((bid, True))
        self._push_env()
        node.body = [self.visit(n) for n in node.body]
        self._pop_env()
        self._guard_stack.pop()

        node.orelse = [self.visit(n) for n in node.orelse]
        return node

    def visit_For(self, node: ast.For):
        bid = self._alloc()
        extra_guards = []            # iter 분석에서 추가로 얻은 non-empty guards

        rng = node.iter              # for <target> in rng:
        # (A) IfExp iter 처리 (지금 코드 그대로 유지)
        if isinstance(rng, ast.IfExp) and _is_const_collection(rng.body) and _is_const_collection(rng.orelse):
            body_len   = _coll_len(rng.body)
            orelse_len = _coll_len(rng.orelse)

            # test를 '조건 컨텍스트'로 방문하여 비교/부울들을 계측 가능한 형태로 변환
            test_v   = self._visit_in_cond(rng.test)

            # test 자체도 분기 하나로 계측: cond_bid 할당 -> record_If(test, cond_bid)
            cond_bid = self._alloc()
            test_rec = ast.Call(
                func=ast.Attribute(value=ast.Name(id="__probe", ctx=ast.Load()),
                                attr="record_If", ctx=ast.Load()),
                args=[test_v, ast.Constant(cond_bid)],
                keywords=[]
            )

            # iter 자리에 IfExp(test_rec, body, orelse) 유지 (body/orelse는 안전히 방문)
            node.iter = ast.IfExp(
                test=test_rec,
                body=self.visit(rng.body),
                orelse=self.visit(rng.orelse),
            )

            # 어느 쪽이 non-empty인지에 따라 가드 극성 결정
            if (body_len >= 1) and (orelse_len == 0):
                extra_guards.append((cond_bid, True))    # test True -> non-empty
            elif (body_len == 0) and (orelse_len >= 1):
                extra_guards.append((cond_bid, False))   # test False -> non-empty
            rng = node.iter  # 이후 흐름에서 (B) 검사 등에 동일 변수 사용

        # (B) iter가 Name이면, assign 등에서 저장해 둔 non-empty 가드를 붙인다
        #    - 예: if xs: xs_nonempty_guard 기록 -> 이후 for x in xs: 진입 가드로 활용
        if isinstance(rng, ast.Name):
            nm = rng.id
            if nm in self._name_to_nonempty_guard:
                cond_bid, want_true = self._name_to_nonempty_guard[nm]
                extra_guards.append((cond_bid, want_true))

        self.if_root_bids.append(bid)
        self.if_guards[bid] = list(self._guard_stack) + extra_guards
        
        cur_func = self._func_stack[-1] if self._func_stack else "<module>"
        self.bid_to_func[bid] = cur_func
        self.func_to_bids[cur_func].append(bid)

        self.loop_minlen[bid] = None

        node.iter = ast.Call(
            func = ast.Attribute(value = ast.Name(id = "__probe", ctx = ast.Load()),
                                 attr = "record_For", ctx = ast.Load()),
            args = [node.iter, ast.Constant(bid), ast.Constant(self.loop_minlen[bid])],
            keywords = []
        )

        self._guard_stack.append((bid, True))
        node.body = [self.visit(n) for n in node.body]
        self._guard_stack.pop()

        node.orelse = [self.visit(n) for n in node.orelse]
        return node
    
    def visit_IfExp(self, node: ast.IfExp):
        # 조건 컨텍스트에서 먼저 '원본 노드'로 상수 판정을 시도
        if self._in_cond():
            tb = self._const_truth(node.body)
            fb = self._const_truth(node.orelse)

            if tb is True and fb is False:
                # (True if cond else False) -> cond
                # test만 조건 컨텍스트로 방문
                return self._visit_in_cond(node.test)

            if tb is False and fb is True:
                # (False if cond else True) -> not cond
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="__probe", ctx=ast.Load()),
                        attr="bool_not",
                        ctx=ast.Load(),
                    ),
                    args=[ self._visit_in_cond(node.test) ],
                    keywords=[],
                )

            # ★ 단순화 불가: body/orelse는 '조건 컨텍스트를 끄고' 방문
            test_v   = self._visit_in_cond(node.test)  # test는 조건 컨텍스트로 변환(계측/리프팅 허용)
            with self._suspend_cond():                 # body/orelse는 값 컨텍스트로 안전 방문
                body_v   = self.visit(node.body)
                orelse_v = self.visit(node.orelse)
            node = ast.IfExp(test=test_v, body=body_v, orelse=orelse_v)
            return node

        # 조건 컨텍스트가 아니면 기존대로 전개(일반 재귀 방문)
        node.test   = self.visit(node.test)
        node.body   = self.visit(node.body)
        node.orelse = self.visit(node.orelse)
        return node
    
    def visit_Match(self, node: ast.Match):
        def _withloc(n: ast.AST, ref: ast.AST):
            ast.copy_location(n, ref)

            return ast.fix_missing_locations(n)

        # (1) subject 캡처: 한 번만 평가되도록 임시 변수에 저장
        tmp_name  = f"_match_tmp_{self._alloc()}"
        tmp_store = ast.Name(id=tmp_name, ctx=ast.Store())
        tmp_load  = ast.Name(id=tmp_name, ctx=ast.Load())

        subj_expr = self.visit(node.subject)          # subject 내부는 먼저 방문/변환
        assign    = ast.Assign(targets=[tmp_store], value=subj_expr)
        assign    = _withloc(assign, node)

        left_for_cmp = tmp_load

        # 와일드카드(_) 인지 판정
        def _is_wildcard(pat: ast.AST) -> bool:
            if isinstance(pat, ast.MatchAs):
                return pat.name == "_" or (pat.name is None and pat.pattern is None)
            return False

        # case 패턴에서 "상수 값" 노드들을 추출 (OR/AND로 구성 가능한지 확인)
        def _value_nodes_from_pattern(pat: ast.AST):
            def _const_expr_from_subpat(sp: ast.AST):
                # 서브패턴을 정수/불 상수(또는 부호 단항 정수)로 끌어낼 수 있으면 반환
                if isinstance(sp, ast.MatchSingleton) and (sp.value in (True, False)):
                    return ast.Constant(sp.value)
                if isinstance(sp, ast.MatchValue):
                    v = sp.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, (int, bool)):
                        return v
                    if (isinstance(v, ast.UnaryOp)
                        and isinstance(v.operand, ast.Constant) and isinstance(v.operand.value, int)
                        and isinstance(v.op, (ast.UAdd, ast.USub))):
                        return v
                return None

            # A | B | C  (OR 패턴): 재귀적으로 전개해서 값 리스트를 합침
            if isinstance(pat, ast.MatchOr):
                acc = []
                for p in pat.patterns:
                    sub = _value_nodes_from_pattern(p)
                    if sub is None:
                        return None
                    acc.extend(sub)
                return acc

            # 시퀀스 패턴(정해진 길이, 각 요소가 상수여야 함): 튜플 정수/불 상수의 1개 값으로 모델링
            if isinstance(pat, ast.MatchSequence) and hasattr(pat, "patterns"):
                elts = []
                for sp in pat.patterns:
                    ce = _const_expr_from_subpat(sp)
                    if ce is None:
                        return None
                    elts.append(ce)
                return [ast.Tuple(elts=elts, ctx=ast.Load())]

            # 단일 값/단일톤 패턴: 헬퍼만 호출해서 처리 (중복 제거)
            ce = _const_expr_from_subpat(pat)
            if ce is not None:
                return [ce]
            return None  # 위 규칙으로는 전개 불가 -> fallback

        # 여러 비교식을 OR로 연결
        def _or_chain(nodes):
            if not nodes:
                return None
            cur = nodes[0]
            for nxt in nodes[1:]:
                cur = ast.BoolOp(op=ast.Or(), values=[cur, nxt])
            return cur

        prepared = []     # [(test_expr, body_nodes), ...]
        default_body = None

        # (2) 각 case를 if용 (test, body) 튜플로 준비
        for cs in node.cases:
            # 와일드카드는 default(else)로 보관
            if _is_wildcard(cs.pattern):
                default_body = cs.body
                continue

            # 패턴을 값 비교로 모델링 가능해야만 진행
            vals = _value_nodes_from_pattern(cs.pattern)
            if vals is None:
                # 지원 불가 패턴 포함 -> 원본 노드로 돌려보냄 (일반 방문으로 처리)
                self.generic_visit(node)
                return node

            # (left_for_cmp == v1) or (== v2) or ...
            eq_atoms  = [ast.Compare(left=left_for_cmp, ops=[ast.Eq()], comparators=[v]) for v in vals]
            test_expr = _or_chain(eq_atoms)

            # case guard가 있으면 AND로 결합
            if cs.guard is not None:
                guard = self.visit(cs.guard)
                test_expr = ast.BoolOp(op=ast.And(), values=[test_expr, guard])

            prepared.append((test_expr, cs.body))

        # 모든 case가 와일드카드뿐이었다면: assign 후 default만 실행
        if not prepared and default_body is not None:
            return [assign] + [self.visit(n) for n in default_body]

        # (3) if-else 사슬 구성
        head_if = None
        cur_if  = None
        for test_expr, body_nodes in prepared:
            if_node = ast.If(test=test_expr, body=body_nodes, orelse=[])
            if_node = _withloc(if_node, node)
            if head_if is None:
                head_if = if_node
                cur_if  = if_node
            else:
                cur_if.orelse = [if_node]
                cur_if        = if_node

        # default(와일드카드)가 있으면 마지막 else에 배치
        if default_body is not None:
            cur_if.orelse = default_body

        # 위치 정보 보정 후, 구성된 if 사슬을 다시 방문하여 일반 분기 계측 파이프라인(visit_If 등) 태움
        head_if    = _withloc(head_if, node)
        visited_if = self.visit(head_if)

        # 최종 결과: [subject 캡처(assign), if-else 사슬]
        debug_print(f"assign {assign}")
        debug_print(f"visited_if {visited_if}")
        return [assign, visited_if]

    def _replace_name_in_ifexp_tests(self, ife: ast.IfExp, old: str, new: str):
        def rec(n):
            if isinstance(n, ast.IfExp):
                n = copy.deepcopy(n)
                n.test   = _NameReplacer(old, new).visit(n.test)
                n.body   = rec(n.body)
                n.orelse = rec(n.orelse)
                return n
            return n
        return rec(ife)

    def _replace_name_in_ifexp_tests(self, ife: ast.IfExp, old: str, new: str):
        # [유틸] IfExp 트리 안에서 'test' 부분에만 한정해 Name(old) -> Name(new) 치환.
        # - body/orelse 내부 IfExp도 재귀 처리(깊은 복사로 원본 훼손 방지)
        # - 목적: x = (A if x>0 else B)처럼, test에서 '자기 자신'을 참조하는 경우
        #        대입 전 값(pre-state)으로 test를 평가할 수 있게 만들기 위함.
        def rec(n):
            if isinstance(n, ast.IfExp):
                n = copy.deepcopy(n)
                n.test   = _NameReplacer(old, new).visit(n.test)  # test에만 치환 적용
                n.body   = rec(n.body)
                n.orelse = rec(n.orelse)
                return n
            return n
        return rec(ife)

    def _mk_and(self, a, b):
        return b if a is None else ast.BoolOp(op=ast.And(), values=[a, b])

    def _mk_not(self, e):
        return ast.UnaryOp(op=ast.Not(), operand=e)

    def _collect_ifexp_const_paths(self, ife: ast.IfExp):
        # [핵심] IfExp 트리에서 'leaf가 정수 상수(int)'인 경로들을 수집.
        # - 각 leaf까지의 경로 조건을 _simplify_for_collect로 표준화하여 함께 반환.
        # - 결과: [(정수상수K, 경로조건AST), ...]
        # - 용도: eqenv[name][K] = 그 K가 되도록 보장하는 경로조건  로 채우기.
        out = []
        def dfs(node, path_cond):
            if isinstance(node, ast.IfExp):
                t = self._simplify_for_collect(node.test)     # test 표준화(리프팅/부울화)
                dfs(node.body,   self._mk_and(path_cond, t))  # T-분기
                dfs(node.orelse, self._mk_and(path_cond, self._mk_not(t)))  # F-분기
            elif isinstance(node, ast.Constant) and isinstance(node.value, int):
                cond = path_cond if path_cond is not None else ast.Constant(True)
                out.append((int(node.value), ast.fix_missing_locations(cond)))
        dfs(ife, None)
        return out

    def _replace_name_in_ifexp_tests(self, ife: ast.IfExp, old: str, new: str):
        # (중복 정의: 위 함수와 동일) IfExp test 안에서만 old->new 치환
        def rec(n):
            if isinstance(n, ast.IfExp):
                n = copy.deepcopy(n)
                n.test   = _NameReplacer(old, new).visit(n.test)
                n.body   = rec(n.body)
                n.orelse = rec(n.orelse)
                return n
            return n
        return rec(ife)

    def _name_used_in_ifexp_tests(self, ife: ast.IfExp, target: str) -> bool:
        # IfExp 트리의 test들에서 target 이름이 쓰였는지 확인.
        # - x = ( ... if x > 0 else ... ) 같은 자기참조 대입 검출에 사용.
        used = False
        class F(ast.NodeVisitor):
            def visit_IfExp(self, n):
                self.visit(n.test); self.visit(n.body); self.visit(n.orelse)
            def visit_Name(self, n):
                nonlocal used
                if n.id == target: used = True
        F().visit(ife)
        return used

    def visit_Assign(self, node: ast.Assign):

        # --- A) 환경 갱신 & IfExp 전처리(통합) ---
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            val  = node.value

            if isinstance(val, ast.IfExp):
                # 1) (선행) IfExp가 '상수 컬렉션 선택'이면 test를 계측(record_If)하고
                #   어떤 극성에서 non-empty인지 가드를 기록해둔다.
                is_const_coll = _is_const_collection(val.body) and _is_const_collection(val.orelse)
                test_rec = None
                if is_const_coll:
                    body_len   = _coll_len(val.body)
                    orelse_len = _coll_len(val.orelse)

                    test_v   = self._visit_in_cond(val.test)  # test는 조건 컨텍스트로 변환(리프팅 허용)
                    cond_bid = self._alloc()
                    test_rec = ast.Call(
                        func=ast.Attribute(value=ast.Name(id="__probe", ctx=ast.Load()),
                                        attr="record_If", ctx=ast.Load()),
                        args=[test_v, ast.Constant(cond_bid)],
                        keywords=[]
                    )
                    # non-empty 극성 기록 (for/iter 가드로 재사용하기 위함)
                    if (body_len >= 1) and (orelse_len == 0):
                        self._name_to_nonempty_guard[name] = (cond_bid, True)
                    elif (body_len == 0) and (orelse_len >= 1):
                        self._name_to_nonempty_guard[name] = (cond_bid, False)
                    # 둘 다 비거나 둘 다 비비면 non-empty 보장X -> 기록 생략

                # 2) test에 자기 자신(name)이 쓰였는지 판단
                need_pre = self._name_used_in_ifexp_tests(val, name)

                if need_pre:
                    # 2-1) pre-state 캡처: 대입 전 값을 보존
                    pre_name = f"__pre_{name}_{self._alloc()}"
                    pre_assign = ast.Assign(
                        targets=[ast.Name(id=pre_name, ctx=ast.Store())],
                        value=ast.Name(id=name, ctx=ast.Load())
                    )
                    ast.copy_location(pre_assign, node)

                    # 2-2) eqenv 경로 수집: test 내부 name->pre_name으로 치환 후, 각 정수 leaf 경로 수집
                    val_for_paths = self._replace_name_in_ifexp_tests(val, name, pre_name)
                    paths = self._collect_ifexp_const_paths(val_for_paths)
                    if paths:
                        eqmap = self._eqenv().setdefault(name, {})
                        for K, cond_ast in paths:
                            eqmap[int(K)] = cond_ast       # name == K 를 보장하는 경로조건 기록
                        self._env().pop(name, None)        # 인라인(def_env)은 제거(비순수로 간주)
                                                    # (= eqenv와 충돌 방지)

                    # 2-3) 실제 대입식 구성
                    #     - 상수 컬렉션 IfExp면 test를 test_rec(record_If로 감싼 것)으로 교체
                    #     - 그 뒤 test 안의 name -> pre_name 치환
                    new_value = val
                    if test_rec is not None:
                        new_value = ast.IfExp(test=test_rec, body=val.body, orelse=val.orelse)
                    new_value = self._replace_name_in_ifexp_tests(new_value, name, pre_name)
                    new_value = self.visit(new_value)  # 내부 노드들 일반 방문

                    new_assign = ast.Assign(targets=node.targets, value=new_value)
                    ast.copy_location(new_assign, node)
                    return [pre_assign, new_assign]     # pre-캡처 후 실제 대입을 두 문장으로 반환

                else:
                    # 3) 자기참조가 아니면, 원본 IfExp 기준으로 eqenv 경로 수집만 수행
                    paths = self._collect_ifexp_const_paths(val)
                    if paths:
                        eqmap = self._eqenv().setdefault(name, {})
                        for K, cond_ast in paths:
                            eqmap[int(K)] = cond_ast
                        self._env().pop(name, None)
                    else:
                        # 경로를 만들 수 없으면 eqenv[name] 제거
                        self._eqenv().pop(name, None)
                        # 그리고 값이 순수식이면 def_env[name]에 등록(인라인 후보),
                        # 아니면 name 관련 def_env 엔트리 제거
                        if self._is_pure_expr(val):
                            self._env()[name] = val
                        else:
                            self._env().pop(name, None)

                    # 3-1) 실제 대입식: 상수 컬렉션 IfExp면 test를 record_If로 감싼 버전 사용
                    if test_rec is not None:
                        node.value = ast.IfExp(
                            test=test_rec,
                            body=self.visit(val.body),
                            orelse=self.visit(val.orelse)
                        )
                    else:
                        node.value = self.visit(val)
                    return node

            # IfExp가 아닌 일반 케이스
            if self._is_pure_expr(val):
                # 순수식: def_env[name] 갱신(인라인 후보), eqenv[name]은 클리어(충돌 방지)
                self._env()[name] = val
                self._eqenv().pop(name, None)
            else:
                # 비순수/함수호출/속성/첨자 등: name에 대한 def/eq 지식 제거(보수적)
                self._env().pop(name, None)
                self._eqenv().pop(name, None)

        else:
            # 다중 타깃 대입 등 복잡한 형태: 보수적으로 환경 초기화
            self._def_env_stack[-1].clear()
            self._eqenv().clear()

        # --- B) 일반 방문 (우변 변환 적용) ---
        node.value = self.visit(node.value)
        return node   



def _is_const_collection(n):
    # 상수 컬렉션 리터럴 여부 (List/Tuple/Set/Dict)
    if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
        return all(isinstance(e, ast.Constant) for e in n.elts)
    if isinstance(n, ast.Dict):
        return all((k is None or isinstance(k, ast.Constant)) for k in n.keys) and \
            all(isinstance(v, ast.Constant) for v in n.values)
    return False

def _coll_len(n):
    # 상수 컬렉션 길이
    if isinstance(n, (ast.List, ast.Tuple, ast.Set)): return len(n.elts)
    if isinstance(n, ast.Dict): return len(n.keys)
    return None


def _instrument_and_load_internal(code: str):
    """Internal function that returns low-level instrumentation objects.
    Returns: (namespace, tx, instrumented_code, original_tree)
    """
    # code -> AST parsing
    tree = ast.parse(code)
    tmp_tree = IntConstFolder().visit(tree)
    ast.fix_missing_locations(tmp_tree)
    tmp_source = ast.unparse(tmp_tree)
    debug_print(f"[tmp_source]\n {tmp_source}")

    # create Bool -> Proble generator 
    tx = BoolToProbe()
    # convert the AST like form of B/Probe 
    new_tree = tx.visit(tmp_tree)
    ast.fix_missing_locations(new_tree)
    # [DEBUG] return the new AST to the code 
    code = ast.unparse(new_tree)
    debug_print(f"[code]\n {code}")

    ns = {"__probe": __probe, "B": B}
    # load ftn/class def to the ns
    exec(compile(code, "<inst>", "exec"), ns, ns)

    return ns, tx, code, tree


def instrument_and_load(source_code: str):
    """
    Main API function for instrumenting and loading Python code.
    Returns: (namespace, traveler, record, instrumented_tree)
    
    This matches the sbst_core.py API where:
    - namespace: dict with instrumented functions and classes
    - traveler: contains functions list, branches info, and metadata
    - record: for recording execution traces
    - instrumented_tree: the original AST before instrumentation
    """
    # Use the internal instrumentation
    ns, tx, code, tree = _instrument_and_load_internal(source_code)
    
    # Create compatibility wrappers
    traveler = Traveler(tx, ns, tree)
    record = Record()
    
    # Return in the expected format
    return ns, traveler, record, tree 



def seed_candidates_for_target(param_names,verbose=False):
    n = len(param_names)
    seeds = []
    seen = set()

    def push(t):
        if t not in seen:
            seen.add(t)
            seeds.append(t)

    # 1) 원점
    push(tuple(0 for _ in range(n)))

    # 2) 축별 ±1
    for i in range(n):
        v = [0]*n
        v[i] = 1
        push(tuple(v))
        v[i] = -1
        push(tuple(v))


    if verbose:
        debug_print(f"[seed] {len(seeds)} seeds generated for {n} params")

    return seeds



def avm_baseline(
    eval_fitness: Callable[[Tuple[int, ...]], float],
    dim: int,
    xmin: int,
    xmax: int,
    restarts: int = 10,
    max_rounds: int = 1000,
    rng_seed: Optional[int] = None,
    init_points: Optional[list] = None,
    eval_limit_per_restart: Optional[int] = None,
):
    assert dim >= 1

    # ---- 내부 하이퍼 ----
    EPS = 1e-9
    PATIENCE = 1
    dom = max(1, xmax - xmin)
    # INIT_STEP = min(16, max(1, dom // 4))  # 상한 16

    INIT_STEP = max(1, min(16, dom // 4))  # <- 0 방지 + 과도한 대스텝 억제

    def better(new, old):
        return new < old

    # ---- 공통 헬퍼 ----
    def init_starts(init_points):
        if rng_seed is not None:
            random.seed(rng_seed)
        starts = list(init_points or [])
        while len(starts) < restarts:
            starts.append(tuple(random.randint(xmin, xmax) for _ in range(dim)))
        return starts[:restarts], None, float("inf")

    def reach_target(val):
        return (val is not None) and (val <= EPS)

    def set_coord(x: Tuple[int, ...], i: int, v: int) -> Tuple[int, ...]:
        if v == x[i]:
            return x
        return tuple(v if j == i else x[j] for j in range(dim))

    # ---- main ----
    starts, best_x, best_f = init_starts(init_points)

    for start in starts:
        calls = 0
        cache: Dict[Tuple[int, ...], float] = {}

        def safe_eval(v: Tuple[int, ...]):
            nonlocal calls
            if eval_limit_per_restart is not None and calls >= eval_limit_per_restart:
                return None
            if v in cache:
                return cache[v]
            calls += 1
            val = eval_fitness(v)
            cache[v] = val
            return val

        x = start
        f = safe_eval(start)
        if f is None:
            continue
        if reach_target(f):
            best_x, best_f = x, f
            break

        rounds = 0
        stall = 0
        last_dir = [0] * dim  # 각 변수 마지막 성공 방향 (-1/0/+1)

        while rounds < max_rounds and not reach_target(f):
            rounds += 1
            round_improved = False

            for i in range(dim):
                step = INIT_STEP

                # 방향 우선 시도
                dir_order = ([last_dir[i]] if last_dir[i] in (-1, +1) else []) + [d for d in (+1, -1) if d != last_dir[i]]

                moved = False
                fn = None
                while step >= 1:
                    for d in dir_order:
                        xn = set_coord(x, i, x[i] + d * step)
                        if xn == x:
                            continue
                        fn = safe_eval(xn)
                        if fn is None:
                            break
                        if better(fn, f):
                            x, f = xn, fn
                            last_dir[i] = d
                            round_improved = True
                            moved = True
                            break
                    # 순서 중요: fn 미정 참조 방지 + 불필요 비교 최소화
                    if moved or reach_target(f) or fn is None:
                        break
                    if step == 1:
                        break
                    step = max(1, step // 2)

                if fn is None or reach_target(f):
                    break

            if fn is None or reach_target(f):
                break

            if round_improved:
                stall = 0
            else:
                stall += 1
                if stall >= PATIENCE:
                    break

        if f is not None and f < best_f:
            best_x, best_f = x, f
        if reach_target(best_f):
            break

    return best_x, best_f


def hill_climb_baseline(
    eval_fitness,
    dim,
    xmin,
    xmax,
    restarts=10,
    max_rounds=1000,
    rng_seed=None,
    init_points=None,
    eval_limit_per_restart=None,
):
    # ---- helpers (전역) ----
    def init_starts(init_points):
        if rng_seed is not None:
            random.seed(rng_seed)
        starts = list(init_points or [])
        debug_print(f"starts {starts}")
        while len(starts) < restarts:
            starts.append(tuple(random.randint(xmin, xmax) for _ in range(dim)))
        return starts[:restarts], None, float("inf")

    def no_more_budget(val):
        return val is None

    def reach_target(val):
        return val == 0.0

    def new_neighbor(x, i, d):
        xn = list(x)
        # xn[i] = clip(xn[i] + d, xmin, xmax)
        xn[i] = xn[i] + d
        xn = tuple(xn)
        return None if xn == x else xn
    
    def not_new_neigbor(xn):
        return xn is None



    # ---- main ----
    starts, best_x, best_f = init_starts(init_points)

    for start in starts:
        # per-restart state
        calls = 0

        def safe_eval(v):
            nonlocal calls
            if eval_limit_per_restart is not None and calls >= eval_limit_per_restart:
                return None
            calls += 1
            return eval_fitness(v)

        x = start
        f = safe_eval(start)
        if no_more_budget(f):
            continue

        for _ in range(max_rounds): # 각 시작 포인트마다 max_rounds만큼의 기회
            if reach_target(f): # 시작 포인트가 바로 답인 경우
                break

            improved = False
            fn = None  # 라운드 시작 시 초기화

            for i in range(dim):
                for d in (-1, 1):
                    xn = new_neighbor(x, i, d)
                    if not_new_neigbor(xn):
                        continue

                    fn = safe_eval(xn)
                    if no_more_budget(fn):
                        break

                    if fn < f:
                        x, f = xn, fn
                        improved = True
                        break  # first-improvement: 개선 방향 발견 시 즉시 이동
                if reach_target(f) or no_more_budget(fn): # 현 라운드에서 답을 발견한 경우, fn = 0.0 이어서 그 밖 루프에서 
                    # 여기에 no_more_budget을 작성하는 이유는 단계적으로 탈출하기 위해서임
                    break 

                # fn = 2, f =0

            if not improved or no_more_budget(fn): # 2 * dim 개수만큼의 주변을 봤는데 improve가 없다면, 다시 좀 더 +- 1로 이동해보기
                # 여기에 no_more_budget을 작성하는 이유는 단계적으로 탈출하기 위해서임
                break

        if f < best_f: 
            best_x, best_f = x, f # 다른 시작점에서의 x, f값과 비교해서 개선하기 위해
        if reach_target(best_f): # 정답을 찾으면, 현재 시작점에서 멈추기
            break

    return best_x, best_f



# =====================
# AL + BD fitness
# ======================
def fitness_AL(func, args_tuple, target_bid: int, want_true: bool, tx: BoolToProbe):
    # 1. Initialize the probe and execute the target function
    __probe.clear(); func(*args_tuple); recs = __probe.records
    # 2. Retrieve the guard chain that must be satisfied to reach the target branch
    guards = tx.if_guards.get(target_bid, [])
    # 3. Check the guards sequentially: find the first unsatisfied guard
    for idx, (gbid, greq) in enumerate(guards):
        r = recs.get(gbid)
        if (r is None) or (not (r["seen_true"] if greq else r["seen_false"])):
            AL = len(guards) - idx
            BD = 1e6 if r is None else (r["d_true"] if greq else r["d_false"])
            return AL + normalise(BD)
    
    # 4. If all guards are satisfied, check the target branch itself, AL = 0
    rt = recs.get(target_bid)
    if rt is None:
        return 1.0 + normalise(1.0)
    if (rt["seen_true"] if want_true else rt["seen_false"]):
        return 0.0
    BD = rt["d_true"] if want_true else rt["d_false"]
    return normalise(BD)


def make_targets_for_func(tx: BoolToProbe, func_name: str, want="both", skip_for_false=True, skip_while_true_false=True):
    """
    Generate targets for a function.
    
    Args:
        tx: BoolToProbe transformer with branch metadata
        func_name: Name of the function
        want: "both" for True/False, or True/False for specific direction
        skip_for_false: If True, skip False direction for for-loops (usually unreachable)
        skip_while_true_false: If True, skip while-True False branches (unreachable)
    """
    targets = []
    for bid in tx.func_to_bids.get(func_name, []):
        # Record the number of guards required to reach this branch as its depth
        depth = len(tx.if_guards.get(bid, []))
        # Set both True and False directions as targets for each branch
        wants = [True, False] if want == "both" else [want]
        
        # Check if this bid is a for-loop or while-True
        is_for_loop = bid in getattr(tx, "loop_minlen", {})
        is_while_true = bid in getattr(tx, "while_always_true", {})
        
        for w in wants:
            # Skip while-True False branches (never exit loop) - unreachable
            if skip_while_true_false and is_while_true and (w is False):
                debug_print(f"Skipping while-True False branch: bid={bid} (unreachable)")
                continue
            
            # Skip for-loop False branches (not entering loop) - usually unreachable
            if skip_for_false and is_for_loop and (w is False):
                debug_print(f"Skipping for-loop False branch: bid={bid} (often unreachable)")
                continue
            
            # for-헤더였고, minlen>=1이 확정이면 False는 UNSAT
            if is_for_loop and tx.loop_minlen[bid] is not None:
                if (tx.loop_minlen[bid] >= 1) and (w is False):
                    continue  # UNSAT -> 스킵
        
            targets.append((depth, bid, w))

    # Sort targets by depth (shallow branches first) to simplify the search
    targets.sort(key = lambda t: t[0])
    debug_print(f"targets {targets}")
    return targets

def _gather_rhs_elems(node):
    seq = node.elts if isinstance(node, (ast.Set, ast.List, ast.Tuple)) else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "set" and node.args and isinstance(node.args[0], (ast.Set, ast.List, ast.Tuple))):
        seq = node.args[0].elts
    if not seq: return []
    vals = []
    for e in seq:
        if isinstance(e, ast.Constant) and isinstance(e.value, int):
            vals.append(int(e.value))
    return vals


def collect_after_fold(tree: ast.AST, target_ftns):
    out = defaultdict(lambda: {"plain": set(), "sets": []})
    target_ftns = set(target_ftns or [])

    class CollectAfterFoldHelper(ast.NodeVisitor):
        def __init__(self): self.fn = None
        def visit_FunctionDef(self, node: ast.FunctionDef):
            if target_ftns and node.name not in target_ftns: return
            prev = self.fn; self.fn = node.name
            out.setdefault(node.name, {"plain":set(), "sets": []})
            for stmt in node.body: self.visit(stmt)
            self.fn = prev
        def visit_Compare(self, node: ast.Compare):
            is_membership_op = isinstance(node.ops[0], (ast.In, ast.NotIn))
            is_single_comparison = len(node.ops) == 1 and len(node.comparators) == 1
            inside_function = bool(self.fn)

            is_func_membership_cmp = inside_function and is_single_comparison and is_membership_op
            if is_func_membership_cmp:
                elems = _gather_rhs_elems(node.comparators[0])
                if elems:
                    out[self.fn]["sets"].append(set(elems))
                self.visit(node.left)
                return
            for ch in ast.iter_child_nodes(node):
                self.visit(ch)
        def visit_Constant(self, node: ast.Constant):
            is_const = isinstance(node.value, int)
            inside_function = bool(self.fn)
            is_func_const = inside_function and is_const
            if is_func_const:
                out[self.fn]["plain"].add(int(node.value))
    CollectAfterFoldHelper().visit(tree)
    return out

def endpoint_distance(x, lo, hi):
    return min(abs(x - lo), abs(x - hi))

def autotune_hparams_for_func(
    tree: ast.AST,
    tx: BoolToProbe,
    ns: Dict[str, Any],
    default = (-10, 10),
):
    target_ftns  = list(tx.func_to_bids.keys())

    func_datas = collect_after_fold(tree, target_ftns)

    hparams_map = {}
    for func_name in target_ftns:
        func_data = func_datas.get(func_name, {"plain": set(), "sets": {}})
        plain = set(func_data.get("plain", set()))
        set_list = list(func_data.get("sets", []))

        if plain:
            tmp_lb, tmp_ub = min(plain), max(plain)
        else:
            tmp_lb, tmp_ub = 0, 0
    
        pool = set()
        for _set in set_list:
            if not _set:
                continue
            sample_value = min(_set, key = lambda x: endpoint_distance(x, tmp_lb, tmp_ub))
            pool.add(sample_value)
        pool |= plain
        xmin = min(pool)
        xmax = max(pool)

        dim = len(inspect.signature(ns[func_name]).parameters)

        num_branches = len(tx.func_to_bids.get(func_name, [])) * 2
        restarts = max(6, num_branches * dim)
        width = max(1, (xmax - xmin))
        max_rounds = max(400, min(4000, width * dim * 8))

        hparams_map[func_name] ={
            "xmin": int(xmin),
            "xmax": int(xmax),
            "restarts": int(restarts),
            "max_rounds": int(max_rounds),
        }
    return hparams_map

def _mark_covered_for_args(covered: set[tuple[int,bool]], func, args, tx, func_name):
    __probe.clear()
    try:
        func(*args)
    except Exception:
        pass  # Execution might fail, but we still want partial coverage info
    
    # Mark all directions that were actually taken in THIS execution
    for b in tx.func_to_bids.get(func_name, []):
        rec = __probe.records.get(b)
        if not rec: 
            continue
        if rec.get("seen_true"):  
            covered.add((b, True))
        if rec.get("seen_false"): 
            covered.add((b, False))



def _hits_with(self_suite, func, bid, want_true):
    for args in self_suite:
        __probe.clear(); func(*args)
        rec = __probe.records.get(bid)
        if rec and ((want_true and rec.get("seen_true")) or
                    (not want_true and rec.get("seen_false"))):
            return True
    return False

def solve_all_branches_for_func(ns, tx, func_name, xmin=-10, xmax=10,
                                restarts=6, base_seed=42, max_rounds=200,
                                verbose=False, algo: str = "avm", compare: bool = False, 
                                eval_limit_per_restart=20000, skip_for_false=True, skip_while_true_false=True):
    """
    Solve all branches for a function using search-based testing.
    
    Args:
        skip_for_false: If True, skip for-loop False branches (often unreachable)
        skip_while_true_false: If True, skip while-True False branches (unreachable)
    """
    covered: set[tuple[int,bool]] = set()
    
    func = ns[func_name]
    param_names = list(inspect.signature(func).parameters.keys())
    dim = len(param_names)
    targets = make_targets_for_func(tx, func_name, want="both", skip_for_false=skip_for_false, 
                                   skip_while_true_false=skip_while_true_false)
    results, suite, seen = [], [], set()

    debug_print(f"[DEBUG] targets {targets}")
    total_targets = len(tx.func_to_bids.get(func_name, [])) * 2

    if not targets:
        default = tuple(0 for _ in range(dim))
        suite.append(default)
        return param_names, suite, results
    
    warm_cache = {}
    for i, (depth, bid, want_true) in enumerate(targets, start=1):
        eval_fit = (lambda b, w: (lambda v: fitness_AL(func, v, b, w, tx)))(bid, want_true)

        debug_print(f"[Target {i}/{len(targets)}] bid={bid}, want_true={want_true}, depth={depth}")
        debug_print(f"  Covered so far: {covered}")
        
        if (bid, want_true) in covered:
            debug_print(f"  -> SKIP: already in covered set")
            continue
        
        if _hits_with(suite, func, bid, want_true):
            debug_print(f"  -> SKIP: already hit by existing test in suite")
            covered.add((bid, want_true))
            continue


        guided = seed_candidates_for_target(param_names)

        key = (tuple(tx.if_guards.get(bid, [])), want_true)
        if key in warm_cache:
            guided = [warm_cache[key]] + guided

        runs = []
        if compare:
            for solver_name in ("avm", "hc"):
                sol, f, m = _run_solver(
                    solver_name,
                    eval_fit, dim, xmin, xmax,
                    restarts, max_rounds,
                    rng_seed=base_seed + i, verbose=verbose, init_points=guided,
                    eval_limit_per_restart=eval_limit_per_restart,
                )
                runs.append((solver_name, sol, f, m))
        else:
            solver_name = "avm" if algo == "avm" else "hc"
            sol, f, m = _run_solver(
                solver_name,
                eval_fit, dim, xmin, xmax,
                restarts, max_rounds,
                rng_seed=base_seed + i, verbose=verbose, init_points=guided,
                eval_limit_per_restart=eval_limit_per_restart,
            )
            runs.append((solver_name, sol, f, m))


        # --- per_solver 구축(오직 compare=True일 때) ---
        per_solver_records = None
        if compare:
            per_solver_records = []
            for (name, solx, fx, mx) in runs:
                __probe.clear(); func(*solx)
                rec_x = __probe.records.get(bid)
                hit_x = None
                if rec_x is not None:
                    hit_x = rec_x.get("seen_true") if want_true else rec_x.get("seen_false")
                ok_x = (fx == 0.0 and bool(hit_x))
                per_solver_records.append({
                    "solver": name,
                    "solution": solx,
                    "fitness": fx,
                    "hit": (bool(hit_x) if hit_x is not None else None),
                    "ok": ok_x,
                    "evals": mx["evals"],
                    "time_sec": mx["time_sec"],
                })

        def _score(item):
            _name, _sol, _f, _m = item
            return (_f, _m["time_sec"])
        solver_name_best, sol_best, f_best, met_best = min(runs, key=_score)


        __probe.clear(); func(*sol_best)
        rec = __probe.records.get(bid)
        hit = None
        if rec is not None:
            hit = rec.get("seen_true") if want_true else rec.get("seen_false")
        if compare:
            debug_print(f"[compare] target#{i} bid={bid} want_true={want_true}")
            for name, solx, fx, mx in runs:
                debug_print(f"  - {name:2}  f={fx:.7g}  evals={mx['evals']}  time={mx['time_sec']:.4f}s  sol={solx}")
            debug_print(f"  => chosen: {solver_name_best} (f={f_best:.7g}, time={met_best['time_sec']:.4f}s)")
        else:
            debug_print(f"sol {i} ({solver_name_best}): {sol_best} | fitness: {f_best} | evals={met_best['evals']} | time={met_best['time_sec']:.4f}s")
        


        # === Adopt results/accumulate test cases ===
        ok = (f_best == 0.0 and bool(hit))  # Considered successful only when fitness is 0 and the actual branch direction is taken
        debug_print(f"  Result: ok={ok}, f={f_best}, hit={hit}, sol={sol_best}")
        
        if ok:
            warm_cache[key] = sol_best       # Reuse for the same chain/direction
        if ok and sol_best not in seen:      # Remove duplicate inputs
            seen.add(sol_best); suite.append(sol_best)
            before_covered = len(covered)
            _mark_covered_for_args(covered, func, sol_best, tx, func_name)
            debug_print(f"  Added to suite. Coverage: {before_covered} -> {len(covered)} branches")
        elif ok:
            debug_print(f"  Solution is duplicate, not adding to suite")
        else:
            debug_print(f"  Solution FAILED: fitness={f_best}, hit={hit}")

        outcome_last = (rec.get("outcome") if rec is not None else None)
        result_entry = {
            "bid": bid, "want_true": want_true, "solution": sol_best, "fitness": f_best,
            "hit": bool(hit) if hit is not None else None,
            "outcome_last": outcome_last,
            "d_true": rec.get("d_true") if rec else None,
            "d_false": rec.get("d_false") if rec else None,
            "ok": ok,
            "solver": solver_name_best,
            "evals": met_best["evals"],
            "time_sec": met_best["time_sec"],
        }
        if compare:
            result_entry["per_solver"] = per_solver_records  # ★ summarize용
        results.append(result_entry)

        if len(covered) >= total_targets:
            debug_print(f"[COMPLETE] All {total_targets} targets covered!")
            break
    
    debug_print(f"\n[SUMMARY] Function '{func_name}':")
    debug_print(f"  Total targets: {total_targets}")
    debug_print(f"  Covered: {len(covered)}/{total_targets}")
    debug_print(f"  Covered branches: {sorted(covered)}")
    debug_print(f"  Test suite size: {len(suite)}")
    
    return param_names, suite, results


def _wrap_eval_with_counter(eval_fitness):
    cnt = {"n": 0}
    def wrapped(x):
        cnt["n"] += 1
        return eval_fitness(x)
    return wrapped, cnt

def _run_solver(
    solver_name: str,
    eval_fitness,
    dim, xmin, xmax,
    restarts, max_rounds,
    rng_seed, verbose, init_points,
    eval_limit_per_restart=None,
):
    wrapped_eval, counter = _wrap_eval_with_counter(eval_fitness)
    t0 = time.perf_counter()
    if solver_name == "avm":
        sol, f = avm_baseline(
            wrapped_eval, dim=dim, xmin=xmin, xmax=xmax,
            restarts=restarts, max_rounds=max_rounds,
            rng_seed=rng_seed, init_points=init_points,
            eval_limit_per_restart=eval_limit_per_restart,
        )    
    else:
        sol, f = hill_climb_baseline(
            wrapped_eval, dim=dim, xmin=xmin, xmax=xmax,
            restarts=restarts, max_rounds=max_rounds,
            rng_seed=rng_seed, init_points=init_points,
            eval_limit_per_restart=eval_limit_per_restart,
        )
    t1 = time.perf_counter()
    metrics = {"solver": hill_climb_baseline, "time_sec": t1 - t0, "evals": counter["n"]}
    return sol, f, metrics


def emit_minimal_call_file(module_name: str, func_to_cases: dict, out_path: str,
                           test_func_prefix: str = "test_"):
    lines = [f"import {module_name}", ""]
    for fname, suite in func_to_cases.items():
        lines.append(f"def {test_func_prefix}{fname}():")
        if not suite:
            lines.append("    pass")
        else:
            for args in suite:
                if not isinstance(args, (tuple, list)): args = (args,)
                call = f"{module_name}.{fname}(" + ", ".join(repr(v) for v in args) + ")"
                lines.append(f"    {call}")
        lines.append("")
    code = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)
    debug_print(f"[gen] wrote: {out_path}")

def detailed_results(results):
    lines = []
    for rec in results:
        bid = rec["bid"]
        want_true = rec["want_true"]
        lines.append(f"Target: bid={bid}, want_true={want_true}")
        per = rec.get("per_solver", [])
        for item in per:
            name = item["solver"]
            ok = "OK" if item["ok"] else "FAIL"
            f = item["fitness"]
            ev = item["evals"]
            t = item["time_sec"]
            sol = item["solution"]
            lines.append(f"  [{name}] {ok} f={f:.6g}, evals={ev}, time={t:.4g}s, sol={sol}")
        lines.append("")
    return "\n".join(lines)


# =====================
# Compatibility API (matching sbst_core.py)
# =====================

class FunctionInfo:
    """Compatibility wrapper for function metadata"""
    def __init__(self, name: str, args: list, node: ast.FunctionDef, tx: BoolToProbe, tree: ast.AST):
        self.name = name
        self.args = args
        self.args_dim = len(args)
        self.node = node
        
        # Extract constants using the original tree
        func_data = collect_after_fold(tree, [name]).get(name, {"plain": set(), "sets": []})
        plain = set(func_data.get("plain", set()))
        set_list = list(func_data.get("sets", []))
        
        pool = plain.copy()
        if plain:
            tmp_lb, tmp_ub = min(plain), max(plain)
        else:
            tmp_lb, tmp_ub = 0, 0
        
        for _set in set_list:
            if not _set:
                continue
            sample_value = min(_set, key=lambda x: endpoint_distance(x, tmp_lb, tmp_ub))
            pool.add(sample_value)
        
        if pool:
            self.min_const = min(pool) - 10
            self.max_const = max(pool) + 10
        else:
            self.min_const = -300
            self.max_const = 300
        
        self.var_constants = {}
        self.total_constants = pool

    def __repr__(self):
        return f"Function {self.name} with {self.args_dim} arg(s): {self.args}"


class BranchInfo:
    """Compatibility wrapper for branch metadata"""
    def __init__(self, bid: int, tx: BoolToProbe):
        self.bid = bid
        self.tx = tx
        # For backwards compatibility: .node returns self so old notebooks work
        self.node = self  # Returns BranchInfo itself
        self.subject = None
        self.match_lineno = None
    
    def __repr__(self):
        func = self.tx.bid_to_func.get(self.bid, "<unknown>")
        return f"Branch bid={self.bid} in func={func}"


class Traveler:
    """Compatibility wrapper that mimics the Traveler class from sbst_core"""
    def __init__(self, tx: BoolToProbe, ns: dict, tree: ast.AST):
        self.tx = tx
        self.ns = ns
        self.tree = tree
        self.branches: dict[str, dict[int, BranchInfo]] = {}
        self.functions: list[FunctionInfo] = []
        self.parent_map = {}  # bid -> parent_bid mapping
        
        # Build functions list
        for func_name in tx.func_to_bids.keys():
            if func_name in ns and callable(ns[func_name]):
                func_obj = ns[func_name]
                param_names = list(inspect.signature(func_obj).parameters.keys())
                # Find the FunctionDef node
                func_node = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func_name:
                        func_node = node
                        break
                if func_node:
                    self.functions.append(FunctionInfo(func_name, param_names, func_node, tx, tree))
        
        # Build branches dict and parent_map
        for func_name, bids in tx.func_to_bids.items():
            self.branches[func_name] = {}
            for bid in bids:
                self.branches[func_name][bid] = BranchInfo(bid, tx)
                # Build parent map from guards
                guards = tx.if_guards.get(bid, [])
                if guards:
                    # The last guard is the immediate parent
                    self.parent_map[bid] = guards[-1][0]


class Record:
    """Compatibility wrapper that mimics the Record class from sbst_core"""
    def __init__(self):
        self.records = {}
        self.trace = []
        # Access the global __probe variable (avoiding name mangling by using globals())
        self._probe = globals()['_' + '_probe']
    
    def write(self, bid, vars_dict):
        """Record branch execution (compatibility method)"""
        self.records[bid] = vars_dict
        if bid not in self.trace:
            self.trace.append(bid)
    
    def get_records(self, bid):
        return self.records.get(bid)
    
    def get_trace(self):
        return list(self.trace)
    
    def clear(self):
        self.records.clear()
        self.trace.clear()
        self._probe.clear()


class FitnessCalculator:
    """Compatibility wrapper for fitness calculation"""
    def __init__(self, traveler: Traveler, record: Record, namespace: dict):
        self.traveler = traveler
        self._record = record
        self.namespace = namespace
        self.evals = 0
        self.tx = traveler.tx
    
    def calculate_fitness(self, target_branch_bid, trace, log, parent_map, target_outcome, subject_node=None):
        """Calculate fitness using the new AL+BD approach"""
        self.evals += 1
        
        # Get the function for this branch
        func_name = self.tx.bid_to_func.get(target_branch_bid, None)
        if not func_name or func_name not in self.namespace:
            return float('inf')
        
        # Since we already executed, use the probe records
        recs = self._record._probe.records
        guards = self.tx.if_guards.get(target_branch_bid, [])
        
        # Check guards
        for idx, (gbid, greq) in enumerate(guards):
            r = recs.get(gbid)
            if (r is None) or (not (r["seen_true"] if greq else r["seen_false"])):
                AL = len(guards) - idx
                BD = 1e6 if r is None else (r["d_true"] if greq else r["d_false"])
                return AL + normalise(BD)
        
        # Check target
        rt = recs.get(target_branch_bid)
        if rt is None:
            return 1.0 + normalise(1.0)
        if (rt["seen_true"] if target_outcome else rt["seen_false"]):
            return 0.0
        BD = rt["d_true"] if target_outcome else rt["d_false"]
        return normalise(BD)
    
    def fitness_for_candidate(self, func, candidate_args, target_branch_node, target_outcome, 
                             subject_node=None, parent_map=None):
        """Run func with candidate_args and return fitness"""
        # Extract target bid from various possible input types
        if isinstance(target_branch_node, BranchInfo):
            target_bid = target_branch_node.bid
        elif isinstance(target_branch_node, int):
            target_bid = target_branch_node
        elif hasattr(target_branch_node, 'lineno'):  # Old-style AST node
            # Find the bid for this AST node by matching line numbers
            # This is a fallback for compatibility
            func_name = func.__name__ if hasattr(func, '__name__') else None
            if func_name and func_name in self.traveler.branches:
                for bid, branch_info in self.traveler.branches[func_name].items():
                    # Use the first branch as a fallback
                    target_bid = bid
                    break
                else:
                    return float('inf')
            else:
                return float('inf')
        else:
            return float('inf')
        
        self._record.clear()
        try:
            func(*candidate_args)
        except Exception as e:
            # Even if function fails, we might have partial branch info
            if VERBOSE:
                debug_print(f"[fitness] Exception during execution: {e}")
        
        trace = self._record.get_trace()
        log = self._record.records
        return self.calculate_fitness(target_bid, trace, log, parent_map or {}, target_outcome, subject_node)


def instrument_and_load_compatible(source_code: str):
    """
    Alias for instrument_and_load() for backward compatibility.
    Returns: (namespace, traveler, record, instrumented_tree)
    """
    return instrument_and_load(source_code)


def hill_climbing_search(func, initial_args, target_branch_node, target_outcome, 
                        fitness_calc: FitnessCalculator, actual_indices, parent_map,
                        subject_node=None, match_lineno=None, max_iters=1000):
    """
    Compatibility wrapper for hill climbing search.
    Uses the advanced AVM/HC algorithms under the hood.
    """
    # Extract target bid
    if isinstance(target_branch_node, BranchInfo):
        target_bid = target_branch_node.bid
    elif isinstance(target_branch_node, int):
        target_bid = target_branch_node
    else:
        return None
    
    dim = len(actual_indices)
    
    # Create fitness function
    def eval_fit(args_tuple):
        return fitness_calc.fitness_for_candidate(func, args_tuple, target_bid, target_outcome, 
                                                 subject_node, parent_map)
    
    # Determine search bounds from function info
    func_name = fitness_calc.tx.bid_to_func.get(target_bid)
    func_info = None
    for fi in fitness_calc.traveler.functions:
        if fi.name == func_name:
            func_info = fi
            break
    
    if func_info:
        xmin, xmax = func_info.min_const, func_info.max_const
    else:
        xmin, xmax = -10, 10
    
    # Use AVM (more robust than basic hill climbing)
    sol, f = avm_baseline(
        eval_fit,
        dim=dim,
        xmin=xmin,
        xmax=xmax,
        restarts=6,
        max_rounds=max_iters,
        rng_seed=42,
        init_points=[tuple(initial_args)],
        eval_limit_per_restart=None
    )
    
    if f == 0.0:
        return list(sol)
    return None


# =====================
# End Compatibility API
# =====================

# usage: python sbst.py examples/example$N$.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="the target python file to generate unit tests for")
    parser.add_argument("--xmin", type=int, default=None)
    parser.add_argument("--xmax", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--eval-limit-per-restart", type=int, default=10000)
    parser.add_argument("--algo", choices=["avm", "hc"], default="avm")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--test-for-false", action="store_true", 
                       help="Test for-loop False branches (usually unreachable, slow)")
    parser.add_argument("--test-while-true-false", action="store_true",
                       help="Test while-True False branches (unreachable)")

    args = parser.parse_args()

    with open(args.target, "r") as f:
        code = f.read()
    
    tree = ast.parse(code)
    # debug_print(ast.dump(tree)) #DEBUG

    """
    TODO: generate a test suite for the target python file
    You can modify the code below to generate a test suite for any target python file in the examples folder.
    """

    ns, tx, inst, tree = _instrument_and_load_internal(code)
    debug_print("[DEBUG] Branch IDs:")
    for bid in tx.if_root_bids:
        func = tx.bid_to_func.get(bid, "<unknown>")
        guards = tx.if_guards.get(bid, [])
        debug_print(f"  bid={bid}, func={func}, guards={guards}")

    target_module = os.path.basename(args.target).removesuffix(".py")
    target_ftns = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    # debug_print(f"[DEBUG] functions in {target_module}:", [n.name for n in target_ftns])
    debug_print(f"[DEBUG] functions in {target_module}: { [n.name for n in target_ftns] }")




    
    funcs_hparams = autotune_hparams_for_func(tree, tx, ns)
    debug_print("[SET] autotuned hparams")
    for fn, func_hparams in funcs_hparams.items():
        debug_print(f"target func name: {fn}, hparams {func_hparams}")


    func_to_cases = {};     funcs_results = []
    for fn_node in target_ftns:
        fname = fn_node.name
        if fname not in ns:
            debug_print(f"[warn] function {fname} not compiled? skip")
            continue
        debug_print(f"[solve] {fname}")

        func_hparams = funcs_hparams.get(fname, {"xmin": -10, "xmax": 10, "restarts": 6, "max_rounds": 1000})
        xmin = args.xmin if args.xmin is not None else func_hparams["xmin"]
        xmax = args.xmax if args.xmax is not None else func_hparams["xmax"]
        restarts = args.restarts if args.restarts is not None else func_hparams["restarts"]
        max_rounds = args.max_rounds if args.max_rounds is not None else func_hparams["max_rounds"]
        eval_limit_per_restart = args.eval_limit_per_restart

        param_names, suite, results = solve_all_branches_for_func(
            ns, tx, fname,
            xmin=xmin, xmax=xmax, restarts=restarts,
            base_seed=args.seed, verbose=False,
            algo=args.algo, compare=args.compare,
            max_rounds=max_rounds,
            eval_limit_per_restart=eval_limit_per_restart,
            skip_for_false=not args.test_for_false,  # By default, skip for-loop False branches
            skip_while_true_false=not args.test_while_true_false,  # By default, skip while-True False
        )
        if args.compare:
            funcs_results.append(results)
        debug_print(f"  -> {len(suite)} inputs (xmin={xmin}, xmax={xmax}, restarts={restarts}, max_rounds={max_rounds})")
        func_to_cases[fname] = suite


    if args.compare:
        debug_print("\n\n============ Summary ============")
        for (i, fn_node) in enumerate(target_ftns):
            debug_print("---------------------------------")
            debug_print(f"[Func] {fn_node.name}")
            debug_print("[DETAIL]\n")
            debug_print(detailed_results(funcs_results[i]))
            debug_print("---------------------------------")

    
    test_file_name = os.path.join(os.path.dirname(args.target), f"test_{target_module}.py")
    emit_minimal_call_file(target_module, func_to_cases, test_file_name)

    debug_print(f"Test suite generated in {test_file_name}")
