import importlib, sys
from abc import ABCMeta, abstractmethod
from typing import Literal
# from wurlitzer import pipes
from .common import log
import threading

class NoSolutionsError(Exception):
    """Raised if a model is infeasible."""
    pass


class SolverMeta(metaclass=ABCMeta):
    @abstractmethod
    def solve(time_limit, threads, log_path) -> None:
        """_summary_

        Args:
            time_limit (int): Max time limit in seconds
            threads (int): Max thread usage
            log_path (str): Path to log file to write
        """
        raise NotImplementedError



class IlpSolverMeta(SolverMeta):
    """Meta class for ILP solvers. Child classes must implement all abstract classes"""
    SOLUTION_PRECISION = 1e-7
    VALID_VARIABLE_TYPES = set(['b', 'c', 'i'])
    

    def add_var(self, name: str, type: Literal['b', 'c', 'i'], lb: int=0, ub=None):
        """Adds variable to solver model. Calls self._add_var

        Args:
            name (str): Name of variable
            type (str): One of ('c', 'b', 'i')
            lb (int, optional): Lower bound of variable. Defaults to 0.
            ub (_type_, optional): Upper bound of variable. Defaults to None which will result in being set to infinity

        Raises:
            ValueError: If type is invalid

        Returns:
            None
        """
        type = type.lower()
        if type not in self.VALID_VARIABLE_TYPES:
            raise ValueError(f'Variable type {type} not one of valid options: {str(self.VALID_VARIABLE_TYPES)}')
        if ub is None:
            ub = self.INF
        return self._add_var(name, type, lb, ub)


    @abstractmethod
    def _add_var(self, name, type, lb, ub):
        '''Private method to actually create the variant - to be implemented in child class'''
        raise NotImplementedError


    @abstractmethod
    def solve(self, method: Literal['minimize', 'maximize'], 
              time_limit: int,
               threads: int, 
               log_path: str) -> None:
        """Solve the model using self.objective as the objective function

        Args:
            method (Literal[&#39;minimize&#39;, &#39;maximize&#39;]): Maximize or minimize the objective functions
            time_limit (int): Time limit in seconds
            threads (int): 
            log_path (str): Path to save solver log if applicable
        """


    @abstractmethod
    def add_constr(self, name: str, *args, **kwargs):
        """Add constraint. Args should include linear expression of variables.

        Args:
            name (str): Name of variable
        """
        raise NotImplementedError


    @abstractmethod
    def quicksum(self, expr: list):
        """Sum list of variables and coefficients and return linear expression.

        Args:
            expr (list): List of variables and coefficiants in form [var, -coef*var2,...]
        """
        raise NotImplementedError


    @staticmethod
    @abstractmethod
    def var_name(var):
        """Return variable name

        Args:
            var: Variable instance
        """
        raise NotImplementedError


    @abstractmethod
    def LinExpr(self, *args, **kwargs):
        """Return a linear expression instance
        """
        raise NotImplementedError


    @abstractmethod
    def get_value(self, var):
        """Return variable value

        Args:
            var: variable instance
        """
        raise NotImplementedError
    





class GurobiSolver(IlpSolverMeta):
    def __init__(self, name):
        self.gurobipy = importlib.import_module('gurobipy')

        self.INF = self.gurobipy.GRB.INFINITY
        self.GUROBI_STATUS = {
            getattr(self.gurobipy.GRB.status, v): v
            for v in dir(self.gurobipy.GRB.status)
            if v[:2] != '__'
        }
        self.model = self.gurobipy.Model(name)
        self.model.reset()

    def add_constr(self, *args, **kwargs):
        return self.model.addConstr(*args, **kwargs)

    def addGenConstrAbs(self, *args, **kwargs):
        return self.model.addGenConstrAbs(*args, **kwargs)

    def _add_var(self, name: str, type: str, lb: int=0, ub=None, *args, **kwargs):    
        if type == "b":
            type = self.gurobipy.GRB.BINARY
        elif type == "i":
            type = self.gurobipy.GRB.INTEGER
        else:
            type = self.gurobipy.GRB.CONTINUOUS
        v = self.model.addVar(*args, vtype=type, lb=lb, ub=ub, name=name, **kwargs)
        return v

    def quicksum(self, expr):
        return self.gurobipy.quicksum(expr)

    def update(self):
        self.model.update()

    @staticmethod
    def var_name(var):
        return var.var_name
    
    def LinExpr(self, *args, **kwargs):
        return self.gurobipy.LinExpr(*args, **kwargs)

    def abssum(self, variables, coeffs=None):
        total = 0
        for i, v in enumerate(variables):
            coeff = 1 if coeffs is None else coeffs[i]
            absvar = self.add_var()
            total += absvar * coeff
            self.add_constr(absvar + v >= 0)
            self.add_constr(absvar - v >= 0)
        return total

    def solve(self, objective=None, method='min', threads=6, time_limit=4000, log_path=None, MIPGap=None): # ret obj val
        if objective is not None:
            self.objective = objective
        self.model.setObjective(
            self.objective,
            self.gurobipy.GRB.MINIMIZE if method == 'min' else self.gurobipy.GRB.MAXIMIZE
        )
        self.model.update()

        if time_limit: self.model.params.timeLimit = time_limit
        self.model.params.threads = threads
        # self.model.params.outputFlag = 1
        self.model.params.LogToConsole = 1
        if not log_path:
            log_path = 'gurobi.log'
        self.model.params.logFile = log_path
        if MIPGap: 
            self.model.params.MIPGap = MIPGap
        else:
            self.model.params.MIPGap = self.SOLUTION_PRECISION
        print('RUNNING MODEL')
        self.model.optimize()

        status = self.GUROBI_STATUS[self.model.status]
        if self.model.status == self.gurobipy.GRB.INFEASIBLE:
            raise NoSolutionsError(status)
        return status.lower(), self.model.objVal

    # def solveAll(self, objective, var, method='min'):
    #     status, opt_value = self.solve(objective, method)
    #     solutions = [frozenset(a for a, y in var.items() if round(self.get_value(y)) > 0)]
    #     solutions += get_all_solutions(
    #         self, var, opt_value,
    #         candidates=solutions[0], iteration=0, mem=list()
    #     )
    #     solutions = [tuple(sorted(y for y in x)) for x in solutions]

    #     solutions = list(set(solutions))
    #     return status, opt_value, solutions

    def get_value(self, var):
        try:
            if var.vtype == self.gurobipy.GRB.BINARY:
                return round(var.getAttr("X")) > 0
            elif var.vtype == self.gurobipy.GRB.INTEGER:
                return int(round(var.getAttr("X")))
            else:
                return var.getAttr("X")
        except AttributeError:
            return var.getValue()

    def changeUb(self, var, ub):
        var.ub = ub
        self.model.update()


class OrToolsSolver(IlpSolverMeta):
    '''Wrapper for Google OR tools CP_SAT'''

    def __init__(self, name: str) -> None:
        sys.argv.extend(["--logtostderr", "--stderrthreshold=INFO", "--v=1"])
        self.ortools = importlib.import_module("ortools.linear_solver.pywraplp")
        self.model = self.ortools.Solver.CreateSolver("CP_SAT")
        self.model.EnableOutput()

        self.INF = self.model.infinity() #TODO
        self.STATUS = {
                self.ortools.Solver.INFEASIBLE: "INFEASIBLE",
                self.ortools.Solver.FEASIBLE: "FEASIBLE",
                self.ortools.Solver.UNBOUNDED: "UNBOUNDED",
                self.ortools.Solver.OPTIMAL: "OPTIMAL",
                self.ortools.Solver.NOT_SOLVED: "NOT_SOLVED",
                self.ortools.Solver.ABNORMAL: "ABNORMAL",
            }
        self.names = {}
    
    
    def add_constr(self, *args, name, **kwargs):
        kwargs['name'] = name
        return self.model.Add(*args, **kwargs)

    def _add_var(self, name: str, type: str, lb: int=0, ub=None, *args, **kwargs):    
        if type == "b":
            v = self.model.BoolVar(name)
        elif type == "i":
            v = self.model.IntVar(lb, ub, name)
        else:
            v = self.model.NumVar(lb, ub, name)
        return v

    def quicksum(self, expr):
        return self.model.Sum(expr)

    def setObjective(self, objective, method: str='min'):
        self.objective = objective
        if method == "min":
            self.model.Minimize(self.objective)
        else:
            self.model.Maximize(self.objective)

    def LinExpr(self, *args, **kwargs):
        return self.ortools.LinearExpr(*args, **kwargs)

    @staticmethod
    def var_name(var):
        return var.var_name
    
    def solve(self, time_limit: int,
            threads: int, 
            log_path: str,
            method: Literal['minimize', 'maximize'] = 'minimize') -> None:
        
        self.model.set_time_limit(int(time_limit*1000))
        self.model.SetNumThreads(threads)

        if method == 'minimize':
            self.model.Minimize(self.objective)
        elif method == 'maximize':
            # If maximize is desired, typically you would use a Maximize method.
            # Adjust as needed.
            self.model.Minimize(self.objective)
        else:
            raise ValueError(f"Invalid method argument {method}")
        
        # captured_stdout = []
        # captured_stderr = []
        # file_lock = threading.Lock()

        # # This helper continuously reads from the given pipe, writes each decoded line
        # # to the log file (using a lock for thread-safety) and appends it to a capture list.
        # def stream_reader(pipe, capture_list):
        #     for line in iter(pipe.readline, b''):
        #         decoded_line = line.decode('utf-8')
        #         capture_list.append(decoded_line)
        #         with file_lock:
        #             log_file.write(decoded_line)
        #             log_file.flush()

        # with pipes() as (out, err):
        #     # Open the log file once and have both threads write to it.
        #     with open(log_path, 'w') as log_file:
        #         stdout_thread = threading.Thread(target=stream_reader, args=(out, captured_stdout))
        #         stderr_thread = threading.Thread(target=stream_reader, args=(err, captured_stderr))
                
        #         stdout_thread.start()
        #         stderr_thread.start()
                
        #         # Execute the solver while output is streamed.
        #         status_type = self.model.Solve() 
                
        #         # Wait for both streams to finish reading.
        #         stdout_thread.join()
        #         stderr_thread.join()
        
        status_type = self.model.Solve() 

        status = self.STATUS[status_type]
        if status == 'INFEASIBLE':
            raise NoSolutionsError(status)
        if not self.model.VerifySolution(self.SOLUTION_PRECISION, True):
            raise NoSolutionsError(status)

        # # After the solver finishes, log the complete captured output.
        # log.debug(f"Captured OR-Tools stdout: {''.join(captured_stdout)}")
        # log.debug(f"Captured OR-Tools stderr: {''.join(captured_stderr)}")
        log.debug(f"OR-Tools status: {status}")

        return status, self.model.Objective().Value()

    def get_value(self, var):
        x = var.solution_value()
        if hasattr(var, "integer") and var.integer():
            x = int(round(x))
            if (
                abs(var.lb()) < self.SOLUTION_PRECISION
                and abs(1 - var.ub()) < self.SOLUTION_PRECISION
            ):
                return x > 0
            else:
                return x
        else:
            return x


class _BatchVar:
    """Lightweight variable proxy for deferred batch model construction.

    Supports arithmetic operators so _build() can compose expressions
    without touching HiGHS until solve() materializes the full model.
    """
    __slots__ = ('index', 'name')

    def __init__(self, index, name):
        self.index = index
        self.name = name

    # --- arithmetic ---
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return _BatchLinExpr({}, 0.0)
            return _BatchLinExpr({self.index: float(other)}, 0.0)
        if isinstance(other, _BatchVar):
            raise TypeError("Nonlinear: var * var")
        if isinstance(other, _BatchLinExpr):
            raise TypeError("Nonlinear: var * expr")
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return _BatchLinExpr({self.index: 1.0}, float(other))
        if isinstance(other, _BatchVar):
            coeffs = {self.index: 1.0}
            coeffs[other.index] = coeffs.get(other.index, 0.0) + 1.0
            return _BatchLinExpr(coeffs, 0.0)
        if isinstance(other, _BatchLinExpr):
            coeffs = dict(other.coeffs)
            coeffs[self.index] = coeffs.get(self.index, 0.0) + 1.0
            return _BatchLinExpr(coeffs, other.constant)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return _BatchLinExpr({self.index: -1.0}, 0.0)

    def __sub__(self, other):
        return self + (-other if not isinstance(other, (int, float)) else -other)

    def __rsub__(self, other):
        return (-self) + other

    # --- comparisons → constraints ---
    def __ge__(self, other):
        return (self - other)._as_ge()

    def __le__(self, other):
        return (self - other)._as_le()

    def __eq__(self, other):
        return (self - other)._as_eq()

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"_BatchVar({self.index}, {self.name!r})"


class _BatchLinExpr:
    """Sparse linear expression: sum(coeffs[i]*x_i) + constant.

    Uses lazy composition for O(1) addition: accumulates sub-expressions
    as a list of parts, flattened into a single dict on first access.
    This avoids O(n²) dict-copying when building large objectives.
    """
    __slots__ = ('_flat', '_parts', 'constant')

    def __init__(self, coeffs, constant):
        # coeffs is always a dict for newly created expressions
        self._flat = coeffs        # dict or None (if lazy)
        self._parts = None         # list of dicts (lazy sub-expressions)
        self.constant = constant   # float

    @property
    def coeffs(self):
        """Lazily flatten parts into a single dict."""
        if self._flat is None:
            merged = {}
            for part in self._parts:
                for idx, c in part.items():
                    merged[idx] = merged.get(idx, 0.0) + c
            self._flat = merged
            self._parts = None
        return self._flat

    def _add_lazy(self, other_coeffs, other_constant):
        """O(1) lazy addition — defers flattening."""
        result = _BatchLinExpr.__new__(_BatchLinExpr)
        result.constant = self.constant + other_constant
        result._flat = None
        # Collect parts: if self is already lazy, extend; otherwise start fresh
        if self._parts is not None:
            result._parts = self._parts  # share the list (self is being replaced)
            result._parts.append(other_coeffs)
        elif self._flat is not None:
            result._parts = [self._flat, other_coeffs]
        else:
            result._parts = [other_coeffs]
        return result

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self._parts is not None:
                result = _BatchLinExpr.__new__(_BatchLinExpr)
                result._flat = None
                result._parts = self._parts
                result.constant = self.constant + other
                return result
            return _BatchLinExpr(dict(self._flat), self.constant + other)
        if isinstance(other, _BatchVar):
            return self._add_lazy({other.index: 1.0}, 0.0)
        if isinstance(other, _BatchLinExpr):
            return self._add_lazy(other.coeffs, other.constant)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return _BatchLinExpr({}, 0.0)
            return _BatchLinExpr(
                {idx: c * other for idx, c in self.coeffs.items()},
                self.constant * other,
            )
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return _BatchLinExpr(
            {idx: -c for idx, c in self.coeffs.items()},
            -self.constant,
        )

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self + (-other)
        if isinstance(other, (_BatchVar, _BatchLinExpr)):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other):
        return (-self) + other

    # --- comparisons → constraints ---
    def _as_ge(self):
        """self >= 0  →  _BatchConstraint"""
        return _BatchConstraint(dict(self.coeffs), -self.constant, None)

    def _as_le(self):
        """self <= 0  →  _BatchConstraint"""
        return _BatchConstraint(dict(self.coeffs), None, -self.constant)

    def _as_eq(self):
        """self == 0  →  _BatchConstraint"""
        return _BatchConstraint(dict(self.coeffs), -self.constant, -self.constant)

    def __ge__(self, other):
        return (self - other)._as_ge()

    def __le__(self, other):
        return (self - other)._as_le()

    def __eq__(self, other):
        return (self - other)._as_eq()

    def __hash__(self):
        return id(self)

    def __repr__(self):
        terms = [f"{c}*x{i}" for i, c in self.coeffs.items()]
        return f"_BatchLinExpr({' + '.join(terms)} + {self.constant})"


class _BatchConstraint:
    """Deferred constraint: row_lower <= sum(coeffs[i]*x_i) <= row_upper."""
    __slots__ = ('coeffs', 'lb', 'ub')

    def __init__(self, coeffs, lb, ub):
        self.coeffs = coeffs  # dict: var_index -> float
        self.lb = lb          # float or None (no lower bound)
        self.ub = ub          # float or None (no upper bound)


class HiGHSSolver(IlpSolverMeta):
    """Wrapper for HiGHS MIP solver via highspy.

    Uses deferred batch construction: _add_var() and add_constr() record
    lightweight proxy objects. solve() materializes the full LP as a
    HighsLp and calls passModel() once — O(n) instead of O(n²).
    """

    def __init__(self, name: str) -> None:
        self.highspy = importlib.import_module("highspy")
        self.h = self.highspy.Highs()
        self.h.silent()
        self.INF = self.highspy.kHighsInf

        # Deferred variable specs: list of (name, type, lb, ub)
        self._var_specs = []
        self._var_count = 0

        # Deferred constraints: list of _BatchConstraint
        self._constraints = []

        # Solution array, populated after solve
        self._sol_values = None

    def _add_var(self, name: str, type: str, lb: int = 0, ub=None, *args, **kwargs):
        idx = self._var_count
        if type == "b":
            lb, ub = 0, 1
        self._var_specs.append((name, type, lb, ub))
        self._var_count += 1
        if self._var_count % 50000 == 0:
            log.info(f"HiGHS model build: {self._var_count} variables registered")
        return _BatchVar(idx, name)

    def add_constr(self, *args, name=None, **kwargs):
        if args and isinstance(args[0], _BatchConstraint):
            self._constraints.append(args[0])

    def quicksum(self, terms):
        """O(n) summation — builds flat dict directly, avoids chain-of-adds."""
        coeffs = {}
        constant = 0.0
        for term in terms:
            if isinstance(term, _BatchVar):
                coeffs[term.index] = coeffs.get(term.index, 0.0) + 1.0
            elif isinstance(term, _BatchLinExpr):
                for idx, c in term.coeffs.items():
                    coeffs[idx] = coeffs.get(idx, 0.0) + c
                constant += term.constant
            elif isinstance(term, (int, float)):
                constant += term
            else:
                raise TypeError(f"quicksum: unsupported term type {type(term)}")
        if not coeffs and constant == 0.0:
            return 0
        return _BatchLinExpr(coeffs, constant)

    def LinExpr(self, *args, **kwargs):
        return 0

    def _materialize(self, objective, sense):
        """Build the full LP from deferred specs and pass to HiGHS in one shot."""
        import numpy as np

        num_col = self._var_count
        num_row = len(self._constraints)

        log.info(f"HiGHS materializing model: {num_col} variables, {num_row} constraints")

        # --- columns (variables) ---
        col_cost = np.zeros(num_col, dtype=np.float64)
        col_lower = np.empty(num_col, dtype=np.float64)
        col_upper = np.empty(num_col, dtype=np.float64)
        integrality = np.empty(num_col, dtype=np.int32)

        INT_TYPE = self.highspy.HighsVarType.kInteger
        CONT_TYPE = self.highspy.HighsVarType.kContinuous
        integrality_list = []

        for i, (name, vtype, lb, ub) in enumerate(self._var_specs):
            col_lower[i] = lb
            col_upper[i] = ub
            integrality_list.append(INT_TYPE if vtype in ('b', 'i') else CONT_TYPE)

        # Extract objective coefficients
        if isinstance(objective, _BatchLinExpr):
            for idx, coeff in objective.coeffs.items():
                col_cost[idx] = coeff
            obj_offset = objective.constant
        elif isinstance(objective, _BatchVar):
            col_cost[objective.index] = 1.0
            obj_offset = 0.0
        elif isinstance(objective, (int, float)):
            obj_offset = float(objective)
        else:
            raise TypeError(f"Unsupported objective type: {type(objective)}")

        # --- rows (constraints) in CSR format ---
        # Pre-count total nonzeros
        total_nz = sum(len(c.coeffs) for c in self._constraints)
        row_lower = np.empty(num_row, dtype=np.float64)
        row_upper = np.empty(num_row, dtype=np.float64)

        # Column-wise CSC: starts[col] = offset into index/value arrays
        # But HighsLp supports both colwise and rowwise. Use rowwise (CSR).
        a_start = np.empty(num_row + 1, dtype=np.int32)
        a_index = np.empty(total_nz, dtype=np.int32)
        a_value = np.empty(total_nz, dtype=np.float64)

        nz_pos = 0
        for r, constr in enumerate(self._constraints):
            a_start[r] = nz_pos
            row_lower[r] = constr.lb if constr.lb is not None else -self.INF
            row_upper[r] = constr.ub if constr.ub is not None else self.INF
            for col_idx, coeff in constr.coeffs.items():
                a_index[nz_pos] = col_idx
                a_value[nz_pos] = coeff
                nz_pos += 1
        a_start[num_row] = nz_pos

        # Build HighsLp
        lp = self.highspy.HighsLp()
        lp.num_col_ = num_col
        lp.num_row_ = num_row
        lp.col_cost_ = col_cost
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        lp.a_matrix_.start_ = a_start
        lp.a_matrix_.index_ = a_index
        lp.a_matrix_.value_ = a_value
        lp.a_matrix_.format_ = self.highspy.MatrixFormat.kRowwise
        lp.integrality_ = integrality_list
        lp.sense_ = (self.highspy.ObjSense.kMinimize if sense == 'min'
                      else self.highspy.ObjSense.kMaximize)
        lp.offset_ = obj_offset

        status = self.h.passModel(lp)
        if status != self.highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS passModel failed: {status}")

        log.info(f"HiGHS model materialized: {num_col} cols, {num_row} rows, {total_nz} nonzeros")

    def solve(self, objective=None, method='min', threads=6, time_limit=4000, log_path=None, MIPGap=None, highs_params=None, warm_start_values=None):
        if objective is not None:
            self.objective = objective

        # Materialize the deferred model into HiGHS in one shot
        self._materialize(self.objective, method)

        # Configure solver parameters
        self.h.setOptionValue("time_limit", float(time_limit))
        self.h.setOptionValue("threads", threads)

        # Enable solver progress output
        self.h.setOptionValue("output_flag", True)
        self.h.setOptionValue("log_to_console", True)
        self.h.setOptionValue("mip_report_level", 2)
        self.h.setOptionValue("mip_min_logging_interval", 5.0)
        if log_path and log_path != "/dev/null":
            self.h.setOptionValue("log_file", log_path)

        if MIPGap:
            self.h.setOptionValue("mip_rel_gap", MIPGap)
        else:
            self.h.setOptionValue("mip_rel_gap", self.SOLUTION_PRECISION)

        # Apply custom HiGHS parameters (for solver tuning experiments)
        if highs_params:
            for key, value in highs_params.items():
                log.info(f"HiGHS custom param: {key}={value}")
                self.h.setOptionValue(key, value)

        # Apply warm start if provided
        if warm_start_values:
            log.info(f"Setting warm start ({len(warm_start_values)} values)")
            self.h.setSolution(warm_start_values, [])

        log.info(f"RUNNING MODEL (HiGHS) — time_limit={time_limit}s, mip_rel_gap={MIPGap or self.SOLUTION_PRECISION}, threads={threads}")
        self.h.run()

        model_status = self.h.getModelStatus()
        STATUS_MAP = {
            self.highspy.HighsModelStatus.kOptimal: "optimal",
            self.highspy.HighsModelStatus.kInfeasible: "infeasible",
            self.highspy.HighsModelStatus.kUnbounded: "unbounded",
            self.highspy.HighsModelStatus.kUnboundedOrInfeasible: "infeasible",
            self.highspy.HighsModelStatus.kObjectiveBound: "objective_bound",
            self.highspy.HighsModelStatus.kObjectiveTarget: "objective_target",
            self.highspy.HighsModelStatus.kSolutionLimit: "solution_limit",
            self.highspy.HighsModelStatus.kTimeLimit: "time_limit",
            self.highspy.HighsModelStatus.kIterationLimit: "iteration_limit",
        }
        status = STATUS_MAP.get(model_status, f"unknown_{model_status}")

        if status == "infeasible":
            raise NoSolutionsError(f"HiGHS status: {status}")

        # Cache solution values for get_value()
        sol = self.h.getSolution()
        self._sol_values = list(sol.col_value)

        _, obj_val = self.h.getInfoValue("objective_function_value")
        return status, obj_val

    def get_value(self, var):
        if self._sol_values is None:
            raise RuntimeError("Model has not been solved yet")

        if isinstance(var, _BatchLinExpr):
            val = var.constant
            for idx, coeff in var.coeffs.items():
                val += coeff * self._sol_values[idx]
            rounded = round(val)
            if abs(val - rounded) < self.SOLUTION_PRECISION:
                return int(rounded)
            return val

        if isinstance(var, _BatchVar):
            val = self._sol_values[var.index]
            vtype = self._var_specs[var.index][1]
            if vtype == 'b':
                return round(val) > 0
            elif vtype == 'i':
                return int(round(val))
            else:
                return val

        if isinstance(var, (int, float)):
            return var

        raise TypeError(f"Unsupported var type for get_value: {type(var)}")

    @staticmethod
    def var_name(var):
        return var.name

    def update(self):
        pass

    def changeUb(self, var, ub):
        name, vtype, lb, _old_ub = self._var_specs[var.index]
        self._var_specs[var.index] = (name, vtype, lb, ub)
