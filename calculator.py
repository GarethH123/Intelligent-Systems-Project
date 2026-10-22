"""
Calculates the result of a recognised arithmetic expression such as
"26-3*(17-4)". Only digits and + - * / ( ) are accepted, and the text is
parsed with ast (never eval'd), so a misread symbol can't run anything.

    from calculator import calculate
    calculate("26-3*(17-4)")   # -> -13
    calculate("7/2")           # -> 3.5
    calculate("5/0")           # raises ValueError("Division by zero")
"""
import ast
import operator
import re

_BINARY_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
               ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Not a valid expression")


def calculate(text: str):
    """Return the value of the expression (an int when it's a whole number,
    otherwise a float rounded to 6 places). Raises ValueError with a short
    message if the text isn't a valid expression.
    """
    if not re.fullmatch(r"[0-9+\-*/()]+", text or ""):
        raise ValueError("Not a valid expression")

    # handwriting often leaves the multiplication out: "3(4+1)" means 3*(4+1)
    expr = re.sub(r"(?<=[\d)])(?=\()", "*", text)
    expr = re.sub(r"(?<=\))(?=\d)", "*", expr)
    # Python rejects leading zeros ("007"), but a digit classifier can easily produce them
    expr = re.sub(r"(?<!\d)0+(?=\d)", "", expr)

    try:
        value = _eval(ast.parse(expr, mode="eval").body)
    except (SyntaxError, RecursionError):
        raise ValueError("Not a valid expression") from None
    return int(value) if float(value).is_integer() else round(value, 6)
