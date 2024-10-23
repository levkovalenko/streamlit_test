from fractions import Fraction

import numpy as np
import streamlit as st
import sympy as sym
from sympy import printing

st.title("Проверка контрольной")
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    st.write("##")
    st.latex("""
\\begin{align*}
A = 
\\end{align*} """)
    st.write("##")

with col2:
    a1 = st.number_input("$a_1$", value=0)
    a4 = st.number_input("$a_4$", value=0)
    a7 = st.number_input("$a_7$", value=-2)


with col3:
    a2 = st.number_input("$a_2$", value=1)
    a5 = st.number_input("$a_5$", value=0)
    a8 = st.number_input("$a_8$", value=1)


with col4:
    a3 = st.number_input("$a_3$", value=0)
    a6 = st.number_input("$a_6$", value=1)
    a9 = st.number_input("$a_9$", value=1)


col11, col21, col31, col41 = st.columns([1, 1, 1, 1])

with col11:
    st.latex("""
\\begin{align*}
C = 
\\end{align*} """)

with col21:
    c1 = st.number_input("$c_1$", value=-1)

with col31:
    c2 = st.number_input("$c_2$", value=-1)

with col41:
    c3 = st.number_input("$c_3$", value=0)

st.write("##")


s1 = st.number_input("$s_1$", value=-1)
s2 = st.number_input("$s_2$", value=-1)
s3 = st.number_input("$s_3$", value=-2)
y0 = -s1 * s2 * s3
y1 = s2 * s3 + s1 * s3 + s1 * s2
y2 = -s3 - s2 - s1


A = np.array([[a1, a2, a3], [a4, a5, a6], [a7, a8, a9]])
C = np.array([c1, c2, c3])

N = np.vstack([C, C @ A, C @ A @ A])
_, col_L, col_R = st.columns([1, 1, 2])
with col_L:
    st.latex("""
\\begin{align*}
             \\
N = 
\\end{align*} """)

with col_R:
    st.write(N)

st.latex(f"""
\\begin{{align*}}
rank(N) = {np.linalg.matrix_rank(N)}
\\end{{align*}} """)


sym.init_printing(use_unicode=True)


s, l1, l2, l3 = sym.symbols("s,l_1,l_2,l_3")
symS = sym.diag(s, s, s)
symA = sym.Matrix(A)
symC = sym.Matrix([C])
symL = sym.Matrix([[l1], [l2], [l3]])
sym_sIACL = symS - symA + symL @ symC

st.latex(f"""
\\begin{{align*}}
det(sI - A +LC) = {printing.latex(sym_sIACL)}
\\end{{align*}} """)
sym_Det = sym_sIACL.det()
str_Det = printing.latex(sym_Det)
st.markdown(f"""$det(sI - A +LC) = {str_Det}$""")
sym_poly = sym.poly(sym_Det, s)
coefs = sym_poly.all_coeffs()
st.latex(f"""
\\begin{{align*}}
\\gamma_3 = {printing.latex(coefs[0])} \\\\
\\end{{align*}} """)
st.latex(f"""
\\begin{{align*}}
\\gamma_2 = {printing.latex(coefs[1])} = {y2} \\\\
\\end{{align*}} """)
st.latex(f"""
\\begin{{align*}}
\\gamma_1 = {printing.latex(coefs[2])} = {y1}  \\\\
\\end{{align*}} """)
st.latex(f"""
\\begin{{align*}}
\\gamma_0 = {printing.latex(coefs[3])} = {y0}  \\\\
\\end{{align*}} """)


result = sym.nsolve(
    (coefs[3] - y0, coefs[2] - y1, coefs[1] - y2), (l1, l2, l3), (1, 1, 1)
)

l1 = float(result[0])
l2 = float(result[1])
l3 = float(result[2])


l1 = Fraction(l1).limit_denominator()
l2 = Fraction(l2).limit_denominator()
l3 = Fraction(l3).limit_denominator()

l1_res = (
    f" \\frac{{{l1.numerator}}}{{{l1.denominator}}}"
    if l1.denominator != 1
    else l1.numerator
)
l2_res = (
    f" \\frac{{{l2.numerator}}}{{{l2.denominator}}}"
    if l2.denominator != 1
    else l2.numerator
)
l3_res = (
    f" \\frac{{{l3.numerator}}}{{{l3.denominator}}}"
    if l3.denominator != 1
    else l3.numerator
)
css = f"""
    div.math-display {{
        font-size: {2}rem;
    }}
    """

st.html(f"<style>{css}</style>")
st.latex(
    f"""

\\begin{{align*}}
l_1 = {l1_res} \\\\
l_2 = {l2_res} \\\\
l_3 = {l3_res} \\\\
\\end{{align*}}
"""
)
