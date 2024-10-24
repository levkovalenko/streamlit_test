import numpy as np
import streamlit as st
import sympy as sym
from sympy import printing

st.title("Генератор не наблюдаемых матриц")
css = f"""
    div.math-display {{
        font-size: {2}rem;
    }}
    """


def latex_eq(string):
    st.latex(f"""
\\begin{{align*}}
{string}
\\end{{align*}} """)


st.html(f"<style>{css}</style>")

min_val = st.number_input("minimal_value", value=-5)
max_val = st.number_input("maximal_value", value=5)
zlay_num = st.number_input("Злые точки", value=1, min_value=1, max_value=2)

fixed_head = st.checkbox("Фиксирвоанный head", value=True)
head_A = np.array([[0, 1, 0], [0, 0, 1]])

while True:
    if fixed_head:
        A = np.vstack([head_A, np.random.randint(min_val, max_val, size=(1, 3))])
    else:
        A = np.random.randint(min_val, max_val, size=(3, 3))
    C = np.random.randint(min_val, max_val, size=(3,))
    symA = sym.Matrix(A)
    symC = sym.Matrix([C])
    spec, _ = np.linalg.eig(A)
    zlay_point = []
    for i, si in enumerate(spec):
        if isinstance(si, np.complex128):
            si = si.real
        si = round(si)
        symSpecS = sym.diag(si, si, si) - symA
        symR = symSpecS.row_insert(4, symC)
        if symR.rank() < 3:
            zlay_point.append(si)
    if len(zlay_point) == zlay_num:
        latex_eq(f"A = {printing.latex(symA)}")
        latex_eq(f"C = {printing.latex(symC)}")
        for i, si in enumerate(zlay_point):
            latex_eq(f"s_{i} = {si}")

        st.write(f"A = {printing.latex(symA)}")
        st.write(f"C = {printing.latex(symC)}")
        break
