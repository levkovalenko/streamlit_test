from fractions import Fraction

import numpy as np
import streamlit as st
import sympy as sym
from sympy import printing

st.title("Проверка контрольной")
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


def task_state(id: str):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.write("##")
        latex_eq("A =")
        st.write("##")

    with col2:
        a1 = st.number_input(f"${id}a_1$", value=0)
        a4 = st.number_input(f"${id}a_4$", value=0)
        a7 = st.number_input(f"${id}a_7$", value=-2)

    with col3:
        a2 = st.number_input(f"${id}a_2$", value=1)
        a5 = st.number_input(f"${id}a_5$", value=0)
        a8 = st.number_input(f"${id}a_8$", value=1)

    with col4:
        a3 = st.number_input(f"${id}a_3$", value=0)
        a6 = st.number_input(f"${id}a_6$", value=1)
        a9 = st.number_input(f"${id}a_9$", value=1)

    col11, col21, col31, col41 = st.columns([1, 1, 1, 1])

    with col11:
        latex_eq("C =")
        st.write("####")
        latex_eq("B =")

    with col21:
        c1 = st.number_input(f"${id}c_1$", value=-1)
        st.write("#")
        b1 = st.number_input(f"${id}b_1$", value=-1)

    with col31:
        c2 = st.number_input(f"${id}c_2$", value=-1)
        st.write("#")
        b2 = st.number_input(f"${id}b_2$", value=-1)

    with col41:
        c3 = st.number_input(f"${id}c_3$", value=0)
        st.write("#")
        b3 = st.number_input(f"${id}b_3$", value=0)

    A = np.array([[a1, a2, a3], [a4, a5, a6], [a7, a8, a9]])
    C = np.array([c1, c2, c3])
    B = np.array([[b1], [b2], [b3]])

    N = np.vstack([C, C @ A, C @ A @ A])
    s = sym.symbols("s")
    symS = sym.diag(s, s, s)
    symA = sym.Matrix(A)
    symC = sym.Matrix([C])
    symB = sym.Matrix(B)

    return A, B, C, N, s, symS, symA, symC, symB


tab1, tab2, tab3 = st.tabs(["Задача 1", "Задача 2", "Задача 3"])

with tab1:
    A, B, C, N, s, symS, symA, symC, symB = task_state("")
    s1 = st.number_input("$s_1$", value=-1)
    s2 = st.number_input("$s_2$", value=-1)
    s3 = st.number_input("$s_3$", value=-2)
    y0 = -s1 * s2 * s3
    y1 = s2 * s3 + s1 * s3 + s1 * s2
    y2 = -s3 - s2 - s1

    latex_eq(f"N = {printing.latex(sym.Matrix(N))}")
    latex_eq(f"rank(N) = {np.linalg.matrix_rank(N)}")

    l1, l2, l3 = sym.symbols("l_1,l_2,l_3")
    symL = sym.Matrix([[l1], [l2], [l3]])
    sym_sIACL = symS - symA + symL @ symC
    latex_eq(f"det(sI - A +LC) = {printing.latex(sym_sIACL)}")

    sym_Det = sym_sIACL.det()
    str_Det = printing.latex(sym_Det)
    st.markdown(f"""$det(sI - A +LC) = {str_Det}$""")
    sym_poly = sym.poly(sym_Det, s)
    coefs = sym_poly.all_coeffs()
    st.divider()
    latex_eq(f"\\gamma_3 = {printing.latex(coefs[0])}")
    latex_eq(f"\\gamma_2 = {printing.latex(coefs[1])}")
    latex_eq(f"\\gamma_1 = {printing.latex(coefs[2])}")
    latex_eq(f"\\gamma_0 = {printing.latex(coefs[3])}")

    result = sym.nsolve(
        (coefs[3] - y0, coefs[2] - y1, coefs[1] - y2), (l1, l2, l3), (1, 1, 1)
    )

    l1 = Fraction(float(result[0])).limit_denominator()
    l2 = Fraction(float(result[1])).limit_denominator()
    l3 = Fraction(float(result[2])).limit_denominator()

    l1_res = (
        f"\\frac{{{l1.numerator}}}{{{l1.denominator}}}"
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
    st.divider()
    latex_eq(f"l_1 = {l1_res}")
    latex_eq(f"l_2 = {l2_res}")
    latex_eq(f"l_3 = {l3_res}")


with tab2:
    A, B, C, N, s, symS, symA, symC, symB = task_state(" ")
    A = A.T
    symA = sym.Matrix(A)
    s1 = st.number_input("$ s_1$", value=-1)
    s2 = st.number_input("$ s_2$", value=-1)
    y0 = s1 * s2
    y1 = -s1 - s2
    P = np.array([[1, 0, -y0], [0, 1, -y1], [0, 0, 1]])
    symP = sym.Matrix(P)
    latex_eq(f"P = {printing.latex(symP)}")

    latex_eq(
        f"\\tilde{{A}} = PAP^{{-1}} = {printing.latex(symP)} * {printing.latex(symA)} * {printing.latex(symP.inv())}"
    )
    latex_eq(f"\\tilde{{A}} = {printing.latex(symP@symA@symP.inv())}")
    latex_eq(
        f"\\tilde{{B}} = PB = {printing.latex(symP)}*{printing.latex(symB)} = {printing.latex(symP@symB)}"
    )


with tab3:
    A, B, C, N, s, symS, symA, symC, symB = task_state("  ")
    spec, _ = np.linalg.eig(A)

    zlay_point = []
    for i, si in enumerate(spec):
        si = round(si)
        latex_eq(f"s_{i} = {si}")
        symSpecS = sym.diag(si, si, si) - symA
        symR = symSpecS.row_insert(4, symC)
        latex_eq(
            f"rank(R({si})) = {printing.latex(symR)} = {printing.latex(symR.rank())}"
        )
        if symR.rank() < 3:
            zlay_point.append(si)

    s1 = st.number_input("$  s_1$", value=zlay_point[0] if len(zlay_point) > 0 else 0)
    s2 = st.number_input("$  s_2$", value=zlay_point[1] if len(zlay_point) > 1 else 0)
    s3 = st.number_input("$  s_3$", value=zlay_point[2] if len(zlay_point) > 2 else 0)
    y0 = -s1 * s2 * s3
    y1 = s2 * s3 + s1 * s3 + s1 * s2
    y2 = -s3 - s2 - s1

    l1, l2, l3 = sym.symbols("l_1,l_2,l_3")
    symL = sym.Matrix([[l1], [l2], [l3]])
    sym_sIACL = symS - symA + symL @ symC
    latex_eq(f"det(sI - A +LC) = {printing.latex(sym_sIACL)}")

    sym_Det = sym_sIACL.det()
    str_Det = printing.latex(sym_Det)
    st.markdown(f"""$det(sI - A +LC) = {str_Det}$""")
    sym_poly = sym.poly(sym_Det, s)
    coefs = sym_poly.all_coeffs()
    st.divider()
    latex_eq(f"\\gamma_3 = {printing.latex(coefs[0])}")
    latex_eq(f"\\gamma_2 = {printing.latex(coefs[1])}")
    latex_eq(f"\\gamma_1 = {printing.latex(coefs[2])}")
    latex_eq(f"\\gamma_0 = {printing.latex(coefs[3])}")

    result = sym.solve((coefs[3] - y0, coefs[2] - y1, coefs[1] - y2), (l1, l2, l3))

    latex_eq(f"{printing.latex(result).replace(':', '=')}")
