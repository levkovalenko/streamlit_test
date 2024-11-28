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

colA1, colA2, colA3, colA4 = st.columns([1, 1, 1, 1])
with colA1:
    st.write("##")
    latex_eq("A =")
    st.write("##")

with colA2:
    a1 = st.number_input("$a_1$", value=0)
    a4 = st.number_input("$a_4$", value=0)
    a7 = st.number_input("$a_7$", value=4)

with colA3:
    a2 = st.number_input("$a_2$", value=1)
    a5 = st.number_input("$a_5$", value=0)
    a8 = st.number_input("$a_8$", value=-1)

with colA4:
    a3 = st.number_input("$a_3$", value=0)
    a6 = st.number_input("$a_6$", value=1)
    a9 = st.number_input("$a_9$", value=-1)

A = np.array([[a1, a2, a3], [a4, a5, a6], [a7, a8, a9]])

colC1, colC2, colC3, colC4 = st.columns([1, 1, 1, 1])
with colC1:
    st.write("####")
    latex_eq("C =")
    st.write("####")

with colC2:
    c1 = st.number_input("$c_1$", value=1)
    c4 = st.number_input("$c_4$", value=0)

with colC3:
    c2 = st.number_input("$c_2$", value=-1)
    c5 = st.number_input("$c_5$", value=0)

with colC4:
    c3 = st.number_input("$c_3$", value=0)
    c6 = st.number_input("$c_6$", value=1)
C = np.array([[c1, c2, c3], [c4, c5, c6]])


colD1, colD2, _ = st.columns([1, 1, 2])
with colD1:
    st.write("##")
    latex_eq("D =")
    st.write("##")

with colD2:
    d1 = st.number_input("$d_1$", value=1)
    d2 = st.number_input("$d_2$", value=1)
    d3 = st.number_input("$d_3$", value=-1)

D = np.array([[d1], [d2], [d3]])


C1 = C[:1]
C2 = C[-1:]


colCheck1, colCheck2 = st.columns([1, 1])
with colCheck1:
    c1_choose = st.checkbox("$C'D=0$", value=C1 @ D != 0)
with colCheck2:
    c2_choose = st.checkbox("$C''D=0$", value=C2 @ D != 0)

if c1_choose:
    C_ = C1
    Clast = C2
elif c2_choose:
    C_ = C2
    Clast = C1
else:
    st.stop()


colF1, colF2, colF3, colF4 = st.columns([1, 1, 1, 1])

with colF1:
    st.write("##")
    latex_eq("P =")
    st.write("##")

with colF2:
    f1 = st.number_input("$f_1$", value=1)
    f4 = st.number_input("$f_4$", value=0)
    st.number_input("$C'_1$", value=C_[0, 0], disabled=True)

with colF3:
    f2 = st.number_input("$p_2$", value=0)
    f5 = st.number_input("$p_5$", value=1)
    st.number_input("$C'_2$", value=C_[0, 1], disabled=True)

with colF4:
    f3 = st.number_input("$p_3$", value=0)
    f6 = st.number_input("$p_6$", value=0)
    st.number_input("$C'_3$", value=C_[0, 2], disabled=True)
F = np.array([[f1, f2, f3], [f4, f5, f6]])
P = np.vstack([F, C_])


symA = sym.Matrix(A)
symC = sym.Matrix(C)
symD = sym.Matrix(D)
symP = sym.Matrix(P)
symF = sym.Matrix(F)


latex_eq(f"P^{{-1}} = {printing.latex(symP.inv())}")

P_inv = np.linalg.inv(P)
T1 = P_inv[:, :2]
T2 = P_inv[:, -1:]
symT1 = sym.Matrix(T1)
symT2 = sym.Matrix(T2)

latex_eq(f"T' = {printing.latex(symT1)}, T'' = {printing.latex(symT2)}")

I = np.eye(3)
symI = sym.Matrix(I)
Q = F @ (I - D @ np.linalg.inv(C_ @ D) @ C_)
symQ = sym.Matrix(Q)

latex_eq(f"Q = {printing.latex(symQ)}")

A11 = Q @ A @ T1
A12 = Q @ A @ T2
Cr = Clast @ T1
symA11 = sym.Matrix(A11)
symA12 = sym.Matrix(A12)
symCr = sym.Matrix(Cr)


latex_eq(f"A_{{11}} = {printing.latex(symA11)}")
latex_eq(f"A_{{12}} = {printing.latex(symA12)}")
latex_eq(f"\\tilde{{C}} = {printing.latex(symCr)}")


s1 = st.number_input("$ s_1$", value=-2)
s2 = st.number_input("$ s_2$", value=-2)
y0 = s1 * s2
y1 = -s1 - s2
l1, l2 = sym.symbols("l_1,l_2")
s = sym.symbols("s")
symS = sym.diag(
    s,
    s,
)
symL = sym.Matrix([[l1], [l2]])
sym_sIACL = symS - symA11 + symL @ symCr
latex_eq(f"det(sI - A_{{11}} +L\\tilde{{C}}) = {printing.latex(sym_sIACL)}")
sym_Det = sym_sIACL.det()
str_Det = printing.latex(sym_Det)
st.markdown(f"""$det(sI - A +LC) = {str_Det}$""")
sym_poly = sym.poly(sym_Det, s)
coefs = sym_poly.all_coeffs()
latex_eq(f"s^2 = {printing.latex(coefs[0])}")
latex_eq(f"s^1 = {printing.latex(coefs[1])}")
latex_eq(f"s^0 = {printing.latex(coefs[2])}")

result = sym.solve((coefs[2] - y0, coefs[1] - y1), (l1, l2))

latex_eq(f"{printing.latex(result).replace(':', '=')}")
