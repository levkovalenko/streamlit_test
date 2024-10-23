from fractions import Fraction

import streamlit as st

st.title("Проверка контрольной")

a = st.number_input("$a$", value=-2)
b = st.number_input("$b$", value=1)
c = st.number_input("$c$", value=-7)
d = st.number_input("$d$", value=3)
s1 = st.number_input("$s_1$", value=-1)
s2 = st.number_input("$s_2$", value=-2)
s3 = st.number_input("$s_3$", value=-2)
y0 = -s1 * s2 * s3
y1 = s2 * s3 + s1 * s3 + s1 * s2
y2 = -s3 - s2 - s1
l1 = (a + y0) / (a * d)
l2 = 1 / d * (c + y2)
l3 = 1 / d * (b + c**2 + c * y2 + y1)
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
\\gamma_0= {y0} \\\\
\\gamma_1= {y1} \\\\
\\gamma_2= {y2} \\\\
\\end{{align*}} 
"""
)
