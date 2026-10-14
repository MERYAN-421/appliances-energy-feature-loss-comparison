import streamlit as st
import numpy as np
import pandas as pd
st.title("📉 Loss Functions")
st.write("**Loss functions change how training penalizes residuals** `e = actual − predicted`. We compare four linear estimators/objectives.")
rows=[
("Squared Error (OLS)","e²","Large errors receive quadratic penalty","LinearRegression"),
("Huber","Quadratic near zero; linear for large |e|","Compromise between squared and absolute","HuberRegressor"),
("Absolute Error (Median)","|e|","More robust to large residuals","QuantileRegressor(quantile=0.5)"),
("Epsilon-insensitive","max(0, |e|−ε)","No loss inside epsilon tube","SGDRegressor(loss='epsilon_insensitive')"),
]
st.dataframe(pd.DataFrame(rows,columns=["Training objective","Illustrative loss","Main effect","sklearn implementation"]),hide_index=True,use_container_width=True)
st.subheader("Explore the shapes (illustration only — not model results)")
eps=st.slider("Epsilon-insensitive threshold ε",0.1,2.0,0.5,0.1)
delta=st.slider("Huber transition δ",0.5,3.0,1.0,0.1)
e=np.linspace(-4,4,200)
huber=np.where(np.abs(e)<=delta,0.5*e**2,delta*(np.abs(e)-0.5*delta))
curves=pd.DataFrame({"Residual e":e,"Squared":e**2,"Huber":huber,"Absolute":np.abs(e),"Epsilon-insensitive":np.maximum(0,np.abs(e)-eps)}).set_index("Residual e")
st.line_chart(curves)
st.caption("Controlled comparison caveat: sklearn estimators may differ in solvers, scaling, and regularization as well as objective. Tune those consistently and describe this limitation.")
