import streamlit as st 
from sklearn.tree import DecisionTreeClassifier as dtc 
from sklearn.model_selection import train_test_split as tts 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score as ass
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression as Lr
from sklearn.linear_model import LogisticRegression as LOR
page_bg_img = '''
<style>
body {  
background-image: url("luffy-gear-5-one-piece-thumb.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
columns1,columns2,columns3=st.columns(3)
st.header("This is a IV and ML visualization website ")
st.write("In this website, we can visualize the files which are in csv format and analyze them using histogram, lineplot or 2 and 3 dimensional plot")
st.write("We can use ML to analyze the file using ml models")
upload=st.file_uploader("Upload the file which u want to display")
if upload:
    df=pd.read_csv(upload)
    st.write(df)
    st.write(df.describe())
    st.write("if you want to use ml to analyze:")
    button=st.button("Machine learning")
    st.write("how many feature u want to drop ")
    features=[]
    features=st.text_input("Enter",)
    if features:
        exclude = []
        for i in range(int(features)):
            feature_name = st.text_input(f"Enter feature {i+1} to exclude:")
            if feature_name:
                exclude.append(feature_name)
        target = st.text_input("Enter the target which you want to find:")
    
        if exclude and target:
            X = df.drop(columns=exclude)
            y = df[target]
            xtrain, xtest, ytrain, ytest = tts(X, y, test_size=0.2)
            st.write("which model you want to use")
            radio1=st.radio("Choose the model",["DecisionclassifierTree","Gradient Boosting","KNN","Logistic Regression","Support Vector Machines"])
            if radio1=="DecisionclassifierTree":
                model = dtc()
                model.fit(xtrain, ytrain)
                selection=st.selectbox("Choose one:", ['accuracy'])
                if selection=="accuracy":
                    prediction=model.predict(xtest)
                    accuracy=ass(ytest,prediction)
                    st.write(accuracy)
            elif radio1=="Gradient Boosting":
                model = GradientBoostingClassifier()
                model.fit(xtrain, ytrain)
                selection=st.selectbox("Choose one:", ['accuracy'])
                if selection=="accuracy":
                    prediction=model.predict(xtest)
                    accuracy=ass(ytest,prediction)
                    st.write(accuracy)
            elif radio1=="KNN":
                model = KNN()
                model.fit(xtrain, ytrain)
                selection=st.selectbox("Choose one:", ['accuracy'])
                if selection=="accuracy":
                    prediction=model.predict(xtest)
                    accuracy=ass(ytest,prediction)
                    st.write(accuracy)
            elif radio1=="Support Vector Machines":
                model = SVC()
                model.fit(xtrain, ytrain)
                selection=st.selectbox("Choose one:", ['accuracy'])
                if selection=="accuracy":
                    prediction=model.predict(xtest)
                    accuracy=ass(ytest,prediction)
                    st.write(accuracy)
            elif radio1=="Logistic Regression":
                model = LOR()
                model.fit(xtrain, ytrain)
                selection=st.selectbox("Choose one:", ['accuracy'])
                if selection=="accuracy":
                    prediction=model.predict(xtest)
                    accuracy=ass(ytest,prediction)
                    st.write(accuracy)
    st.write("Which mode u want to  visualize")
    check=st.selectbox('Choose one:', ['2d', '3d'])

    if check=="2d":
        x=st.text_input("Enter the X value")    
        y=st.text_input("Enter the Y value") 
        if x and y:
            radio=st.selectbox("plot type",["Line","scatter"])
            if radio=="scatter":
                fig1=px.scatter(df,x=x,y=y)
                st.plotly_chart(fig1)
            elif radio=="Line" :
                fig=px.line(df,x=x,y=y)
                st.plotly_chart(fig)   
    elif check=="3d":
        x=st.text_input("Enter the X value")    
        y=st.text_input("Enter the Y value") 
        z=st.text_input("Enter the Z value") 
        if x and y and z:
            radio=st.selectbox("plot type",["Line3d","scatter3d"])
            if radio=="scatter3d":
                fig1=px.scatter_3d(df,x=x,y=y,z=z)
                st.plotly_chart(fig1)
            elif radio=="Line3d" :
                fig=px.line_3d(df,x=x,y=y,z=z)
                st.plotly_chart(fig)   
        
    
