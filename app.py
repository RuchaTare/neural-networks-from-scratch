import streamlit as st
import matplotlib.pyplot as plt

from main import main
from ui_utils.utils import (hide_st_styling, add_network_layer)


st.set_page_config(page_title="Neural Networks from scratch", page_icon="", layout="wide", initial_sidebar_state="auto",)
hide_st_styling()

# maintaining session state for adding network layers field
if "count" not in st.session_state:
    st.session_state["count"] = 0

# title for the UI
st.title("F21BC-CW1: Neural Networks")

# details about the application owners
headcol1, headcol2, headcol3 = st.columns([2, 2, 4])
with headcol3: 
    st.markdown("###### - Created by _Rucha Tare_ & _Tinuade Adeleke_ ")

# configuring sidebar for accepting user parameters
st.write("---")
with st.sidebar:
    st.title("*** Create your own Neural Network ***")
    st.markdown("#### Enter number of layers to add rows below")
    col1, col2, col3 = st.columns([3, 1, 3])

    with col1:
        no_of_layers = st.number_input(label="Number of layers", min_value=1, step=1, key="no_of_layers", label_visibility="collapsed")

    for i in range(int(no_of_layers)):
        add_network_layer(i)

    st.write("---")

    nn_col1, pad1, nn_col2 = st.columns([3, 1, 3])
    with nn_col1:
        learning_rate = st.number_input(label="Learning Rate", min_value=0.0001, step=0.0001, format="%.4f", key="learning_rate")
    with nn_col2:
        loss_function = st.radio(label="Loss Function", 
                                options=["MSE", "BCE", "CE"], 
                                key="loss_function",
                                help="MSE: Mean Squared Error, BCE: Binary Cross Entropy, CE: Cross Entropy")    

    nn_col3, pad3, nn_col4 = st.columns([3, 1, 3])
    with nn_col3:
        gradient_type = st.radio(label="Gradient Type", 
                                options=["SGD", "BGD"], 
                                key="gradient_type",
                                help="SGD: Stochastic Gradient Descent, BGD: Batch Gradient Descent")
    with nn_col4:
        epochs = st.number_input(label="Epochs", min_value=10, max_value=1000, step=10, key="epochs")
        
    st.write("")

    submit = st.button("Submit")
    st.write("---")

# list initialization to capture neural network parameters
layer_sizes = []
layer_activations = []

# action performed on submitting the neural network parameters
if submit:
    for i in range(int(no_of_layers)):
        sizes = "layer_size" + str(i)
        activation_function = "layer_act_func" + str(i)
        layer_sizes.append(st.session_state[sizes])
        layer_activations.append(st.session_state[activation_function])

    config = {
            "sizes": layer_sizes,
            "learning_rate": learning_rate,
            "activation_function": layer_activations,
            "loss_function": loss_function,
            "gradient_type": gradient_type,
            "epochs": epochs
            }
    
    train_accuracy, train_loss_list, train_loss, validation_accuracy, validation_loss_list, validation_loss, test_accuracy = main(config)
    
    met1, pad1, met2 = st.columns([3, 1, 3])
    with met1:
        st.markdown(f"#### Training Accuracy: {str(round(train_accuracy*100, 2))} %")
        st.write("---")
    with met2:
        st.markdown(f"#### Training Loss: {str(round(train_loss*100, 2))} %")
        st.write("---")
    
    met3, pad2, met4 = st.columns([3, 1, 3])
    with met3:
        st.markdown(f"#### Validation Accuracy: {str(round(validation_accuracy*100, 2))} %")
    with met4:
        st.markdown(f"#### Validation Loss: {str(round(validation_loss*100, 2))} %")

    pad4, met5, pad5 = st.columns([2, 3, 2])
    with met5:
        st.write("---")
        st.markdown(f"#### Testing Accuracy: {str(round(test_accuracy*100, 2))} %")
    
    st.write("---")

    st.subheader(" Training Loss vs Validation Loss")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.plot(train_loss_list, label='Train')
    plt.plot(validation_loss_list, label='Validation')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    
    plt.legend()

    fig = plt.show()

    st.pyplot(fig)