import streamlit as st


def hide_st_styling():
    hide_st_style = """
                    <style>
                    #MainMenu {visibility:hidden}
                    footer {visibility:hidden}
                    header {visibility:hidden}
                    </style>
                    """
    st.markdown(hide_st_style, unsafe_allow_html=True)


def add_network_layer(layer_id):

    col1, col2, col3 = st.columns([3, 1, 3])

    with col1:
        sizes = "layer_size" + str(layer_id)
        st.number_input(label=f"Size of layer - {layer_id+1}", 
                        min_value=1, step=1,
                        help="Enter size of layer",
                        key=sizes)
    
    with col3:
        activation_function = "layer_act_func" + str(layer_id)
        st.selectbox(label=f"Activation Function for layer - {layer_id+1}", 
                     options=["tanh", "ReLU", "LeakyReLU", "sigmoid"],
                     help="Select activation functions for layer", 
                     key=activation_function)