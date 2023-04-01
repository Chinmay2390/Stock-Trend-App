import streamlit as st
from streamlit_option_menu import option_menu

#st.title('HOME')

# 1.as sidebar
# with st.sidebar:
#     selected = option_menu(
#         menu_title=None,
#         menu_icon='cast',
#         options=['HOME','PRODUCT','CONTACT'],
#         icons=['house','book','cast'],
#     )


# 2.as horizontal menu
selected = option_menu(
        menu_title=None,
        menu_icon=None,
        options=['HOME','PRODUCT','CONTACT'],
        icons=['house','book','cast'],
        orientation='horizontal',
    )

if selected == 'HOME':
    #st.write(f"you have selected: ",{selected})
    st.title(f"YOU HAVE SELECTED {selected}")
if selected == 'PRODUCT':
    #st.write(f"you have selected: ",{selected})
    st.title(f"YOU HAVE SELECTED {selected}")
if selected == 'CONTACT':
    #st.write(f"you have selected: ",{selected})
    st.title(f"YOU HAVE SELECTED {selected}")