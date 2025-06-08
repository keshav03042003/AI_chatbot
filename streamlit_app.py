import streamlit as st
import replicate
import os
import httpx
import time

# App title
st.set_page_config(page_title="🦙💬 AI Chatbot")

# Replicate Credentials & Sidebar UI
with st.sidebar:
    st.title('🦙💬 AI Chatbot')
    st.write('This chatbot uses the open-source AI Chatbot LLM model from Meta.')

    # ✅ Set Replicate API token directly (hardcoded for testing)
    replicate_api = "r8_Rsvb67fIVF0Bu5UX2oGOUwv6iBnzQBE1d4ckH"
    os.environ["REPLICATE_API_TOKEN"] = replicate_api
    st.success('Using hardcoded Replicate API token!', icon='✅')

    # Model selection
    selected_model = st.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    else:
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'

    # Model parameters
    temperature = st.slider('temperature', 0.01, 1.0, 0.1, 0.01)
    top_p = st.slider('top_p', 0.01, 1.0, 0.9, 0.01)
    max_length = st.slider('max_length', 20, 80, 50, 5)

    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Show chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Initialize replicate client once
client = replicate.Client(api_token=replicate_api)

# Generate response using Replicate with retry and timeout handling
def generate_llama2_response(prompt_input, retries=3, timeout=60):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.\n\n"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    input_payload = {
        "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "repetition_penalty": 1,
    }

    for attempt in range(1, retries + 1):
        try:
            # The replicate.Client doesn't support timeout param directly,
            # so we rely on retry logic here.
            output = client.run(llm, input=input_payload)
            return output
        except httpx.ReadTimeout:
            st.warning(f"⏱ Timeout on attempt {attempt}. Retrying...", icon="⚠️")
            time.sleep(2)
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            break

    raise RuntimeError("🛑 Failed to get a response after multiple retries.")

# Chat input and response
if replicate_api:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = generate_llama2_response(prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except RuntimeError as e:
                    st.error(str(e))
else:
    st.info("Please enter a valid Replicate API token to start chatting.")

