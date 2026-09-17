# ========== 导入区 ==========
from langchain.chains import RetrievalQA
# 删掉原来的 Chroma，换成 FAISS
from langchain_community.vectorstores import FAISS
from zhipuai_embedding import ZhipuAIEmbeddings
from zhipuai_llm import ZhipuaiLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
# ========== 工具函数 ==========
def get_retriever():
    """加载本地FAISS向量库，返回检索器"""
    embedding = ZhipuAIEmbeddings()
    
    # 当前文件就在仓库根目录，faiss_index和app.py同级
    current_dir = Path(__file__).parent
    faiss_path = current_dir / "faiss_index"
    
    vectordb = FAISS.load_local(
        folder_path=str(faiss_path),
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})

def combine_docs(docs):
    """把检索到的文档拼接成字符串"""
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    """构建带历史对话的RAG链"""
    retriever = get_retriever()
    # 替换成智谱LLM，不再用OpenAI
    llm = ZhipuaiLLM(
        model_name="glm-4-flash",
        temperature=0,
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False),
         (lambda x: x["input"]) | retriever),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )
    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    qa_history_chain = RunnablePassthrough().assign(
        context=retrieve_docs,
    ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    """流式生成回答"""
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

# ========== Streamlit 界面 ==========
def main():
    st.markdown('### 😊个人大模型应用开发尝试')
    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 初始化RAG链（只创建一次）
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    messages = st.container(height=550)
    # 渲染历史消息
    for message in st.session_state.messages:
        with messages.chat_message(message[0]):
            st.write(message[1])
    # 处理用户输入
    if prompt := st.chat_input("Say something"):
        st.session_state.messages.append(("human", prompt))
        with messages.chat_message("human"):
            st.write(prompt)
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        st.session_state.messages.append(("ai", output))
if __name__ == "__main__":
    main()
