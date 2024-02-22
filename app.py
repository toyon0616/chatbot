import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
OPENAI_API_KEY = st.secrets['APIKEY']['OPENAI_API_KEY']
template = """
あなたは聞かれた質問に答える優秀なアシスタントです。
以下に株式会社Nucoの会社情報を書きます。

会社概要

会社名
株式会社Nuco
(English : Nuco Inc.)

設立日
2017年6月12日

代表者名
小山内美悠

所在地
〒150-0011
東京都渋谷区東1-26-20
東京建物東渋谷ビル 6F

事業内容
AIシステムの企画・開発・運営・販売

お問い合わせ
https://nuco.co.jp/contact

サービス例
データ活用コンサルティング
大規模データ基盤構築
AIシステム開発
AI人材育成コンサルティング
機械学習セミナー
これを元に質問に答えてください。
"""
# 会話のテンプレートを作成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

#会話の読み込みを行う関数を定義
@st.cache_resource
def load_conversation():
    llm = ChatOpenAI(
        model_name="gpt-4-0613",
        temperature=0,
	openai_api_key=OPENAI_API_KEY
    )
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(
        memory=memory,
        prompt=prompt,
        llm=llm)
    return conversation

# 質問と回答を保存するための空のリストを作成
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

# 送信ボタンがクリックされた後の処理を行う関数を定義
def on_input_change():
    user_message = st.session_state.user_message
    conversation = load_conversation()
    answer = conversation.predict(input=user_message)

    st.session_state.generated.append(answer)
    st.session_state.past.append(user_message)

    st.session_state.user_message = ""

# タイトルやキャプション部分のUI
st.title("Nuco Chatbot")
st.caption("Nuco Inc.")
st.write("株式会社Nucoについての質問に答えます。")

# 会話履歴を表示するためのスペースを確保
chat_placeholder = st.empty()

# 会話履歴を表示
with chat_placeholder.container():
    for i in range(len(st.session_state.generated)):
        message(st.session_state.past[i],is_user=True)
        message(st.session_state.generated[i])

# 質問入力欄と送信ボタンを設置
with st.container():
    user_message = st.text_input("質問を入力する", key="user_message")
    st.button("送信", on_click=on_input_change)