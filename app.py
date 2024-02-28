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
以下に木幸スポーツ企画の会社情報を書きます。

会社概要

会社名
木幸スポーツ企画株式会社
(English : KIKOH SPORTS KIKAKU.)

会社沿革

平成16年7月	滋賀県高島町学校プール管理業務受託

平成16年8月	枚方スイミングスクールよりアテネオリンピック競泳女子選手      　　　　　　　　		銅メダリスト輩出

平成17年6月	奈良県葛城市スポーツセンタープール運営管理 業務受託

平成17年8月	枚方スイミングスクール世界水泳モントリオール 競泳女子選		手メダリスト輩出

平成18年4月	交野市総合体育施設指定管理者として運営開始

平成18年8月	枚方スイミングスクール2006パンパシフィック水泳女子選手メ		ダリスト輩出

平成19年10月	奈良県生駒市「木幸スポーツ生駒」開校

平成20年4月　　枚方市総合福祉会館温水プール「ラポールひらかた温水プー		ル」運営管理業務受託

平成20年8月　　	枚方スイミングスクールより北京オリンピック競泳女子選手輩		出

代表者名
新庄幸一

取締役名
太田伸

所在地
〒573-0026 大阪府枚方市朝日丘町2番19号

事業内容
スポーツ愛好者の親睦と交流を図るコミュニケーション誌の発行業務とそれに付帯する一切の業務

スポーツ施設、宿泊施設の経営

フィットネスクラブ、スイミングスクールの経営および経営指導の請負

スポーツインストラクターの養成

温泉施設の運営管理

フィットネスクラブ、スイミングスクールに関する市場調査、企画設計管理
企業の従業員等の体力、健康調査および健康づくり行事の企画代行業

入会金


会員種別のご案内


ひとりひとりに合わせた安心指導プログラム										
　枚方フィットネスクラブでは、充実した施設や設備と経験豊富な専任インストラクターの指導体制により、トレーニングをサポート。ひとり、ひとりの目的や年齢、体力に合わせたプログラムを作成いたします。だから、カラダに無理なく楽しみながらトレーニングＯＫ！　スケジュールに縛られることなく、週何回でもご利用いただけます。										
							
入会金・会費

									
会員種別	入館時間	～	退館時間	利用制限	初期登録料		会費		休会	
月払会員	９：３０　　　　　（始業）	～	２２：００　　　　　　（終業）	"休館日・成人休みを除く

スイミングレッスンを
無制限で受講可能"	16,500円		11,440 	円／月	あり	
いつでも会員							10,560 	円／月	なし	
"ゴールデンエイジ会員
（70歳以上限定）"							10,120 	円／月	なし	
平日会員	９：３０　　　　　（始業）	～	１７：００	"休館日・成人休み
日・祝日を除く"	13,200円		9,240 	円／月	あり	
あさひる会員							8,910 	円／月	なし	
ひるだけ会員	１３：３０	～	１７：３０				6,380 	円／月	なし	
よるだけ会員	１９：００	～	２２：００（終業）	"休館日・成人休み
土・日・祝日を除く"			6,710 	円／月	なし	
										
よるだけプラス会員	１８：００	～	２２：００（終業）	"休館日・成人休み
日・祝日を除く"			7,590 	円／月	なし	
										
休みだけ会員	９：３０　　　　　（始業）	～	２２：００（終業）	"休館日・成人休みを除く
土・日・祝日に限る"			7,040 	円／月	なし	
										
泳ぐだけ会員	９：３０	～	２２：００	"タイムテーブル内の
プール利用時間に限る"			7,040 	円／月	なし	
	（プール利用時間終了後30分にて退館）									
※休会される場合は、休会届に費用を添えて期日までにお届け下さい。（休会制度の有無をご確認ください。）										
※ゴールデンエイジ会員は70歳の誕生日を迎える月から変更が可能です。自動的に変更はされませんので別途フロント										
※で手続きをお願いします。（変更希望月の前月25日までにお願いします。別途変更手数料がいります。）										
●　スイミングレッスン（スクーリングシステム）										
"フィットネスクラブ会員の方で、スイミングレッスンをご希望の方は、枚方スイミングスクールプログラムを受講することができます。（別途申込み、料金が必要です）
※月払・いつでも・ゴールデンエイジ会員の方は、お申し込み不要です。"										
										
										
週１回指導（月　４回）				フィットネスクラブ会費　＋　3,190円						
週２回指導（月　８回）				フィットネスクラブ会費　＋　6,050円						
●　施設受付・営業時間案内										
	営　業　時　間			受　付　時　間	マシーンジム			ｻｳﾅ・ｼﾞｬｸﾞｼﾞｰﾊﾞｽ		
月曜日～土曜日	９：３０～２２：００			９：３０～２１：００	１０：００～２１：３０			９：３０～２１：３０		
日曜日	９：００～１７：３０			９：００～１６：００	１０：００～１７：００			９：３０～１６：３０		
祝　日	９：３０～１７：３０			９：３０～１６：００						
※プール・スタジオプログラム等は、別途タイムテーブルをご覧下さい。										
●　入会手続き										
入会申込・健康申告書（フロントに常備）要捺印・入会事務手数料2,200円・入会金・会費（月払2ヶ月分）・キャッシュカードをお持ち下さい。										
										





お問い合わせ
https://www.kikoh-sports.com/contact/



これを元に質問に答えてください。

上記以外の質問には答えれない場合は以下のURLを提示してください。
https://www.kikoh-sports.com/index.html
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
st.title("KIKOH SPORTS KIKAKU")
st.caption("KIKOU Inc.")
st.write("株式会社木幸スポーツ企画についての質問に答えます。")

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