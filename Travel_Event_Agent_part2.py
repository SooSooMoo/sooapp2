import streamlit as st
from crewai import Crew, Agent, Task
from langchain.llms import OpenAI

# --- UI ---
st.title("🌍 海外観光＆イベント観戦旅行プランナー")

with st.form("input_form"):
    location = st.text_input("🎯 渡航先（都市または国）", "バルセロナ")
    genre = st.selectbox("🏟️ 観戦ジャンル", [
        "サッカー", "野球", "テニス", "音楽フェス", "モータースポーツ", "その他"])
    date = st.date_input("📅 出発日（任意）", None)
    openai_key = st.text_input("🔑 OpenAI APIキー", type="password")
    submitted = st.form_submit_button("旅行プランを生成")

if submitted:
    if not openai_key:
        st.error("OpenAI APIキーを入力してください")
        st.stop()

    # --- LLM初期化 ---
    llm = OpenAI(api_key=openai_key, temperature=0.7)

    # --- エージェント定義 ---
    event_agent = Agent(
        role="イベント調査エージェント",
        goal=f"{location}で{genre}のイベントを調査して日程をまとめる",
        backstory="海外イベントに詳しい文化ジャーナリスト",
        verbose=True,
        llm=llm
    )

    tour_agent = Agent(
        role="観光地ガイドエージェント",
        goal=f"{location}の観光名所をリストアップしておすすめする",
        backstory="旅行ガイドブックを100冊以上執筆した旅のプロ",
        verbose=True,
        llm=llm
    )

    plan_agent = Agent(
        role="旅行プラン作成エージェント",
        goal="イベントと観光を組み合わせた3日間の旅行スケジュールを作る",
        backstory="旅行代理店で20年プランニングしてきたプロ",
        verbose=True,
        llm=llm
    )

    # --- タスク定義 ---
    task_event = Task(
        description=f"{location}で{genre}のイベントを調査してください。日程がわかれば明記してください。",
        agent=event_agent
    )

    task_tour = Task(
        description=f"{location}で人気の観光地を5〜7箇所おすすめしてください。理由付きで。",
        agent=tour_agent
    )

    task_plan = Task(
        description="前の2つの情報をもとに、3日間の旅行スケジュールを日程付きで提案してください。",
        agent=plan_agent
    )

    # --- Crewの作成・実行（ChromaDB無効化） ---
    crew = Crew(
        agents=[event_agent, tour_agent, plan_agent],
        tasks=[task_event, task_tour, task_plan],
        verbose=True,
        memory=False  # ← ChromaDBを使わない設定
    )

    result = crew.kickoff()

    # --- 出力 ---
    st.success("✅ 旅行プランを生成しました！")
    st.markdown("### 🧳 あなたの観光＋観戦プラン")
    st.markdown(result)
