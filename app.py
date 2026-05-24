import os
import streamlit as st
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

load_dotenv()

# setting up groq - llama 3.3 is free and pretty good
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# main prompt - took a while to get the json format right
resume_prompt = PromptTemplate(
    input_variables=["resume", "jd"],
    template="""You are a career coach helping a friend. Be honest but encouraging, not robotic.
Analyze the resume against the job description.

Respond ONLY in raw JSON, no markdown, no extra text:
{{
  "score": <integer 0-100>,
  "verdict": "<2-3 sentences like you're talking to a friend, be real>",
  "found_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "tips": ["tip1", "tip2", "tip3"],
  "next_steps": [
    {{"step": "<short title>", "detail": "<one sentence what to do>"}},
    {{"step": "<short title>", "detail": "<one sentence what to do>"}},
    {{"step": "<short title>", "detail": "<one sentence what to do>"}}
  ]
}}

RESUME:
{resume}

JOB DESCRIPTION:
{jd}"""
)

# tools for the agent
# each one does one specific thing, agent decides which to call

def skill_gap_tool(skill: str) -> str:
    # gives a quick learning plan for a missing skill
    prompt = f"Give a short 2-sentence learning plan to acquire this skill: {skill}. Be specific, mention free resources."
    return llm.invoke(prompt).content

def resume_tip_tool(section: str) -> str:
    # suggests how to improve a resume section
    prompt = f"Give one specific actionable tip to improve this resume section: {section}"
    return llm.invoke(prompt).content

def next_action_tool(summary: str) -> str:
    # agent figures out top priority based on full analysis
    prompt = f"""You're a career agent. Here's the resume analysis:
{summary}

What's the single most important thing this person should do RIGHT NOW?
Start with 'Here's what I'd do first...' and keep it to 3-4 sentences. Be specific."""
    return llm.invoke(prompt).content

tools = [
    Tool(
        name="SkillGapAdvisor",
        func=skill_gap_tool,
        description="use this to give a learning plan for a missing skill"
    ),
    Tool(
        name="ResumeTipAdvisor",
        func=resume_tip_tool,
        description="use this to improve a specific section of the resume"
    ),
    Tool(
        name="NextActionPlanner",
        func=next_action_tool,
        description="use this to decide the top priority action for the user"
    ),
]

# memory so agent remembers the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True
)

# ---- ui starts here ----

st.set_page_config(page_title="Resume Screener", page_icon="🎯")
st.title("🎯 Resume Screener")
st.caption("paste your resume + job description, get a real analysis")

st.divider()

col1, col2 = st.columns(2)
with col1:
    resume = st.text_area("your resume", height=280, placeholder="paste resume text here...")
with col2:
    jd = st.text_area("job description", height=280, placeholder="paste job description here...")

if st.button("analyze →", type="primary", use_container_width=True):
    if not resume.strip() or not jd.strip():
        st.error("need both fields filled in")
        st.stop()

    # step 1 - run the main analysis
    with st.spinner("reading your resume..."):
        chain = LLMChain(llm=llm, prompt=resume_prompt)
        raw = chain.run(resume=resume, jd=jd)

        # groq sometimes adds extra text, strip it just in case
        raw = raw.strip().strip("```json").strip("```").strip()

        # print(raw)  # debug

        try:
            result = json.loads(raw)
        except:
            # sometimes model wraps in extra text, try to extract json
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                result = json.loads(raw[start:end])
            except:
                st.error("something went wrong parsing the response, try again")
                st.stop()

    score = result.get("score", 0)

    # step 2 - show score
    st.divider()
    if score >= 70:
        st.success(f"### ✅ match score: {score}/100 — strong fit")
    elif score >= 40:
        st.warning(f"### ⚠️ match score: {score}/100 — partial fit")
    else:
        st.error(f"### ❌ match score: {score}/100 — needs work")

    # conversational verdict
    st.markdown(f"> 💬 *{result.get('verdict', '')}*")
    st.divider()

    # step 3 - skills breakdown
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**what you already have ✅**")
        found = result.get("found_skills", [])
        if found:
            for s in found:
                st.markdown(f"- {s}")
        else:
            st.caption("nothing matched")

    with col4:
        st.markdown("**what's missing 📌**")
        missing = result.get("missing_skills", [])
        if missing:
            for s in missing:
                st.markdown(f"- {s}")
        else:
            st.caption("nothing missing, great!")

    st.divider()

    # step 4 - quick tips
    st.markdown("### 💡 quick wins")
    for tip in result.get("tips", []):
        st.markdown(f"→ {tip}")

    st.divider()

    # step 5 - next steps from agent
    st.markdown("### 🗂️ your action plan")
    steps = result.get("next_steps", [])
    for i, s in enumerate(steps):
        with st.expander(f"step {i+1}: {s.get('step', '')}"):
            st.write(s.get("detail", ""))

    st.divider()

    # step 6 - agent decides top priority
    st.markdown("### 🧠 what the agent says to do first")
    summary = f"score: {score}/100. missing skills: {', '.join(missing)}. tips: {result.get('tips', [])}"
    with st.spinner("agent is thinking..."):
        top_action = agent.run(
            f"based on this resume analysis, what should the user focus on first? summary: {summary}"
        )
    st.info(top_action)

    # step 7 - how to close skill gaps
    if missing:
        st.divider()
        st.markdown("### 📚 how to learn the missing skills")
        for skill in missing[:3]:  # max 3 so it doesnt take forever
            with st.spinner(f"planning how to learn {skill}..."):
                advice = agent.run(f"give a learning plan for: {skill}")
            with st.expander(f"how to learn {skill}"):
                st.write(advice)

    st.divider()
    st.caption("fix the missing skills, update your resume, then run it again 🚀")
