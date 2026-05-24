# Resume Screener

built this to stop guessing whether my resume fits a job or not

paste your resume + job description and it tells you:
- match score out of 100
- skills you have vs skills missing  
- what to actually do next (ai agent figures this out)
- how to learn the missing skills

## how to run

pip install -r requirements.txt

add your GROQ_API_KEY in a .env file (get it free at console.groq.com)

streamlit run app.py

## tech
- langchain for the agent pipeline
- groq api (llama 3.3, free tier)
- streamlit for the ui

took about 2 days to get the agent + json parsing working properly
