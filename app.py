import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import time
import logging
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIGURAÇÃO INICIAL E LOGGING ---
st.set_page_config(
    page_title="AuraMed OS",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuração de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuraMed")

# --- GESTÃO DE SEGREDOS E SETUP ---
# Tenta pegar a chave do st.secrets, senão usa input manual (fallback de segurança)
api_key = st.secrets.get("GROQ_API_KEY")

if "groq_client" not in st.session_state and api_key:
    st.session_state.groq_client = Groq(api_key=api_key)

# Carregamento de Modelo de Embedding (Cacheado para performance)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

if "embedding_model" not in st.session_state:
    with st.spinner("Inicializando núcleo neural Aura..."):
        st.session_state.embedding_model = load_embedding_model()

# --- CSS PERSONALIZADO (UI/UX) ---
st.markdown("""
<style>
    /* Reset e Fontes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fundo Geral */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Cards Neumórficos */
    .metric-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #eef0f2;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    /* Títulos e Textos */
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 600;
    }
    .subtitle {
        color: #718096;
        font-size: 0.9rem;
    }

    /* Botões Customizados */
    .stButton>button {
        background: linear-gradient(135deg, #319795 0%, #2C7A7B 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(49, 151, 149, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #38B2AC 0%, #319795 100%);
        box-shadow: 0 6px 8px rgba(49, 151, 149, 0.3);
    }

    /* Chat Messages */
    .chat-user {
        background-color: #E6FFFA;
        padding: 10px;
        border-radius: 10px 10px 0 10px;
        margin-bottom: 5px;
        text-align: right;
        color: #234E52;
    }
    .chat-ai {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px 10px 10px 0;
        margin-bottom: 5px;
        border: 1px solid #e2e8f0;
        color: #2D3748;
    }

    /* Ocultar elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- DADOS MOCKADOS (Base de Conhecimento) ---
if "data" not in st.session_state:
    st.session_state.data = {
        "pacientes": [
            {"id": 1, "nome": "Ana Silva", "idade": 32, "historico": "Enxaqueca crônica, alergia a penicilina.", "ultima_consulta": "2023-10-15"},
            {"id": 2, "nome": "Carlos Souza", "idade": 45, "historico": "Hipertensão leve, monitoramento de colesterol.", "ultima_consulta": "2023-11-02"},
            {"id": 3, "nome": "Mariana Lima", "idade": 28, "historico": "Gestante 12 semanas, exames de rotina normais.", "ultima_consulta": "2023-11-10"},
        ],
        "appointments": [
            {"paciente": "Ana Silva", "data": "2023-11-20", "hora": "14:00", "status": "Confirmado"},
            {"paciente": "Carlos Souza", "data": "2023-11-20", "hora": "15:00", "status": "Pendente"},
        ]
    }

# --- FUNÇÕES DE IA (GROQ + EMBEDDINGS) ---

def get_groq_response(system_prompt, user_prompt):
    if not st.session_state.get("groq_client"):
        return "⚠️ Erro: API Key não configurada. Verifique os secrets."
    
    try:
        completion = st.session_state.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Erro Groq: {e}")
        return "⚠️ Aura indisponível no momento."

def semantic_search_patients(query):
    # Simulação de RAG simples local
    if not query:
        return []
    
    model = st.session_state.embedding_model
    docs = [f"{p['nome']} - {p['historico']}" for p in st.session_state.data['pacientes']]
    
    query_emb = model.encode([query])
    doc_embs = model.encode(docs)
    
    scores = cosine_similarity(query_emb, doc_embs)[0]
    results = []
    
    # Retorna pacientes com similaridade > 0.3
    for idx, score in enumerate(scores):
        if score > 0.2:
            results.append((st.session_state.data['pacientes'][idx], score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results]

# --- LÓGICA DE LOGIN ---
def login():
    st.markdown("<div style='text-align: center; margin-top: 50px;'><h1>AuraMed OS</h1><p>Acesse seu espaço clínico</p></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            user_type = st.radio("Eu sou:", ["Médico(a)", "Paciente"], horizontal=True)
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            
            if st.button("Entrar no Sistema", use_container_width=True):
                # Simulação de Auth Simples
                if user_type == "Médico(a)" and username == "admin" and password == "admin":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "doctor"
                    st.session_state.user_name = "Dr. Gênesis"
                    logger.info(f"Login Médico: {username}")
                    st.rerun()
                elif user_type == "Paciente" and username == "user" and password == "user":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "patient"
                    st.session_state.user_name = "Ana Silva" # Simulando paciente 1
                    logger.info(f"Login Paciente: {username}")
                    st.rerun()
                else:
                    st.error("Credenciais inválidas (Tente: admin/admin ou user/user)")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if not api_key:
                st.warning("⚠️ Configure a GROQ_API_KEY no .streamlit/secrets.toml")

# --- DASHBOARD MÉDICO ---
def doctor_dashboard():
    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title(f"Olá, {st.session_state.user_name}")
        st.markdown("<p class='subtitle'>Resumo do dia e inteligência clínica</p>", unsafe_allow_html=True)
    with col_h2:
        if st.button("Sair"):
            st.session_state.logged_in = False
            st.rerun()

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><h3>{len(st.session_state.data['appointments'])}</h3><p class='subtitle'>Consultas Hoje</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h3>3</h3><p class='subtitle'>Pacientes Críticos</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><h3>98%</h3><p class='subtitle'>Satisfação (NPS)</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><h3>R$ 12k</h3><p class='subtitle'>Faturamento Mês</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Layout Principal
    col_main, col_ai = st.columns([2, 1])

    with col_main:
        st.subheader("🗓️ Agenda do Dia")
        df_app = pd.DataFrame(st.session_state.data['appointments'])
        st.dataframe(df_app, use_container_width=True, hide_index=True)

        st.subheader("🔍 Busca Semântica de Prontuários")
        search_query = st.text_input("Digite sintomas, condições ou nome do paciente para buscar na base inteligente:")
        if search_query:
            results = semantic_search_patients(search_query)
            if results:
                for p in results:
                    with st.expander(f"👤 {p['nome']} (Idade: {p['idade']})"):
                        st.write(f"**Histórico:** {p['historico']}")
                        st.write(f"**Última Consulta:** {p['ultima_consulta']}")
                        if st.button(f"Analisar {p['nome']} com IA", key=p['id']):
                            st.session_state.analyze_patient = p
            else:
                st.info("Nenhum paciente encontrado com esses critérios.")

    with col_ai:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🤖 Aura Clinical Copilot")
        
        # Análise de Paciente Selecionado
        if "analyze_patient" in st.session_state:
            p = st.session_state.analyze_patient
            st.info(f"Analisando: {p['nome']}")
            if st.button("Gerar Resumo Clínico"):
                with st.spinner("Aura está processando..."):
                    prompt = f"Analise este paciente: {p}. Sugira próximos passos clínicos e exames de rotina baseados no histórico."
                    response = get_groq_response("Você é um assistente médico sênior. Seja técnico e direto.", prompt)
                    st.write(response)
        
        # Chat Geral
        st.markdown("---")
        st.write("**Quick Consult**")
        q = st.text_input("Dúvida médica rápida:")
        if q and st.button("Consultar Aura"):
            with st.spinner("Pesquisando..."):
                resp = get_groq_response("Você é uma IA médica auxiliar. Responda com base em protocolos médicos padrão.", q)
                st.success(resp)
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- PORTAL DO PACIENTE ---
def patient_dashboard():
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("Seu Espaço de Saúde")
        st.markdown(f"<p class='subtitle'>Bem-vindo(a), {st.session_state.user_name}</p>", unsafe_allow_html=True)
    with col_h2:
        if st.button("Sair"):
            st.session_state.logged_in = False
            st.rerun()
    
    tab1, tab2 = st.tabs(["Minhas Consultas", "Triagem Inteligente"])
    
    with tab1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("### Próximos Agendamentos")
        # Filtra apenas consultas deste paciente (mockado pelo nome)
        my_apps = [a for a in st.session_state.data['appointments'] if a['paciente'] == st.session_state.user_name]
        if my_apps:
            for app in my_apps:
                st.info(f"📅 {app['data']} às {app['hora']} - Status: {app['status']}")
        else:
            st.warning("Você não tem consultas agendadas.")
            st.button("Agendar Nova Consulta")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("### 💬 Converse com a Aura (Triagem)")
        st.markdown("<small>Descreva o que está sentindo. Em caso de emergência, ligue 192.</small>", unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Mostra histórico
        for msg in st.session_state.chat_history:
            role_class = "chat-user" if msg["role"] == "user" else "chat-ai"
            st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

        user_input = st.text_input("Digite aqui...", key="patient_input")
        
        if st.button("Enviar"):
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("Aura está analisando seus sintomas..."):
                    sys_prompt = """
                    Você é a Aura, uma IA de triagem clínica empática. 
                    1. Nunca dê diagnósticos definitivos.
                    2. Sugira a especialidade médica adequada.
                    3. Se parecer grave, instrua ir ao hospital imediatamente.
                    4. Seja breve e acolhedora.
                    """
                    response = get_groq_response(sys_prompt, user_input)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        if st.button("Limpar Conversa"):
            st.session_state.chat_history = []
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

# --- ROTEAMENTO PRINCIPAL ---
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
    else:
        if st.session_state.user_role == "doctor":
            doctor_dashboard()
        else:
            patient_dashboard()

if __name__ == "__main__":
    main()