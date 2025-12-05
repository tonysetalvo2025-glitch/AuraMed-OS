import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import time
import logging
# Bibliotecas de IA
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. CONFIGURAÇÃO INICIAL E LOGGING ---
st.set_page_config(
    page_title="AuraMed OS",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuração de Logging para monitorar acessos no console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuraMed")

# --- 2. GESTÃO DE SEGREDOS E SETUP DE IA ---
# Tenta pegar a chave do st.secrets (configurar no Streamlit Cloud)
api_key = st.secrets.get("GROQ_API_KEY")

# Inicializa cliente Groq se a chave existir
if "groq_client" not in st.session_state and api_key:
    st.session_state.groq_client = Groq(api_key=api_key)

# Carregamento de Modelo de Embedding (Cacheado para performance)
@st.cache_resource
def load_embedding_model():
    # Modelo leve e eficiente para rodar em CPU gratuita
    return SentenceTransformer('all-MiniLM-L6-v2')

if "embedding_model" not in st.session_state:
    with st.spinner("Inicializando núcleo neural Aura..."):
        st.session_state.embedding_model = load_embedding_model()

# --- 3. CSS PROFISSIONAL (CORREÇÃO DE VISIBILIDADE) ---
st.markdown("""
<style>
    /* Importação de Fontes Modernas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    /* RESET GLOBAL DE CORES - CRÍTICO PARA EVITAR TEXTO INVISÍVEL */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa !important; /* Fundo cinza muito claro */
    }

    /* Forçar cor cinza escuro em TODOS os textos para contraste */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, .stMarkdown, .stText {
        color: #1a202c !important;
    }

    /* Cards Estilo Neumorfismo (Sombra Suave) */
    .metric-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #eef0f2;
        transition: transform 0.2s;
        margin-bottom: 15px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    /* Inputs (Caixas de texto) */
    .stTextInput > div > div > input {
        color: #1a202c !important;
        background-color: #ffffff !important;
        border-color: #e2e8f0;
    }
    
    /* Labels de Inputs (Usuário, Senha) */
    .stTextInput > label, .stRadio > label {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }

    /* Botões Customizados (Gradiente Teal) */
    .stButton > button {
        background: linear-gradient(135deg, #319795 0%, #2C7A7B 100%) !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(49, 151, 149, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #38B2AC 0%, #319795 100%) !important;
        box-shadow: 0 6px 8px rgba(49, 151, 149, 0.3);
        transform: translateY(-1px);
    }
    /* Garante que o texto DENTRO do botão seja branco */
    .stButton > button p {
        color: white !important; 
    }
    
    /* Estilo Específico das Mensagens do Chat */
    .chat-user {
        background-color: #E6FFFA;
        padding: 15px;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 10px;
        text-align: right;
        color: #234E52 !important;
        border: 1px solid #B2F5EA;
        float: right;
        clear: both;
        max-width: 80%;
    }
    .chat-ai {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
        color: #2D3748 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        float: left;
        clear: both;
        max-width: 80%;
    }
    
    /* Esconder elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- 4. DADOS MOCKADOS (Base de Conhecimento e Login) ---
if "data" not in st.session_state:
    st.session_state.data = {
        # --- NOVO: BANCO DE USUÁRIOS E SENHAS ---
        "credentials": {
            # Médicos
            "admin": {"senha": "admin", "role": "doctor", "nome": "Dr. Gênesis"},
            # Pacientes (Login: user / Senha: 123)
            "ana": {"senha": "123", "role": "patient", "nome": "Ana Silva"},
            "carlos": {"senha": "123", "role": "patient", "nome": "Carlos Souza"},
            "mariana": {"senha": "123", "role": "patient", "nome": "Mariana Lima"}
        },
        "pacientes": [
            {"id": 1, "nome": "Ana Silva", "idade": 32, "historico": "Enxaqueca crônica, alergia a penicilina. Relata estresse no trabalho.", "ultima_consulta": "2023-10-15"},
            {"id": 2, "nome": "Carlos Souza", "idade": 45, "historico": "Hipertensão leve, monitoramento de colesterol. Pratica atividade física regular.", "ultima_consulta": "2023-11-02"},
            {"id": 3, "nome": "Mariana Lima", "idade": 28, "historico": "Gestante 12 semanas, exames de rotina normais. Leve enjoo matinal.", "ultima_consulta": "2023-11-10"},
        ],
        "appointments": [
            {"paciente": "Ana Silva", "data": "2023-11-20", "hora": "14:00", "status": "Confirmado"},
            {"paciente": "Carlos Souza", "data": "2023-11-20", "hora": "15:00", "status": "Pendente"},
        ]
    }
# --- FIX: GARANTIA DE ESTRUTURA PARA EVITAR KEYERROR EM HOT-RELOAD ---
elif "credentials" not in st.session_state.data:
    st.session_state.data["credentials"] = {
        "admin": {"senha": "admin", "role": "doctor", "nome": "Dr. Gênesis"},
        "ana": {"senha": "123", "role": "patient", "nome": "Ana Silva"},
        "carlos": {"senha": "123", "role": "patient", "nome": "Carlos Souza"},
        "mariana": {"senha": "123", "role": "patient", "nome": "Mariana Lima"}
    }

# --- 5. LÓGICA DE IA (GROQ + EMBEDDINGS) ---

def get_groq_response(system_prompt, user_prompt):
    """Chama a API Groq para gerar texto"""
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
        return "⚠️ Aura indisponível no momento. Verifique a conexão."

def semantic_search_patients(query):
    """Busca vetorial local para encontrar pacientes por contexto"""
    if not query:
        return []
    
    model = st.session_state.embedding_model
    # Cria "documentos" baseados no nome e histórico
    docs = [f"{p['nome']} - {p['historico']}" for p in st.session_state.data['pacientes']]
    
    query_emb = model.encode([query])
    doc_embs = model.encode(docs)
    
    # Calcula similaridade
    scores = cosine_similarity(query_emb, doc_embs)[0]
    results = []
    
    # Filtra resultados relevantes (score > 0.2)
    for idx, score in enumerate(scores):
        if score > 0.2:
            results.append((st.session_state.data['pacientes'][idx], score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results]

# --- 6. TELA DE LOGIN ---
def login():
    st.markdown("<div style='text-align: center; margin-top: 50px; margin-bottom: 30px;'><h1 style='color:#2C7A7B !important; font-size: 3rem;'>AuraMed OS</h1><p style='color:#718096 !important;'>Acesse seu espaço clínico inteligente</p></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            
            # --- ABAS DE LOGIN E CADASTRO ---
            tab_login, tab_register = st.tabs(["Login", "Criar Conta"])
            
            # --- ABA LOGIN ---
            with tab_login:
                user_type = st.radio("Eu sou:", ["Médico(a)", "Paciente"], horizontal=True, key="login_type")
                username = st.text_input("Usuário", key="login_user")
                password = st.text_input("Senha", type="password", key="login_pass")
                
                if st.button("Entrar no Sistema", use_container_width=True):
                    # Busca usuário no "banco de dados" de credenciais
                    users_db = st.session_state.data.get("credentials", {})
                    user_data = users_db.get(username)

                    if user_data:
                        # Verifica senha e tipo de perfil
                        if user_data["senha"] == password:
                            role_match = False
                            # Valida se o perfil selecionado bate com o do banco
                            if user_type == "Médico(a)" and user_data["role"] == "doctor":
                                role_match = True
                            elif user_type == "Paciente" and user_data["role"] == "patient":
                                role_match = True
                            
                            if role_match:
                                st.session_state.logged_in = True
                                st.session_state.user_role = user_data["role"]
                                st.session_state.user_name = user_data["nome"]
                                st.success(f"Bem-vindo(a), {user_data['nome']}!")
                                time.sleep(1) # Feedback visual antes do reload
                                st.rerun()
                            else:
                                st.error(f"Este usuário não tem perfil de {user_type}.")
                        else:
                            st.error("Senha incorreta.")
                    else:
                        st.error("Usuário não encontrado.")
            
            # --- ABA CADASTRO ---
            with tab_register:
                st.markdown("### Novo Cadastro")
                new_name = st.text_input("Nome Completo", key="reg_name")
                new_user = st.text_input("Escolha um Usuário", key="reg_user")
                new_pass = st.text_input("Escolha uma Senha", type="password", key="reg_pass")
                new_type = st.radio("Tipo de Conta:", ["Paciente", "Médico(a)"], horizontal=True, key="reg_type")
                
                # Validação extra para médicos
                admin_key = ""
                if new_type == "Médico(a)":
                    admin_key = st.text_input("Chave de Licença Médica (Simulação: 'crm123')", type="password", key="reg_key")
                
                if st.button("Cadastrar", use_container_width=True):
                    users_db = st.session_state.data.get("credentials", {})
                    
                    if new_user in users_db:
                        st.error("Este nome de usuário já existe.")
                    elif not new_user or not new_pass or not new_name:
                        st.warning("Preencha todos os campos.")
                    else:
                        # Lógica de validação e cadastro
                        valid_register = True
                        role_code = "patient"
                        
                        if new_type == "Médico(a)":
                            if admin_key == "crm123":
                                role_code = "doctor"
                            else:
                                st.error("Chave de licença inválida.")
                                valid_register = False
                        
                        if valid_register:
                            # Salva no estado da sessão
                            st.session_state.data["credentials"][new_user] = {
                                "senha": new_pass,
                                "role": role_code,
                                "nome": new_name
                            }
                            st.success("Cadastro realizado! Faça login na aba ao lado.")
                            # Se for paciente, cria um registro vazio para evitar erros
                            if role_code == "patient":
                                new_id = len(st.session_state.data["pacientes"]) + 1
                                st.session_state.data["pacientes"].append({
                                    "id": new_id,
                                    "nome": new_name,
                                    "idade": 0, 
                                    "historico": "Novo cadastro. Histórico pendente.",
                                    "ultima_consulta": "Nunca"
                                })

            st.markdown("</div>", unsafe_allow_html=True)
            
            # Dica para teste (remover em produção real)
            st.info("💡 **Dica de Teste:**\n\n**Médico:** admin / admin\n\n**Pacientes:** ana / 123 | carlos / 123")
            
            if not api_key:
                st.warning("⚠️ Configure a GROQ_API_KEY no .streamlit/secrets.toml")

# --- 7. DASHBOARD MÉDICO ---
def doctor_dashboard():
    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"<h1 style='color:#2C7A7B !important;'>Olá, {st.session_state.user_name}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#718096 !important;'>Resumo do dia e inteligência clínica</p>", unsafe_allow_html=True)
    with col_h2:
        if st.button("Sair"):
            st.session_state.logged_in = False
            st.rerun()

    # Cards de KPI
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><h3 style='margin:0; font-size: 2rem;'>{len(st.session_state.data['appointments'])}</h3><p style='margin:0; color:#718096 !important;'>Consultas Hoje</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h3 style='margin:0; font-size: 2rem;'>3</h3><p style='margin:0; color:#718096 !important;'>Pacientes Críticos</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><h3 style='margin:0; font-size: 2rem;'>98%</h3><p style='margin:0; color:#718096 !important;'>Satisfação (NPS)</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><h3 style='margin:0; font-size: 2rem;'>R$ 12k</h3><p style='margin:0; color:#718096 !important;'>Faturamento Mês</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Layout Principal: Agenda vs IA
    col_main, col_ai = st.columns([2, 1])

    with col_main:
        st.subheader("🗓️ Agenda do Dia")
        df_app = pd.DataFrame(st.session_state.data['appointments'])
        # Exibe tabela estilizada
        st.dataframe(df_app, use_container_width=True, hide_index=True)

        st.subheader("🔍 Busca Semântica de Prontuários")
        st.markdown("<p style='font-size:0.9em; color:#666 !important;'>Encontre pacientes descrevendo sintomas ou contextos (ex: 'dor de cabeça', 'gestante')</p>", unsafe_allow_html=True)
        search_query = st.text_input("Busca inteligente:", placeholder="Digite aqui...")
        
        if search_query:
            results = semantic_search_patients(search_query)
            if results:
                for p in results:
                    with st.expander(f"👤 {p['nome']} (Idade: {p['idade']})"):
                        st.markdown(f"**Histórico:** {p['historico']}")
                        st.markdown(f"**Última Consulta:** {p['ultima_consulta']}")
                        # Botão para ativar análise de IA
                        if st.button(f"Analisar {p['nome']} com IA", key=p['id']):
                            st.session_state.analyze_patient = p
            else:
                st.info("Nenhum paciente encontrado com esses critérios.")

    with col_ai:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🤖 Aura Clinical Copilot")
        
        # Área de Análise de Paciente Selecionado
        if "analyze_patient" in st.session_state:
            p = st.session_state.analyze_patient
            st.info(f"Paciente em foco: {p['nome']}")
            if st.button("Gerar Plano Clínico"):
                with st.spinner("Aura está analisando protocolos..."):
                    prompt = f"Analise este paciente: {p}. Sugira: 1. Hipóteses diagnósticas baseadas no histórico. 2. Exames recomendados. 3. Orientações preventivas."
                    response = get_groq_response("Você é um assistente médico sênior. Responda em Markdown, de forma estruturada.", prompt)
                    st.markdown(response)
        
        # Chat Geral (Quick Consult)
        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
        st.markdown("**Quick Consult (Segunda Opinião)**")
        q = st.text_input("Dúvida médica rápida:")
        if q and st.button("Consultar Aura"):
            with st.spinner("Pesquisando na base médica..."):
                resp = get_groq_response("Você é uma IA médica auxiliar. Responda com base em protocolos médicos padrão. Seja concisa.", q)
                st.success(resp)
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- 8. PORTAL DO PACIENTE ---
def patient_dashboard():
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("<h1 style='color:#2C7A7B !important;'>Seu Espaço de Saúde</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#718096 !important;'>Bem-vindo(a), {st.session_state.user_name}</p>", unsafe_allow_html=True)
    with col_h2:
        if st.button("Sair"):
            st.session_state.logged_in = False
            st.rerun()
    
    # Abas para separar consultas de triagem
    tab1, tab2 = st.tabs(["Minhas Consultas", "Triagem Inteligente"])
    
    with tab1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Próximos Agendamentos")
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
        st.markdown("### 💬 Converse com a Aura (Triagem)")
        st.markdown("<small style='color: #E53E3E !important;'>⚠️ Em caso de emergência, não use este sistema. Ligue 192.</small>", unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Container do chat com mensagens anteriores
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                role_class = "chat-user" if msg["role"] == "user" else "chat-ai"
                st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
                # Div vazio para limpar o float do CSS
                st.markdown("<div style='clear: both;'></div>", unsafe_allow_html=True)

        user_input = st.text_input("Descreva o que está sentindo:", key="patient_input")
        
        if st.button("Enviar"):
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("Aura está analisando..."):
                    sys_prompt = """
                    Você é a Aura, uma IA de triagem clínica empática. 
                    1. Nunca dê diagnósticos definitivos (diga "pode ser X").
                    2. Sugira a especialidade médica adequada para agendar.
                    3. Se parecer grave (dor no peito, falta de ar), instrua ir ao hospital imediatamente.
                    4. Seja breve, acolhedora e humana.
                    """
                    response = get_groq_response(sys_prompt, user_input)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        if st.button("Limpar Conversa"):
            st.session_state.chat_history = []
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

# --- 9. ROTEAMENTO PRINCIPAL ---
def main():
    # Inicializa estado de login se não existir
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
