import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import time
import logging
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. CONFIGURAÇÃO E CSS "NEXT-GEN" ---
st.set_page_config(page_title="AuraMed OS", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

logging.basicConfig(level=logging.INFO)

# --- CSS AVANÇADO ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700&display=swap');
    
    html, body, .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #f0f2f5 !important;
    }

    /* Tipografia e Cores de Alto Contraste */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, span, div {
        color: #1e293b !important;
    }

    /* Sidebar Estilizada */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Cards Flutuantes (Glassmorphism Light) */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Botões de Ação Principal */
    .stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    .stButton > button p { color: white !important; }

    /* Inputs Modernos */
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        background-color: #f8fafc;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0d9488;
        box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.1);
    }

    /* Badge de Status */
    .status-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .status-ok { background-color: #dcfce7; color: #166534 !important; }
    .status-warning { background-color: #fef9c3; color: #854d0e !important; }
    
    /* Remoção de ruído visual */
    #MainMenu, footer, header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- 2. SETUP DE DADOS E IA ---
api_key = st.secrets.get("GROQ_API_KEY")
if "groq_client" not in st.session_state and api_key:
    st.session_state.groq_client = Groq(api_key=api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = load_embedding_model()

# --- INIT SESSION STATE COM DADOS ROBUSTOS ---
if "data" not in st.session_state:
    st.session_state.data = {
        "credentials": {
            "admin": {"senha": "admin", "role": "doctor", "nome": "Dr. Gênesis", "especialidade": "Clínica Geral"},
            "ana": {"senha": "123", "role": "patient", "nome": "Ana Silva"},
            "carlos": {"senha": "123", "role": "patient", "nome": "Carlos Souza"}
        },
        "pacientes": [
            {
                "id": 1, "nome": "Ana Silva", "idade": 32, "sexo": "F", 
                "historico": "Enxaqueca crônica (CID G43). Alergia a Dipirona.",
                "vitals": {"pressao": [120, 122, 118, 130, 120], "datas": ["Jan", "Fev", "Mar", "Abr", "Mai"]}
            },
            {
                "id": 2, "nome": "Carlos Souza", "idade": 45, "sexo": "M", 
                "historico": "Hipertensão (CID I10). Uso contínuo de Losartana.",
                "vitals": {"pressao": [140, 138, 135, 142, 130], "datas": ["Jan", "Fev", "Mar", "Abr", "Mai"]}
            },
        ],
        "appointments": [
            {"id": 101, "paciente": "Ana Silva", "data": "2023-11-20", "hora": "14:00", "tipo": "Retorno", "status": "Confirmado"},
            {"id": 102, "paciente": "Carlos Souza", "data": "2023-11-20", "hora": "15:00", "tipo": "Primeira Vez", "status": "Pendente"},
        ]
    }
elif "credentials" not in st.session_state.data:
    # Fallback de segurança para hot-reload
    st.session_state.data["credentials"] = {
        "admin": {"senha": "admin", "role": "doctor", "nome": "Dr. Gênesis"},
        "ana": {"senha": "123", "role": "patient", "nome": "Ana Silva"}
    }

# --- 3. CORE INTELLIGENCE (FUNÇÕES AVANÇADAS) ---

def ai_structure_soap(raw_notes):
    """Transforma anotações bagunçadas em Prontuário SOAP estruturado"""
    if not st.session_state.get("groq_client"): return "⚠️ Erro: IA Offline"
    
    sys_prompt = """
    Você é um assistente médico especialista em documentação clínica.
    Sua tarefa: Converter anotações brutas em formato S.O.A.P. (Subjetivo, Objetivo, Avaliação, Plano).
    - Extraia sintomas, sinais vitais, diagnósticos prováveis e conduta.
    - Sugira o código CID-10 se possível.
    - Use formatação Markdown profissional com negrito e listas.
    - Seja conciso e técnico.
    """
    try:
        completion = st.session_state.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_notes}],
            temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erro ao processar: {e}"

def plot_vitals(patient_data):
    """Gera gráfico de sinais vitais"""
    dates = patient_data['vitals']['datas']
    values = patient_data['vitals']['pressao']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name='PAS (Sistólica)', line=dict(color='#0d9488', width=3)))
    fig.add_trace(go.Scatter(x=dates, y=[80]*len(dates), mode='lines', name='Meta', line=dict(color='#94a3b8', dash='dash')))
    
    fig.update_layout(
        title="Histórico de Pressão Arterial",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        font=dict(color='#1e293b')
    )
    return fig

# --- 4. INTERFACE DO SISTEMA ---

def sidebar_nav():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #0d9488 !important;'>AuraMed OS <span style='font-size:0.5em'>PRO</span></h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        menu = st.radio("Navegação", ["Dashboard", "Prontuário Inteligente", "Pacientes", "Configurações"], label_visibility="collapsed")
        
        st.markdown("---")
        st.info(f"Usuário: **{st.session_state.user_name}**")
        if st.button("Sair (Logout)", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
        
        return menu

def page_doctor_dashboard():
    st.markdown("### ⚡ Painel de Controle")
    
    # KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='glass-card'><h4>4</h4><span class='status-badge status-ok'>Pacientes Hoje</span></div>", unsafe_allow_html=True)
    k2.markdown("<div class='glass-card'><h4>15min</h4><span class='status-badge status-warning'>Tempo Médio</span></div>", unsafe_allow_html=True)
    k3.markdown("<div class='glass-card'><h4>98%</h4><span class='status-badge status-ok'>Eficiência IA</span></div>", unsafe_allow_html=True)
    k4.markdown("<div class='glass-card'><h4>R$ 2.4k</h4><span class='status-badge status-ok'>Receita Est.</span></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Agenda em Tempo Real")
        df = pd.DataFrame(st.session_state.data['appointments'])
        st.dataframe(
            df[['hora', 'paciente', 'tipo', 'status']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "status": st.column_config.SelectboxColumn("Status", options=["Confirmado", "Pendente", "Cancelado", "Em Atendimento"])
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Assistente Rápido")
        q = st.text_input("Pergunta clínica rápida (ex: dose amoxicilina pediatria)")
        if q:
            with st.spinner("Consultando bases..."):
                resp = ai_structure_soap(f"Apenas responda a dúvida médica de forma sucinta: {q}")
                st.success(resp)
        st.markdown("</div>", unsafe_allow_html=True)

def page_magic_prontuario():
    st.markdown("### ✨ Prontuário Mágico (IA Generativa)")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 1. Entrada de Dados (Rascunho)")
        patient_select = st.selectbox("Selecione o Paciente", [p['nome'] for p in st.session_state.data['pacientes']])
        raw_input = st.text_area("Digite anotações soltas (ex: 'dor cabeça forte 3 dias, vomito, pa 140/90, sem febre')", height=200)
        
        if st.button("🪄 Processar com IA"):
            if raw_input:
                with st.spinner("Aura está estruturando o caso clínico..."):
                    structured_note = ai_structure_soap(f"Paciente: {patient_select}. Notas: {raw_input}")
                    st.session_state.generated_soap = structured_note
            else:
                st.warning("Digite algo para a IA processar.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        if "generated_soap" in st.session_state:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 2. Prontuário Estruturado (S.O.A.P)")
            st.markdown(st.session_state.generated_soap)
            col_act1, col_act2 = st.columns(2)
            col_act1.button("💾 Salvar no Histórico", use_container_width=True)
            col_act2.button("🖨️ Imprimir Receita", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

def page_patient_list():
    st.markdown("### 📂 Base de Pacientes")
    
    search = st.text_input("Buscar paciente por nome ou condição...", placeholder="Ex: Hipertensão")
    
    # Lógica de busca simples + vetorial (simulada na busca textual simples aqui para robustez)
    filtered = [p for p in st.session_state.data['pacientes'] if search.lower() in p['nome'].lower() or search.lower() in p['historico'].lower()] if search else st.session_state.data['pacientes']
    
    for p in filtered:
        with st.expander(f"👤 {p['nome']} | {p['idade']} anos"):
            c_info, c_chart = st.columns([1, 2])
            with c_info:
                st.write(f"**Histórico:** {p['historico']}")
                st.write(f"**Sexo:** {p['sexo']}")
                st.button("Ver Prontuário Completo", key=f"btn_{p['id']}")
            with c_chart:
                st.plotly_chart(plot_vitals(p), use_container_width=True, config={'displayModeBar': False})

# --- 5. LÓGICA DE LOGIN (MANTIDA MAS ESTILIZADA) ---
def login_screen():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: #0d9488 !important;'>AuraMed OS</h1>", unsafe_allow_html=True)
        st.markdown("<p>Acesso Seguro</p>", unsafe_allow_html=True)
        
        tab_login, tab_create = st.tabs(["Entrar", "Criar Conta"])
        
        with tab_login:
            user = st.text_input("Usuário", key="l_user")
            pwd = st.text_input("Senha", type="password", key="l_pwd")
            role = st.radio("Perfil", ["Médico(a)", "Paciente"], horizontal=True)
            
            if st.button("Acessar Sistema", use_container_width=True):
                creds = st.session_state.data.get("credentials", {})
                u_data = creds.get(user)
                
                valid = False
                if u_data and u_data['senha'] == pwd:
                    if (role == "Médico(a)" and u_data['role'] == "doctor") or (role == "Paciente" and u_data['role'] == "patient"):
                        valid = True
                
                if valid:
                    st.session_state.logged_in = True
                    st.session_state.user_role = u_data['role']
                    st.session_state.user_name = u_data['nome']
                    st.rerun()
                else:
                    st.error("Acesso negado. Verifique as credenciais.")

        with tab_create:
            # Lógica simplificada de cadastro para manter o código limpo
            new_u = st.text_input("Novo Usuário")
            new_p = st.text_input("Nova Senha", type="password")
            new_n = st.text_input("Nome Completo")
            
            if st.button("Registrar"):
                if new_u and new_p:
                    st.session_state.data["credentials"][new_u] = {"senha": new_p, "role": "patient", "nome": new_n}
                    st.success("Conta criada! Faça login.")
                    st.session_state.data["pacientes"].append({"id": 99, "nome": new_n, "idade": 30, "sexo": "N/A", "historico": "Novo", "vitals": {"pressao": [], "datas": []}})

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; color:gray; font-size:0.8em;'>Demo: admin/admin (Médico) | ana/123 (Paciente)</div>", unsafe_allow_html=True)

# --- 6. ROTEADOR PRINCIPAL ---
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_screen()
    else:
        # Se for médico, mostra sidebar avançada
        if st.session_state.user_role == "doctor":
            page = sidebar_nav()
            if page == "Dashboard": page_doctor_dashboard()
            elif page == "Prontuário Inteligente": page_magic_prontuario()
            elif page == "Pacientes": page_patient_list()
            else: st.info("Configurações em desenvolvimento.")
        else:
            # View Simples do Paciente
            st.markdown(f"## Olá, {st.session_state.user_name}")
            st.markdown("<div class='glass-card'><h3>Meus Sinais Vitais</h3>", unsafe_allow_html=True)
            # Busca dados do paciente logado
            meus_dados = next((p for p in st.session_state.data['pacientes'] if p['nome'] == st.session_state.user_name), None)
            if meus_dados:
                st.plotly_chart(plot_vitals(meus_dados), use_container_width=True)
            else:
                st.info("Sem dados vitais registrados.")
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("Sair"):
                st.session_state.logged_in = False
                st.rerun()

if __name__ == "__main__":
    main()
