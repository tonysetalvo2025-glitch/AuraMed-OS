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

# --- CSS AVANÇADO (CORREÇÃO DE VISIBILIDADE DE INPUTS & ESTILO ENTERPRISE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700&display=swap');
    
    html, body, .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #f0f2f5 !important;
        color: #1e293b !important;
    }

    /* FORÇAR TEXTO ESCURO GERAL */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, span, div, label, li {
        color: #1e293b !important;
    }

    /* CORREÇÃO CRÍTICA DE INPUTS (TEXTO DIGITADO) */
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div, .stDateInput input {
        color: #1e293b !important;          
        -webkit-text-fill-color: #1e293b !important;
        caret-color: #0d9488 !important;    
        background-color: #ffffff !important; 
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.1) !important;
    }

    /* Sidebar Estilizada */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] * {
        color: #334155 !important;
    }

    /* Cards Flutuantes (Glassmorphism Light) */
    .glass-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
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
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.2);
    }
    .stButton > button p { color: white !important; -webkit-text-fill-color: white !important; }

    /* Badge de Status */
    .status-badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .status-ok { background-color: #dcfce7; color: #166534 !important; }
    .status-warning { background-color: #fef9c3; color: #854d0e !important; }
    .status-danger { background-color: #fee2e2; color: #991b1b !important; }
    
    /* Tabelas */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Receita Médica (Visualização de Papel) */
    .paper-sheet {
        background: #fff;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        padding: 40px;
        min-height: 500px;
        border: 1px solid #ddd;
        font-family: 'Times New Roman', serif;
    }

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

# --- INIT SESSION STATE COM DADOS ROBUSTOS (MOCK) ---
if "data" not in st.session_state:
    st.session_state.data = {
        "credentials": {
            "admin": {"senha": "admin", "role": "doctor", "nome": "Dr. Gênesis", "especialidade": "Clínica Geral", "crm": "12345-SP"},
            "ana": {"senha": "123", "role": "patient", "nome": "Ana Silva"},
            "carlos": {"senha": "123", "role": "patient", "nome": "Carlos Souza"}
        },
        "pacientes": [
            {
                "id": 1, "nome": "Ana Silva", "idade": 32, "sexo": "F", 
                "historico": "Enxaqueca crônica (CID G43). Alergia a Dipirona.",
                "vitals": {"pressao": [120, 122, 118, 130, 120], "datas": ["Jan", "Fev", "Mar", "Abr", "Mai"]},
                "timeline": [
                    {"data": "2023-01-15", "evento": "Consulta Inicial", "detalhe": "Queixa de dor de cabeça."},
                    {"data": "2023-02-10", "evento": "Exame Laboratorial", "detalhe": "Hemograma completo - Normal."},
                    {"data": "2023-05-20", "evento": "Retorno", "detalhe": "Ajuste de medicação."}
                ]
            },
            {
                "id": 2, "nome": "Carlos Souza", "idade": 45, "sexo": "M", 
                "historico": "Hipertensão (CID I10). Uso contínuo de Losartana.",
                "vitals": {"pressao": [140, 138, 135, 142, 130], "datas": ["Jan", "Fev", "Mar", "Abr", "Mai"]},
                "timeline": [
                    {"data": "2023-03-01", "evento": "Consulta Cardíaca", "detalhe": "PA elevada."},
                    {"data": "2023-03-05", "evento": "MAPA 24h", "detalhe": "Solicitado."}
                ]
            },
        ],
        "appointments": [
            {"id": 101, "paciente": "Ana Silva", "data": "2023-11-20", "hora": "14:00", "tipo": "Retorno", "status": "Confirmado", "valor": 350.00},
            {"id": 102, "paciente": "Carlos Souza", "data": "2023-11-20", "hora": "15:00", "tipo": "Primeira Vez", "status": "Pendente", "valor": 500.00},
            {"id": 103, "paciente": "Mariana Lima", "data": "2023-11-20", "hora": "16:00", "tipo": "Retorno", "status": "Em Atendimento", "valor": 350.00}
        ],
        "financeiro": {
            "meses": ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun"],
            "receita": [12500, 15000, 13200, 18000, 16500, 19200],
            "despesas": [5000, 5200, 4800, 6000, 5500, 5800]
        },
        "medicamentos": ["Amoxicilina 500mg", "Dipirona 1g", "Losartana 50mg", "Omeprazol 20mg", "Ibuprofeno 600mg", "Rivotril 0.5mg"]
    }
elif "credentials" not in st.session_state.data:
    st.session_state.data["credentials"] = {
        "admin": {"senha": "admin", "role": "doctor", "nome": "Dr. Gênesis", "especialidade": "Clínica Geral", "crm": "12345-SP"},
        "ana": {"senha": "123", "role": "patient", "nome": "Ana Silva"}
    }

# --- 3. CORE INTELLIGENCE & UTILS ---

def ai_structure_soap(raw_notes):
    """IA para estruturar Prontuário"""
    if not st.session_state.get("groq_client"): return "⚠️ Erro: IA Offline"
    sys_prompt = "Você é um assistente médico. Converta as notas em formato SOAP (Subjetivo, Objetivo, Avaliação, Plano) profissional."
    try:
        completion = st.session_state.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_notes}],
            temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erro ao processar: {e}"

def plot_finance_chart():
    """Gera gráfico financeiro avançado"""
    fin = st.session_state.data['financeiro']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fin['meses'], y=fin['receita'], name='Receita', marker_color='#0d9488'))
    fig.add_trace(go.Bar(x=fin['meses'], y=fin['despesas'], name='Despesas', marker_color='#ef4444'))
    fig.update_layout(barmode='group', title="Fluxo de Caixa Semestral", plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'))
    return fig

# --- 4. INTERFACE E MÓDULOS ---

def sidebar_nav():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #0d9488 !important;'>AuraMed OS <span style='font-size:0.5em'>ENT</span></h2>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio("Módulos", ["Dashboard", "Prontuário IA", "Receituário", "Pacientes", "Financeiro"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown(f"**{st.session_state.user_name}**<br><span style='font-size:0.8em; color:gray'>CRM: {st.session_state.data['credentials']['admin'].get('crm', 'N/A')}</span>", unsafe_allow_html=True)
        if st.button("Sair (Logout)", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
        return menu

def page_doctor_dashboard():
    st.markdown("### ⚡ Command Center")
    
    # Notificações
    st.info("🔔 **Notificação:** Exames de Carlos Souza disponíveis para análise. (Há 2 horas)")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    total_dia = sum([a['valor'] for a in st.session_state.data['appointments']])
    k1.markdown(f"<div class='glass-card'><h4>{len(st.session_state.data['appointments'])}</h4><span class='status-badge status-ok'>Agendamentos</span></div>", unsafe_allow_html=True)
    k2.markdown("<div class='glass-card'><h4>3</h4><span class='status-badge status-warning'>Em Espera</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='glass-card'><h4>R$ {total_dia:.2f}</h4><span class='status-badge status-ok'>Faturamento Dia</span></div>", unsafe_allow_html=True)
    k4.markdown("<div class='glass-card'><h4>4.9/5</h4><span class='status-badge status-ok'>NPS Clínica</span></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Fluxo de Pacientes Hoje")
        df = pd.DataFrame(st.session_state.data['appointments'])
        st.dataframe(
            df[['hora', 'paciente', 'tipo', 'status', 'valor']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "status": st.column_config.SelectboxColumn("Status", options=["Confirmado", "Pendente", "Em Atendimento", "Finalizado"]),
                "valor": st.column_config.NumberColumn("Valor", format="R$ %.2f")
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Ações Rápidas")
        if st.button("➕ Novo Agendamento", use_container_width=True):
            st.toast("Módulo de Agendamento Aberto")
        if st.button("📞 Chamar Próximo", use_container_width=True):
            st.success("Chamando: Carlos Souza - Sala 02")
        st.markdown("</div>", unsafe_allow_html=True)

def page_magic_prontuario():
    st.markdown("### ✨ Prontuário Inteligente (IA)")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        pat = st.selectbox("Paciente", [p['nome'] for p in st.session_state.data['pacientes']])
        raw = st.text_area("Notas Clínicas (Dite ou digite)", height=250, placeholder="Paciente relata cefaleia frontal há 3 dias...")
        if st.button("Processar Prontuário"):
            if raw:
                with st.spinner("Estruturando dados..."):
                    st.session_state.generated_soap = ai_structure_soap(f"Paciente: {pat}. Notas: {raw}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        if "generated_soap" in st.session_state:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Documento Gerado")
            st.markdown(st.session_state.generated_soap)
            st.button("Assinar e Salvar", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

def page_prescription():
    st.markdown("### 💊 Receituário & Atestados")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        pat = st.selectbox("Selecione o Paciente", [p['nome'] for p in st.session_state.data['pacientes']], key="presc_pat")
        meds = st.multiselect("Medicamentos", st.session_state.data['medicamentos'])
        obs = st.text_area("Instruções de Uso", "Tomar 1 comprimido a cada 8 horas por 5 dias.")
        add_atestado = st.checkbox("Gerar Atestado Médico")
        dias_afastamento = 0
        if add_atestado:
            dias_afastamento = st.number_input("Dias de afastamento", min_value=1, value=1)
        
        gerar = st.button("Gerar Documento Oficial", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        if gerar:
            st.markdown(f"""
            <div class='paper-sheet'>
                <div style='text-align:center; border-bottom: 2px solid #333; padding-bottom:10px; margin-bottom:20px;'>
                    <h2>CLÍNICA AURAMED</h2>
                    <small>Rua da Saúde, 1000 - Centro | Tel: (11) 99999-9999</small>
                </div>
                <p><strong>Paciente:</strong> {pat}</p>
                <p><strong>Data:</strong> {datetime.date.today().strftime('%d/%m/%Y')}</p>
                <hr>
                <h4>USO INTERNO/ORAL</h4>
                <ul>
                    {''.join([f'<li style="margin-bottom:10px;"><b>{m}</b><br>{obs}</li>' for m in meds])}
                </ul>
                {f"<hr><h4>ATESTADO</h4><p>Atesto para os devidos fins que o paciente necessita de <b>{dias_afastamento} dias</b> de afastamento.</p>" if add_atestado else ""}
                <br><br><br>
                <div style='text-align:center;'>
                    <p>___________________________________</p>
                    <p>{st.session_state.data['credentials']['admin']['nome']}</p>
                    <p>{st.session_state.data['credentials']['admin']['crm']}</p>
                    <p style='color:#0d9488; font-size:0.8em;'>Assinado Digitalmente via AuraMed OS</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

def page_financial():
    st.markdown("### 💰 Gestão Financeira")
    
    # Top Stats
    f1, f2, f3 = st.columns(3)
    rec_total = sum(st.session_state.data['financeiro']['receita'])
    desp_total = sum(st.session_state.data['financeiro']['despesas'])
    lucro = rec_total - desp_total
    
    f1.metric("Receita Semestral", f"R$ {rec_total:,.2f}")
    f2.metric("Despesas", f"R$ {desp_total:,.2f}", delta="-12%", delta_color="inverse")
    f3.metric("Lucro Líquido", f"R$ {lucro:,.2f}", delta="Bom")
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.plotly_chart(plot_finance_chart(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def page_patient_list():
    st.markdown("### 📂 Prontuário Eletrônico (Timeline)")
    search = st.text_input("Buscar paciente...", placeholder="Nome, CPF ou Telefone")
    
    filtered = [p for p in st.session_state.data['pacientes'] if search.lower() in p['nome'].lower()] if search else st.session_state.data['pacientes']
    
    for p in filtered:
        with st.expander(f"👤 {p['nome']} | Última Visita: {p['timeline'][-1]['data']}"):
            t1, t2 = st.tabs(["Timeline Clínica", "Dados Vitais"])
            with t1:
                for event in reversed(p.get('timeline', [])):
                    st.markdown(f"""
                    <div style='border-left: 2px solid #e2e8f0; padding-left: 15px; margin-bottom: 20px;'>
                        <small style='color:#0d9488; font-weight:bold;'>{event['data']}</small>
                        <h5 style='margin:0;'>{event['evento']}</h5>
                        <p style='margin:0; color:#64748b;'>{event['detalhe']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            with t2:
                # Simples gráfico de linha usando plotly express
                fig = px.line(x=p['vitals']['datas'], y=p['vitals']['pressao'], markers=True, title="Evolução Pressão Arterial")
                fig.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

# --- 5. PAINEL DO PACIENTE (NOVO) ---
def page_patient_dashboard():
    st.markdown(f"## Olá, {st.session_state.user_name}")
    
    p_data = next((p for p in st.session_state.data['pacientes'] if p['nome'] == st.session_state.user_name), None)
    
    if p_data:
        c1, c2 = st.columns([2, 1])
        with c1:
             st.markdown("<div class='glass-card'><h4>📈 Meus Sinais Vitais</h4>", unsafe_allow_html=True)
             fig = px.line(x=p_data['vitals']['datas'], y=p_data['vitals']['pressao'], markers=True)
             fig.update_layout(plot_bgcolor='white', height=300, margin=dict(l=20, r=20, t=20, b=20))
             st.plotly_chart(fig, use_container_width=True)
             st.markdown("</div>", unsafe_allow_html=True)
             
             st.markdown("<div class='glass-card'><h4>🗓️ Minha Timeline</h4>", unsafe_allow_html=True)
             for event in reversed(p_data.get('timeline', [])):
                    st.markdown(f"""
                    <div style='border-left: 3px solid #0d9488; padding-left: 15px; margin-bottom: 15px;'>
                        <b>{event['data']}</b> - {event['evento']}<br>
                        <span style='color:gray'>{event['detalhe']}</span>
                    </div>
                    """, unsafe_allow_html=True)
             st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Próxima Consulta")
            # Mock de consulta futura
            st.info("📅 15/12/2025 às 14:00\n\nDr. Gênesis - Clínica Geral")
            st.button("Reagendar", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Bem-vindo ao AuraMed. Seus dados clínicos aparecerão aqui após a primeira consulta.")

    if st.sidebar.button("Sair", key="logout_pat"):
        st.session_state.logged_in = False
        st.rerun()

# --- 6. LOGIN ---
def login_screen():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: #0d9488 !important;'>AuraMed OS</h1>", unsafe_allow_html=True)
        st.markdown("<p>Enterprise Edition</p>", unsafe_allow_html=True)
        
        tab_login, tab_create = st.tabs(["Acesso Profissional/Paciente", "Cadastrar"])
        
        with tab_login:
            user = st.text_input("Usuário", key="l_user")
            pwd = st.text_input("Senha", type="password", key="l_pwd")
            role = st.radio("Perfil", ["Médico(a)", "Paciente"], horizontal=True)
            
            if st.button("Entrar", use_container_width=True):
                creds = st.session_state.data.get("credentials", {})
                u_data = creds.get(user)
                if u_data and u_data['senha'] == pwd:
                    st.session_state.logged_in = True
                    st.session_state.user_role = u_data['role']
                    st.session_state.user_name = u_data['nome']
                    st.rerun()
                else:
                    st.error("Dados incorretos.")

        with tab_create:
            new_u = st.text_input("Novo Usuário")
            new_p = st.text_input("Nova Senha", type="password")
            new_n = st.text_input("Nome Completo")
            # NOVA OPÇÃO DE TIPO DE CONTA
            new_role_sel = st.selectbox("Tipo de Conta", ["Paciente", "Médico(a)"])
            
            if st.button("Criar Conta"):
                if new_u:
                    role_code = "doctor" if new_role_sel == "Médico(a)" else "patient"
                    st.session_state.data["credentials"][new_u] = {"senha": new_p, "role": role_code, "nome": new_n}
                    
                    if role_code == "patient":
                        st.session_state.data["pacientes"].append({"id": 99, "nome": new_n, "idade": 0, "sexo": "-", "historico": "Novo", "vitals": {"pressao":[], "datas":[]}, "timeline": []})
                    
                    st.success(f"Conta de {new_role_sel} criada! Faça login.")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; color:gray; font-size:0.8em;'>Demo: admin/admin | ana/123</div>", unsafe_allow_html=True)

# --- 7. ROUTER ---
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_screen()
    else:
        if st.session_state.user_role == "doctor":
            page = sidebar_nav()
            if page == "Dashboard": page_doctor_dashboard()
            elif page == "Prontuário IA": page_magic_prontuario()
            elif page == "Receituário": page_prescription()
            elif page == "Pacientes": page_patient_list()
            elif page == "Financeiro": page_financial()
        else:
            # Chama o novo Dashboard Completo do Paciente
            page_patient_dashboard()

if __name__ == "__main__":
    main()
