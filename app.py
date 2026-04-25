from flask import Flask, request, jsonify, session, redirect
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import warnings, os, sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "inventario_secret_2024"

# ══════════════════════════════════════════════════════
#  ★  CONFIGURACIÓN — EDITA ESTOS DATOS  ★
# ══════════════════════════════════════════════════════
RUTA_EXCEL = os.path.join(os.path.dirname(__file__), "Inventario_fullhd.xlsx")

USUARIOS = {
    "admin": "admin123",
    "juanfe": "supermercado2024",
}

# Correo que ENVÍA las alertas (debe ser Gmail)
EMAIL_REMITENTE  = "jcruzc12@ucentral.edu.co"
EMAIL_CONTRASENA = "1234"   # contraseña de app de Google

# Correo(s) que RECIBEN las alertas (puedes poner varios separados por coma)
EMAIL_DESTINATARIOS = ["juanfecruz2021@gmail.com"]

# Días de stock mínimo para disparar alerta
UMBRAL_CRITICO   = 3   # días → alerta ROJA
UMBRAL_PRECAUCION = 7  # días → alerta AMARILLA
# ══════════════════════════════════════════════════════

# ──────────────────────────────────────────
# CARGA Y ENTRENAMIENTO
# ──────────────────────────────────────────
def cargar_y_entrenar(ruta):
    if not os.path.exists(ruta):
        print(f"❌ No se encontró: {ruta}")
        sys.exit(1)
    print("🧠 Cargando datos y entrenando modelo...")
    df = pd.read_excel(ruta).fillna(0)
    df['Producto_Limpio'] = df['Producto'].astype(str).str.strip()
    lista = sorted(df['Producto_Limpio'].unique().tolist())
    for col in ['Stock Inicial','Entradas','Salidas','Costo Unitario','PVP']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    dm = pd.get_dummies(df, columns=['Categoría','Marca'], drop_first=True)
    X  = dm.drop(columns=['Fecha','Código SKU','Producto','Producto_Limpio','Salidas','Stock Final'], errors='ignore')
    y  = dm['Salidas']
    m  = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(X, y)
    print(f"✅ Listo. {len(lista)} productos cargados.")
    return df, X, m, lista

df, X, modelo_rf, lista_productos = cargar_y_entrenar(RUTA_EXCEL)

# ──────────────────────────────────────────
# GENERAR ALERTAS
# ──────────────────────────────────────────
def generar_alertas():
    alertas = []
    for nombre in lista_productos:
        try:
            mask  = df['Producto_Limpio'] == nombre
            idx   = df[mask].index[-1]
            stock = int(df.loc[idx,'Stock Inicial']) + int(df.loc[idx,'Entradas'])
            pred  = int(np.round(modelo_rf.predict(X.loc[[idx]])[0]))
            if pred <= 0:
                continue
            dias_r = round(stock / pred, 1)
            pedir  = max(0, pred * 30 - stock)
            if dias_r < UMBRAL_PRECAUCION:
                alertas.append({
                    "producto": nombre,
                    "sku":      str(df.loc[idx,'Código SKU']),
                    "stock":    stock,
                    "dias":     dias_r,
                    "pedir":    int(pedir),
                    "nivel":    "critico" if dias_r < UMBRAL_CRITICO else "precaucion",
                })
        except:
            continue
    alertas.sort(key=lambda x: x["dias"])
    return alertas

alertas_globales = generar_alertas()
print(f"🔔 {len(alertas_globales)} productos requieren atención.")

# ──────────────────────────────────────────
# ENVÍO DE CORREO
# ──────────────────────────────────────────
ultimo_envio = {"estado": "nunca", "mensaje": ""}

def construir_html_correo(alertas):
    criticos   = [a for a in alertas if a["nivel"] == "critico"]
    precaucion = [a for a in alertas if a["nivel"] == "precaucion"]

    def filas(lista, color, icono):
        html = ""
        for a in lista:
            html += f"""
            <tr>
              <td style="padding:10px 14px;border-bottom:1px solid #1e2d40">
                <strong style="color:#e2e8f0">{a['producto']}</strong><br>
                <span style="color:#64748b;font-size:11px">SKU: {a['sku']}</span>
              </td>
              <td style="padding:10px 14px;border-bottom:1px solid #1e2d40;text-align:center">
                <span style="background:{color}20;color:{color};border:1px solid {color}40;
                             border-radius:6px;padding:3px 10px;font-size:12px;font-weight:700;white-space:nowrap">
                  {icono} {a['dias']} días
                </span>
              </td>
              <td style="padding:10px 14px;border-bottom:1px solid #1e2d40;text-align:center;
                         color:#e2e8f0;font-family:monospace">{a['stock']}</td>
              <td style="padding:10px 14px;border-bottom:1px solid #1e2d40;text-align:center;
                         color:#fbbf24;font-family:monospace;font-weight:700">{a['pedir']}</td>
            </tr>"""
        return html

    seccion_criticos = ""
    if criticos:
        seccion_criticos = f"""
        <div style="margin-bottom:24px">
          <h3 style="color:#f87171;font-size:15px;margin:0 0 12px 0">🔴 Alertas Críticas ({len(criticos)} productos)</h3>
          <table style="width:100%;border-collapse:collapse;background:#0d1117;border-radius:10px;overflow:hidden">
            <thead><tr style="background:#1a0a0a">
              <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.6px">Producto</th>
              <th style="padding:10px 14px;text-align:center;color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.6px">Días Restantes</th>
              <th style="padding:10px 14px;text-align:center;color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.6px">Stock Actual</th>
              <th style="padding:10px 14px;text-align:center;color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.6px">Sugerido Pedir</th>
            </tr></thead>
            <tbody>{filas(criticos,'#f87171','🔴')}</tbody>
          </table>
        </div>"""

    seccion_precaucion = ""
    if precaucion:
        seccion_precaucion = f"""
        <div style="margin-bottom:24px">
          <h3 style="color:#fbbf24;font-size:15px;margin:0 0 12px 0">🟡 Precaución ({len(precaucion)} productos)</h3>
          <table style="width:100%;border-collapse:collapse;background:#0d1117;border-radius:10px;overflow:hidden">
            <thead><tr style="background:#1a1400">
              <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.6px">Producto</th>
              <th style="padding:10px 14px;text-align:center;color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.6px">Días Restantes</th>
              <th style="padding:10px 14px;text-align:center;color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.6px">Stock Actual</th>
              <th style="padding:10px 14px;text-align:center;color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.6px">Sugerido Pedir</th>
            </tr></thead>
            <tbody>{filas(precaucion,'#fbbf24','🟡')}</tbody>
          </table>
        </div>"""

    return f"""
    <html><body style="margin:0;padding:0;background:#070c14;font-family:'Inter',Arial,sans-serif">
    <div style="max-width:640px;margin:32px auto;background:#0d1524;border:1px solid #1e2d40;border-radius:16px;overflow:hidden">

      <!-- Header -->
      <div style="background:linear-gradient(135deg,#0d1524,#111d2e);padding:28px 32px;border-bottom:1px solid #1e2d40;position:relative">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#22c55e,transparent)"></div>
        <table style="width:100%"><tr>
          <td><span style="font-size:24px">📦</span></td>
          <td style="padding-left:12px">
            <h1 style="color:#e2e8f0;font-size:18px;font-weight:700;margin:0">Alerta de Inventario</h1>
            <p style="color:#64748b;font-size:13px;margin:4px 0 0">Sistema Predictivo · Reporte Automático</p>
          </td>
          <td style="text-align:right">
            <span style="background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.25);
                         border-radius:20px;padding:4px 12px;font-size:12px;font-weight:700">
              ⚠ {len(alertas)} productos en riesgo
            </span>
          </td>
        </tr></table>
      </div>

      <!-- Body -->
      <div style="padding:28px 32px">
        <p style="color:#94a3b8;font-size:14px;margin:0 0 24px 0;line-height:1.6">
          El sistema de IA detectó productos que requieren reabastecimiento urgente.
          A continuación el detalle:
        </p>
        {seccion_criticos}
        {seccion_precaucion}
      </div>

      <!-- Footer -->
      <div style="padding:16px 32px;border-top:1px solid #1e2d40;text-align:center">
        <p style="color:#334155;font-size:11.5px;margin:0">
          Sistema Predictivo de Inventario &nbsp;·&nbsp; Powered by Random Forest AI<br>
          Este es un correo automático, no respondas a este mensaje.
        </p>
      </div>
    </div>
    </body></html>"""

def enviar_correo_alertas(alertas, destinatarios):
    global ultimo_envio
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🔴 Alerta Inventario — {len(alertas)} productos en riesgo"
        msg["From"]    = EMAIL_REMITENTE
        msg["To"]      = ", ".join(destinatarios)

        # Texto plano de respaldo
        texto_plano = f"ALERTA DE INVENTARIO — {len(alertas)} productos requieren atención:\n\n"
        for a in alertas:
            emoji = "🔴" if a["nivel"] == "critico" else "🟡"
            texto_plano += f"{emoji} {a['producto']} — {a['dias']} días restantes — Pedir: {a['pedir']} unidades\n"

        msg.attach(MIMEText(texto_plano, "plain"))
        msg.attach(MIMEText(construir_html_correo(alertas), "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(EMAIL_REMITENTE, EMAIL_CONTRASENA)
            servidor.sendmail(EMAIL_REMITENTE, destinatarios, msg.as_string())

        ultimo_envio = {"estado": "ok", "mensaje": f"Correo enviado a {', '.join(destinatarios)}"}
        print(f"✅ Alerta enviada a: {destinatarios}")

    except Exception as e:
        ultimo_envio = {"estado": "error", "mensaje": str(e)}
        print(f"❌ Error enviando correo: {e}")

# ──────────────────────────────────────────
# HTML LOGIN
# ──────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html lang="es"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inventario IA · Acceso</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#070c14;min-height:100vh;display:flex;align-items:center;justify-content:center;overflow:hidden}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:40px 40px;pointer-events:none}
body::after{content:'';position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);width:600px;height:600px;background:radial-gradient(circle,rgba(34,197,94,.06) 0%,transparent 70%);pointer-events:none}
.card{background:#0d1524;border:1px solid #1e2d40;border-radius:20px;padding:48px 44px;width:100%;max-width:420px;position:relative;z-index:1;box-shadow:0 25px 60px rgba(0,0,0,.5);animation:up .4s ease}
@keyframes up{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.card::before{content:'';position:absolute;top:0;left:10%;right:10%;height:1px;background:linear-gradient(90deg,transparent,#22c55e,transparent)}
.logo{text-align:center;margin-bottom:32px}
.logo-icon{width:60px;height:60px;background:linear-gradient(135deg,#166534,#15803d);border-radius:16px;display:inline-flex;align-items:center;justify-content:center;font-size:28px;margin-bottom:16px;box-shadow:0 8px 24px rgba(34,197,94,.2)}
.logo h1{color:#f1f5f9;font-size:20px;font-weight:700}.logo p{color:#475569;font-size:13px;margin-top:4px}
.field{margin-bottom:18px}
.field label{display:block;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
.field input{width:100%;background:#070c14;border:1px solid #1e2d40;border-radius:10px;color:#e2e8f0;font-family:'Inter',sans-serif;font-size:14px;padding:12px 16px;transition:border-color .2s,box-shadow .2s;outline:none}
.field input:focus{border-color:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,.12)}
.field input::placeholder{color:#334155}
.error{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:8px;color:#f87171;font-size:12.5px;padding:10px 14px;margin-bottom:18px}
.btn{width:100%;background:linear-gradient(135deg,#16a34a,#22c55e);border:none;border-radius:10px;color:#fff;cursor:pointer;font-family:'Inter',sans-serif;font-size:14px;font-weight:600;padding:13px;transition:all .2s;margin-top:6px}
.btn:hover{background:linear-gradient(135deg,#22c55e,#4ade80);box-shadow:0 6px 20px rgba(34,197,94,.3);transform:translateY(-1px)}
.hint{margin-top:24px;padding-top:20px;border-top:1px solid #1e2d40;text-align:center}
.hint p{color:#334155;font-size:11.5px;line-height:1.6}
code{font-family:'JetBrains Mono',monospace;color:#22c55e;font-size:11px;background:rgba(34,197,94,.07);padding:1px 5px;border-radius:4px}
</style></head><body>
<div class="card">
  <div class="logo"><div class="logo-icon">📦</div><h1>Inventario IA</h1><p>Sistema Predictivo de Stock</p></div>
  {error_block}
  <form method="POST" action="/login">
    <div class="field"><label>Usuario</label><input type="text" name="usuario" placeholder="Ingresa tu usuario" required></div>
    <div class="field"><label>Contraseña</label><input type="password" name="clave" placeholder="••••••••••" required></div>
    <button type="submit" class="btn">Iniciar sesión →</button>
  </form>
  <div class="hint"><p>Demo: <code>admin</code> / <code>admin123</code></p></div>
</div></body></html>"""

# ──────────────────────────────────────────
# HTML DASHBOARD
# ──────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inventario IA · Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#070c14;--surface:#0d1524;--surface2:#111d2e;--border:#1e2d40;--border2:#243447;--green:#22c55e;--green-d:#16a34a;--text:#e2e8f0;--muted:#64748b;--dim:#334155}
html,body{height:100%}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;min-height:100vh}
.topbar{background:var(--surface);border-bottom:1px solid var(--border);padding:0 32px;height:58px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.topbar-left{display:flex;align-items:center;gap:12px}
.topbar-logo{width:34px;height:34px;background:linear-gradient(135deg,var(--green-d),var(--green));border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px}
.topbar-title{font-size:15px;font-weight:700;color:var(--text);letter-spacing:-.2px}
.topbar-right{display:flex;align-items:center;gap:12px}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);border-radius:20px;padding:4px 12px;font-size:11.5px;font-weight:500;color:var(--green)}
.pill-dot{width:6px;height:6px;background:var(--green);border-radius:50%;animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.user-badge{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}
.avatar{width:30px;height:30px;background:var(--surface2);border:1px solid var(--border);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;color:var(--green);font-weight:600}
.logout{background:transparent;border:1px solid var(--border);border-radius:7px;color:var(--muted);cursor:pointer;font-size:12px;padding:5px 12px;transition:all .15s;text-decoration:none;font-family:'Inter',sans-serif}
.logout:hover{border-color:#ef4444;color:#f87171}
.bell-btn{position:relative;background:transparent;border:1px solid var(--border);border-radius:9px;color:var(--muted);cursor:pointer;font-size:16px;padding:6px 10px;transition:all .2s;line-height:1}
.bell-btn:hover{border-color:var(--green);color:var(--green)}
.bell-badge{position:absolute;top:-6px;right:-6px;background:#ef4444;color:#fff;font-size:10px;font-weight:700;border-radius:10px;padding:1px 5px;min-width:18px;text-align:center;animation:pop .3s ease}
@keyframes pop{from{transform:scale(0)}to{transform:scale(1)}}
.alert-panel{position:fixed;top:66px;right:24px;width:420px;background:var(--surface);border:1px solid var(--border);border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.7);z-index:200;display:none;animation:slideIn .25s ease}
.alert-panel.open{display:block}
@keyframes slideIn{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
.ap-header{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border)}
.ap-header h3{font-size:14px;font-weight:700;color:var(--text)}
.ap-close{background:none;border:none;color:var(--muted);cursor:pointer;font-size:18px;padding:0 4px}
.ap-close:hover{color:var(--text)}
.ap-body{max-height:400px;overflow-y:auto;padding:12px}
.ap-body::-webkit-scrollbar{width:4px}
.ap-body::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
.alert-item{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:13px;margin-bottom:8px}
.alert-item.critico{border-left:3px solid #ef4444}
.alert-item.precaucion{border-left:3px solid #fbbf24}
.ai-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.ai-name{font-size:12.5px;font-weight:600;color:var(--text);flex:1;margin-right:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ai-badge{font-size:10px;font-weight:700;border-radius:6px;padding:2px 8px}
.ai-badge.critico{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.2)}
.ai-badge.precaucion{background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.2)}
.ai-info{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
.ai-stat{text-align:center;background:var(--surface);border-radius:6px;padding:6px 4px}
.s-val{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--text)}
.s-val.red{color:#f87171}.s-val.yellow{color:#fbbf24}
.s-lbl{font-size:9.5px;color:var(--dim);margin-top:2px;text-transform:uppercase;letter-spacing:.5px}
.ap-footer{padding:12px 16px;border-top:1px solid var(--border)}
.btn-send-email{width:100%;background:linear-gradient(135deg,#7c3aed,#8b5cf6);border:none;border-radius:9px;color:#fff;cursor:pointer;font-family:'Inter',sans-serif;font-size:13px;font-weight:600;padding:10px;transition:all .2s}
.btn-send-email:hover{background:linear-gradient(135deg,#8b5cf6,#a78bfa);box-shadow:0 4px 14px rgba(139,92,246,.3);transform:translateY(-1px)}
.btn-send-email:disabled{opacity:.5;cursor:not-allowed;transform:none}
.ap-empty{text-align:center;padding:40px 20px;color:var(--dim)}
.ap-empty .em-icon{font-size:32px;margin-bottom:10px}
.toast{position:fixed;bottom:28px;right:28px;padding:13px 20px;border-radius:10px;font-size:13.5px;font-weight:600;z-index:999;animation:toastIn .3s ease;display:none}
.toast.ok{background:#166534;border:1px solid #22c55e;color:#4ade80}
.toast.error{background:#7f1d1d;border:1px solid #ef4444;color:#f87171}
@keyframes toastIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.tabs{display:flex;gap:4px;margin-bottom:20px;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:4px}
.tab{flex:1;background:transparent;border:none;border-radius:7px;color:var(--muted);cursor:pointer;font-family:'Inter',sans-serif;font-size:13px;font-weight:500;padding:9px;transition:all .2s;text-align:center}
.tab.active{background:var(--surface2);color:var(--text);font-weight:600;box-shadow:0 1px 4px rgba(0,0,0,.3)}
.tab-content{display:none}.tab-content.active{display:block}
main{flex:1;padding:28px 32px 32px;max-width:1280px;width:100%;margin:0 auto}
.page-header{margin-bottom:24px}
.page-header h2{font-size:22px;font-weight:700;color:var(--text);letter-spacing:-.4px}
.page-header p{font-size:13px;color:var(--muted);margin-top:4px}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;transition:border-color .2s}
.kpi:hover{border-color:var(--border2)}
.kpi-label{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:10px}
.kpi-value{font-size:26px;font-weight:700;color:var(--text);letter-spacing:-.5px;font-family:'JetBrains Mono',monospace}
.kpi-value.green{color:var(--green)}.kpi-value.yellow{color:#fbbf24}.kpi-value.red{color:#f87171}
.kpi-sub{font-size:11.5px;color:var(--dim);margin-top:4px}
.panel-grid{display:grid;grid-template-columns:360px 1fr;gap:16px;margin-bottom:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:24px}
.card-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--muted);padding-bottom:14px;margin-bottom:20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}
.field{margin-bottom:20px}
.field label{display:block;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin-bottom:8px}
.field select{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:9px;color:var(--text);font-family:'Inter',sans-serif;font-size:13.5px;padding:11px 14px;outline:none;transition:border-color .2s,box-shadow .2s;appearance:none;-webkit-appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2364748b' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 14px center;padding-right:36px}
.field select:focus{border-color:var(--green);box-shadow:0 0 0 3px rgba(34,197,94,.1)}
option{background:#0d1524}
.slider-value{text-align:center;font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:600;color:var(--green);margin-bottom:8px}
input[type=range]{width:100%;appearance:none;-webkit-appearance:none;height:4px;background:var(--border2);border-radius:4px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;background:var(--green);border-radius:50%;box-shadow:0 0 8px rgba(34,197,94,.4);cursor:pointer}
.slider-labels{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);margin-top:6px}
.btn-primary{width:100%;background:linear-gradient(135deg,var(--green-d),var(--green));border:none;border-radius:10px;color:#fff;cursor:pointer;font-family:'Inter',sans-serif;font-size:14px;font-weight:600;padding:13px;transition:all .2s;margin-top:4px}
.btn-primary:hover{background:linear-gradient(135deg,var(--green),#4ade80);box-shadow:0 6px 20px rgba(34,197,94,.3);transform:translateY(-1px)}
.btn-primary:active{transform:translateY(0)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed;transform:none}
.alert-banner{border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:12px;margin-bottom:14px;font-size:13.5px;font-weight:600;opacity:0;transition:opacity .3s}
.alert-banner.show{opacity:1}
.alert-banner.seguro{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);color:#4ade80}
.alert-banner.precaucion{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.25);color:#fbbf24}
.alert-banner.critico{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);color:#f87171}
.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.result-box{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px}
.rb-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:6px}
.rb-value{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--text);line-height:1.65}
.empty-state{text-align:center;padding:48px 20px;color:var(--dim)}
.empty-state .es-icon{font-size:40px;margin-bottom:12px}
.empty-state p{font-size:13px}
.notif-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:24px}
.mono-area{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:10px;color:#a5b4c8;font-family:'JetBrains Mono',monospace;font-size:12.5px;line-height:1.75;padding:16px;resize:none;outline:none;min-height:140px}
.copy-row{display:flex;justify-content:flex-end;margin-top:6px}
.copy-btn{background:var(--surface2);border:1px solid var(--border);border-radius:7px;color:var(--muted);cursor:pointer;font-size:11.5px;padding:5px 12px;transition:all .15s;font-family:'Inter',sans-serif}
.copy-btn:hover{background:var(--green);color:#fff;border-color:var(--green)}
.spinner{display:none;width:18px;height:18px;border:2px solid rgba(255,255,255,.2);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;margin:0 auto}
@keyframes spin{to{transform:rotate(360deg)}}
.info-box{background:var(--bg);border:1px solid var(--border);border-left:3px solid var(--green);border-radius:9px;padding:14px 16px;margin-top:16px}
.info-box p{color:var(--muted);font-size:11.5px;margin:0;line-height:1.65}
.alerts-table-wrap{overflow-x:auto}
.alerts-table{width:100%;border-collapse:collapse;font-size:13px}
.alerts-table th{font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);padding:10px 14px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
.alerts-table td{padding:12px 14px;border-bottom:1px solid #0f1923;color:var(--text);vertical-align:middle}
.alerts-table tr:last-child td{border-bottom:none}
.alerts-table tr:hover td{background:rgba(255,255,255,.02)}
.badge{font-size:10.5px;font-weight:700;border-radius:6px;padding:3px 9px;white-space:nowrap}
.badge.critico{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.25)}
.badge.precaucion{background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.25)}
.dias-val{font-family:'JetBrains Mono',monospace;font-weight:700}
.dias-val.red{color:#f87171}.dias-val.yellow{color:#fbbf24}
footer{border-top:1px solid var(--border);padding:14px 32px;text-align:center;font-size:11.5px;color:var(--dim)}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
@media(max-width:900px){.panel-grid{grid-template-columns:1fr}.kpi-grid{grid-template-columns:1fr 1fr}.result-grid{grid-template-columns:1fr}main{padding:20px 16px}.topbar{padding:0 16px}.alert-panel{width:calc(100vw - 32px);right:16px}}
</style></head><body>

<!-- TOPBAR -->
<header class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">📦</div>
    <span class="topbar-title">Inventario IA</span>
  </div>
  <div class="topbar-right">
    <div class="pill"><div class="pill-dot"></div> Sistema activo</div>
    <button class="bell-btn" onclick="toggleAlertas()" title="Ver alertas">
      🔔{badge_html}
    </button>
    <div class="user-badge">
      <div class="avatar">{avatar}</div>
      <span>{usuario}</span>
    </div>
    <a href="/logout" class="logout">Cerrar sesión</a>
  </div>
</header>

<!-- PANEL ALERTAS -->
<div class="alert-panel" id="alertPanel">
  <div class="ap-header">
    <h3>🔔 Alertas de Inventario</h3>
    <button class="ap-close" onclick="toggleAlertas()">✕</button>
  </div>
  <div class="ap-body">{alertas_html}</div>
  <div class="ap-footer">
    <button class="btn-send-email" id="btnEmail" onclick="enviarCorreo()">
      📧 Enviar alerta por correo ahora
    </button>
    <p id="ultimo-envio" style="color:var(--muted);font-size:11px;text-align:center;margin-top:8px">{ultimo_envio_txt}</p>
  </div>
</div>

<!-- TOAST -->
<div class="toast" id="toast"></div>

<main>
  <div class="page-header">
    <h2>Panel de Control Predictivo</h2>
    <p>Analiza el stock y proyecta la demanda con inteligencia artificial · Random Forest</p>
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-label">Total Productos</div><div class="kpi-value green">{total_productos}</div><div class="kpi-sub">en el inventario</div></div>
    <div class="kpi"><div class="kpi-label">Alertas Críticas</div><div class="kpi-value red">{criticos}</div><div class="kpi-sub">menos de 3 días de stock</div></div>
    <div class="kpi"><div class="kpi-label">En Precaución</div><div class="kpi-value yellow">{precauciones}</div><div class="kpi-sub">menos de 7 días de stock</div></div>
    <div class="kpi"><div class="kpi-label">Stock Analizado</div><div class="kpi-value" id="kpi-stock">—</div><div class="kpi-sub">unidades disponibles</div></div>
  </div>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('analisis',this)">📊 Análisis Individual</button>
    <button class="tab" onclick="switchTab('alertas',this)">⚠️ Tabla de Alertas <span style="background:rgba(239,68,68,.15);color:#f87171;border-radius:10px;padding:1px 7px;font-size:11px;margin-left:4px">{total_alertas}</span></button>
  </div>

  <div class="tab-content active" id="tab-analisis">
    <div class="panel-grid">
      <div class="card">
        <div class="card-title">⚙ Configuración del análisis</div>
        <div class="field"><label>Producto</label><select id="producto">{opciones}</select></div>
        <div class="field">
          <label>Horizonte de proyección</label>
          <div class="slider-value" id="slider-label">30 días</div>
          <input type="range" id="dias" min="7" max="90" value="30" oninput="document.getElementById('slider-label').textContent=this.value+' días'">
          <div class="slider-labels"><span>7 días</span><span>90 días</span></div>
        </div>
        <button class="btn-primary" id="btn" onclick="analizar()">
          <span id="btn-text">⚡ Analizar y Calcular Pedido</span>
          <div class="spinner" id="spinner"></div>
        </button>
        <div class="info-box"><p>🤖 El modelo analiza el historial de ventas por categoría y marca para proyectar la demanda y calcular el reabastecimiento óptimo.</p></div>
      </div>
      <div class="card">
        <div class="card-title">📊 Resultados del análisis</div>
        <div class="alert-banner" id="alert-banner"><span id="alert-icon" style="font-size:20px"></span><span id="alert-text"></span></div>
        <div id="result-area">
          <div class="empty-state"><div class="es-icon">📈</div><p>Selecciona un producto y presiona<br><strong>Analizar</strong> para ver los resultados.</p></div>
        </div>
      </div>
    </div>
    <div class="notif-card" id="notif-card" style="display:none">
      <div class="card-title">🔔 Notificación de orden de compra</div>
      <textarea class="mono-area" id="correo" readonly rows="7"></textarea>
      <div class="copy-row"><button class="copy-btn" onclick="copiar()">📋 Copiar correo</button></div>
    </div>
  </div>

  <div class="tab-content" id="tab-alertas">
    <div class="card">
      <div class="card-title">⚠️ Productos que requieren atención inmediata</div>
      {tabla_alertas}
    </div>
  </div>
</main>

<footer>Sistema Predictivo de Inventario &nbsp;·&nbsp; Powered by Random Forest AI</footer>

<script>
document.addEventListener('click',function(e){
  const p=document.getElementById('alertPanel');
  if(p.classList.contains('open')&&!p.contains(e.target)&&!e.target.closest('.bell-btn')) p.classList.remove('open');
});
function toggleAlertas(){document.getElementById('alertPanel').classList.toggle('open');}
function switchTab(n,b){
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+n).classList.add('active');b.classList.add('active');
}
function showToast(msg,tipo){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='toast '+tipo;t.style.display='block';
  setTimeout(()=>{t.style.display='none';},4000);
}
async function enviarCorreo(){
  const btn=document.getElementById('btnEmail');
  btn.disabled=true;btn.textContent='⏳ Enviando...';
  try{
    const res=await fetch('/enviar-alertas',{method:'POST',headers:{'Content-Type':'application/json'}});
    const d=await res.json();
    if(d.ok){
      showToast('✅ '+d.mensaje,'ok');
      document.getElementById('ultimo-envio').textContent='Último envío: ahora mismo';
    } else {
      showToast('❌ Error: '+d.mensaje,'error');
    }
  }catch(e){showToast('❌ Error de conexión','error');}
  finally{btn.disabled=false;btn.textContent='📧 Enviar alerta por correo ahora';}
}
async function analizar(){
  const btn=document.getElementById('btn'),txt=document.getElementById('btn-text'),spin=document.getElementById('spinner');
  btn.disabled=true;txt.style.display='none';spin.style.display='block';
  try{
    const res=await fetch('/predecir',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({producto:document.getElementById('producto').value,dias:parseInt(document.getElementById('dias').value)})});
    const d=await res.json();
    if(d.error){alert(d.error);return;}
    document.getElementById('kpi-stock').textContent=d.stock_total.toLocaleString();
    const icons={seguro:'🟢',precaucion:'🟡',critico:'🔴'};
    const labels={seguro:'Stock Seguro',precaucion:'Precaución — Stock bajo',critico:'Riesgo Crítico — Reabastece urgente'};
    const b=document.getElementById('alert-banner');
    b.className='alert-banner show '+d.nivel;
    document.getElementById('alert-icon').textContent=icons[d.nivel];
    document.getElementById('alert-text').textContent=labels[d.nivel];
    document.getElementById('result-area').innerHTML=`
      <div class="result-grid">
        <div class="result-box"><div class="rb-label">Producto</div><div class="rb-value">${d.producto}<br><span style="color:var(--muted)">SKU: ${d.sku}</span></div></div>
        <div class="result-box"><div class="rb-label">Categoría</div><div class="rb-value">${d.categoria}</div></div>
        <div class="result-box"><div class="rb-label">Salidas diarias (IA)</div><div class="rb-value" style="color:var(--green);font-size:20px;font-weight:700">${d.prediccion_diaria} <span style="font-size:13px;color:var(--muted)">und/día</span></div></div>
        <div class="result-box"><div class="rb-label">Ventas proyectadas (${d.dias_proyeccion}d)</div><div class="rb-value" style="font-size:20px;font-weight:700">${d.ventas_proyectadas.toLocaleString()} <span style="font-size:13px;color:var(--muted)">unidades</span></div></div>
      </div>
      ${d.cantidad_pedir>0
        ?`<div style="margin-top:14px;padding:14px 18px;background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.2);border-radius:10px;color:#fbbf24;font-size:13.5px;font-weight:600;">🛒 Cantidad sugerida a pedir: <span style="font-size:18px">${d.cantidad_pedir.toLocaleString()} unidades</span></div>`
        :`<div style="margin-top:14px;padding:14px 18px;background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.2);border-radius:10px;color:#4ade80;font-size:13.5px;font-weight:600;">✅ Stock suficiente para los próximos ${d.dias_proyeccion} días.</div>`}`;
    document.getElementById('correo').value=d.correo;
    document.getElementById('notif-card').style.display='block';
  }catch(e){alert('Error: '+e.message);}
  finally{btn.disabled=false;txt.style.display='inline';spin.style.display='none';}
}
function copiar(){
  const ta=document.getElementById('correo');ta.select();document.execCommand('copy');
  const b=document.querySelector('.copy-btn');b.textContent='✅ Copiado';
  setTimeout(()=>b.textContent='📋 Copiar correo',1800);
}
</script>
</body></html>"""

# ──────────────────────────────────────────
# HELPERS HTML
# ──────────────────────────────────────────
def build_alertas_html(alertas):
    if not alertas:
        return '<div class="ap-empty"><div class="em-icon">✅</div><p>Todos los productos tienen stock suficiente.</p></div>'
    items = ""
    for a in alertas:
        c = "red" if a["nivel"]=="critico" else "yellow"
        items += f"""<div class="alert-item {a['nivel']}">
          <div class="ai-top"><span class="ai-name" title="{a['producto']}">{a['producto']}</span>
          <span class="ai-badge {a['nivel']}">{'🔴 CRÍTICO' if a['nivel']=='critico' else '🟡 PRECAUCIÓN'}</span></div>
          <div class="ai-info">
            <div class="ai-stat"><div class="s-val {c}">{a['dias']}</div><div class="s-lbl">días restantes</div></div>
            <div class="ai-stat"><div class="s-val">{a['stock']}</div><div class="s-lbl">en stock</div></div>
            <div class="ai-stat"><div class="s-val yellow">{a['pedir']}</div><div class="s-lbl">sugerido pedir</div></div>
          </div></div>"""
    return items

def build_tabla_alertas(alertas):
    if not alertas:
        return '<div class="ap-empty"><div class="em-icon">✅</div><p>Todos los productos tienen stock suficiente para los próximos 7 días.</p></div>'
    filas = ""
    for a in alertas:
        c = "red" if a["nivel"]=="critico" else "yellow"
        filas += f"""<tr>
          <td><strong>{a['producto']}</strong><br><span style="color:var(--muted);font-size:11px">{a['sku']}</span></td>
          <td><span class="badge {a['nivel']}">{'🔴 Crítico' if a['nivel']=='critico' else '🟡 Precaución'}</span></td>
          <td style="font-family:'JetBrains Mono',monospace">{a['stock']}</td>
          <td><span class="dias-val {c}">{a['dias']} días</span></td>
          <td style="font-family:'JetBrains Mono',monospace;color:#fbbf24;font-weight:600">{a['pedir']}</td></tr>"""
    return f"""<div class="alerts-table-wrap"><table class="alerts-table">
      <thead><tr><th>Producto</th><th>Estado</th><th>Stock Actual</th><th>Días Restantes</th><th>Sugerido Pedir (30d)</th></tr></thead>
      <tbody>{filas}</tbody></table></div>"""

# ──────────────────────────────────────────
# RUTAS FLASK
# ──────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    if "usuario" not in session:
        return redirect("/login")
    usuario   = session["usuario"]
    criticos  = sum(1 for a in alertas_globales if a["nivel"]=="critico")
    precauc   = sum(1 for a in alertas_globales if a["nivel"]=="precaucion")
    total_al  = len(alertas_globales)
    badge_html = f'<span class="bell-badge">{total_al}</span>' if total_al>0 else ""
    opciones  = "".join(f'<option value="{p}">{p}</option>' for p in lista_productos)
    envio_txt = f"Último envío: {ultimo_envio['mensaje']}" if ultimo_envio["estado"]!="nunca" else "Sin envíos previos en esta sesión"
    html = DASHBOARD_HTML \
        .replace("{opciones}", opciones) \
        .replace("{usuario}", usuario) \
        .replace("{avatar}", usuario[0].upper()) \
        .replace("{total_productos}", str(len(lista_productos))) \
        .replace("{criticos}", str(criticos)) \
        .replace("{precauciones}", str(precauc)) \
        .replace("{total_alertas}", str(total_al)) \
        .replace("{badge_html}", badge_html) \
        .replace("{alertas_html}", build_alertas_html(alertas_globales)) \
        .replace("{tabla_alertas}", build_tabla_alertas(alertas_globales)) \
        .replace("{ultimo_envio_txt}", envio_txt)
    return html

@app.route("/login", methods=["GET","POST"])
def login():
    error_block = ""
    if request.method == "POST":
        usuario = request.form.get("usuario","").strip()
        clave   = request.form.get("clave","").strip()
        if usuario in USUARIOS and USUARIOS[usuario]==clave:
            session["usuario"] = usuario
            return redirect("/")
        error_block = '<div class="error">⚠ Usuario o contraseña incorrectos.</div>'
    return LOGIN_HTML.replace("{error_block}", error_block)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/enviar-alertas", methods=["POST"])
def enviar_alertas():
    if "usuario" not in session:
        return jsonify({"ok": False, "mensaje": "No autorizado"}), 401
    if not alertas_globales:
        return jsonify({"ok": True, "mensaje": "No hay alertas que enviar."})
    if EMAIL_REMITENTE == "tucorreo@gmail.com":
        return jsonify({"ok": False, "mensaje": "Configura tu correo en app.py (EMAIL_REMITENTE y EMAIL_CONTRASENA)"})
    hilo = threading.Thread(target=enviar_correo_alertas, args=(alertas_globales, EMAIL_DESTINATARIOS))
    hilo.start()
    return jsonify({"ok": True, "mensaje": f"Correo enviando a {', '.join(EMAIL_DESTINATARIOS)}..."})

@app.route("/predecir", methods=["POST"])
def predecir():
    if "usuario" not in session:
        return jsonify({"error": "No autorizado"}), 401
    data   = request.get_json()
    nombre = str(data.get("producto","")).strip()
    dias   = int(data.get("dias",30))
    mask   = df['Producto_Limpio']==nombre
    if not mask.any():
        return jsonify({"error": f"Producto '{nombre}' no encontrado."})
    idx   = df[mask].index[-1]
    stock = int(df.loc[idx,'Stock Inicial'])
    ent   = int(df.loc[idx,'Entradas'])
    sku   = str(df.loc[idx,'Código SKU'])
    cat   = str(df.loc[idx,'Categoría'])
    pred  = int(np.round(modelo_rf.predict(X.loc[[idx]])[0]))
    total = stock+ent
    dias_r= round(total/pred,1) if pred>0 else 999
    vproy = pred*dias
    pedir = max(0,vproy-total)
    nivel = "critico" if dias_r<3 else "precaucion" if dias_r<7 else "seguro"
    correo = (
        f"ASUNTO: Orden de Compra Sugerida — {nombre}\n\nEstimado equipo,\n\n"
        f"El sistema detectó que '{nombre}' (SKU: {sku}) requiere reabastecimiento.\n\n"
        f"  • Stock disponible  : {total} unidades\n  • Duración estimada : {dias_r} días\n"
        f"  • Cantidad a pedir  : {int(pedir)} unidades\n\nSistema IA — Gestión de Inventario"
    ) if pedir>0 else (
        f"ASUNTO: Reporte Rutinario — {nombre}\n\nEl stock de '{nombre}' ({total} unidades) "
        f"es suficiente para los próximos {dias} días.\n\nSistema IA — Gestión de Inventario"
    )
    return jsonify({"producto":nombre,"sku":sku,"categoria":cat,"stock_total":total,
                    "dias_restantes":dias_r,"prediccion_diaria":pred,"ventas_proyectadas":int(vproy),
                    "cantidad_pedir":int(pedir),"nivel":nivel,"dias_proyeccion":dias,"correo":correo})

if __name__ == "__main__":
    print("🌐 Abre tu navegador en: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
