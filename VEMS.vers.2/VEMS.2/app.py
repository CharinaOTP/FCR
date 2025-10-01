# ============================================================
# VEMS - Normal Travel (NT) Calculator & Reporting System
# ============================================================

import os
from datetime import date, datetime
import bcrypt
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, ForeignKey, select
from sqlalchemy.orm import declarative_base, relationship, Session
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyPDF2 import PdfMerger

# ============================================================
# ==== CONFIG & DB SETUP =====================================
# ============================================================
APP_TITLE = "DCWD: MONTHLY REPORT OF OFFICIAL TRAVEL CALCULATOR"
DB_FOLDER = "data"
DB_FILE = os.path.join(DB_FOLDER, "vems_nt.sqlite")
DEFAULT_TOLERANCE = 0.10  # 10%

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_FILE}", echo=False, future=True)

# ============================================================
# ==== DB MODELS =============================================
# ============================================================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="Viewer")

class Vehicle(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True)
    plate = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    created_by = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    trips = relationship("Trip", back_populates="vehicle", cascade="all, delete-orphan")

class Trip(Base):
    __tablename__ = "trips"
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False)
    trip_date = Column(Date, nullable=False)
    odo_start = Column(Integer, nullable=False)
    odo_end = Column(Integer, nullable=False)
    fuel_liters = Column(Float, nullable=False)
    created_by = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    vehicle = relationship("Vehicle", back_populates="trips")

Base.metadata.create_all(engine)

# ============================================================
# ==== USER AUTH =============================================
# ============================================================
def get_session():
    return Session(bind=engine, future=True)

def seed_admin():
    with get_session() as s:
        if not s.query(User).count():
            pw_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
            s.add(User(username="admin", password_hash=pw_hash, role="Admin"))
            s.commit()
seed_admin()

def login_user(username, password):
    with get_session() as s:
        user = s.query(User).filter(User.username == username).first()
        if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            return {"id": user.id, "username": user.username, "role": user.role}
    return None

# ============================================================
# ==== HELPERS ===============================================
# ============================================================
def compute_trip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["distance_km"] = out["odo_end"] - out["odo_start"]
    out["ratio_km_per_l"] = np.where(
        out["fuel_liters"] > 0,
        out["distance_km"] / out["fuel_liters"],
        np.nan
    )
    return out

def trimmed_mean(values):
    s = pd.Series(values).dropna().astype(float)
    if len(s) >= 3:
        s = s.sort_values().iloc[1:-1]
    return s.mean() if not s.empty else None

def generate_pdf_report(vehicle_name, df, nt, low, high, tol_pct):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, f"Normal Travel Report: {vehicle_name}")

    # Summary
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, f"Total of Normal Ratios: {df['ratio_km_per_l'].sum():.3f}")
    c.drawString(50, height - 100, f"Number of Samples: {len(df)}")
    c.drawString(50, height - 120, f"Average Fuel Consumption Ratio (NT): {nt:.3f} km/L")
    c.drawString(50, height - 140, f"10% of Average: {(nt*0.1):.3f}")
    c.drawString(50, height - 160, f"Average +10% (High Range): {high:.3f}")
    c.drawString(50, height - 180, f"Average -10% (Low Range): {low:.3f}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 210, f"Fuel Consumption Range {low:.2f} to {high:.2f}")

    # Table header
    c.setFont("Helvetica-Bold", 10)
    y = height - 250
    c.drawString(50, y, "Fuel Liters")
    c.drawString(120, y, "Odo Start")
    c.drawString(190, y, "Odo End")
    c.drawString(260, y, "Distance")
    c.drawString(330, y, "Normal")
    c.drawString(420, y, "Highest & Lowest")

    # Table rows
    c.setFont("Helvetica", 9)
    y -= 20
    for _, row in df.iterrows():
        c.drawString(50, y, str(row["fuel_liters"]))
        c.drawString(120, y, str(row["odo_start"]))
        c.drawString(190, y, str(row["odo_end"]))
        c.drawString(260, y, str(row["distance_km"]))
        c.drawString(330, y, f"{row['Normal']}" if pd.notna(row["Normal"]) else "")
        c.drawString(420, y, str(row["Highest & Lowest"]))
        y -= 15
        if y < 100:  # New page if space is finished
            c.showPage()
            y = height - 50

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ============================================================
# ==== STREAMLIT SESSION =====================================
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
if "user" not in st.session_state:
    st.session_state.user = None

# ============================================================
# ==== LOGIN PAGE ============================================
# ============================================================
if not st.session_state.user:
    st.title(APP_TITLE)
    st.subheader("Login Required")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            user = login_user(username, password)
            if user:
                st.session_state.user = user
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password")
    st.stop()

user = st.session_state.user
st.sidebar.title("Navigation")
menu = ["Trips", "NT Report", "Logout"]
if user["role"] == "Admin":
    menu.insert(0, "Vehicles")
choice = st.sidebar.radio("Go to", menu)

if choice == "Logout":
    st.session_state.user = None
    st.experimental_rerun()

# ============================================================
# ==== VEHICLE MANAGEMENT ====================================
# ============================================================
if choice == "Vehicles" and user["role"] == "Admin":
    st.subheader("Vehicle Management")
    with get_session() as s:
        vdf = pd.read_sql(select(Vehicle.id, Vehicle.plate, Vehicle.name), s.bind)

    with st.form("add_vehicle_form"):
        plate = st.text_input("Plate *")
        name = st.text_input("Name / Description")
        if st.form_submit_button("Add Vehicle"):
            if plate.strip():
                with get_session() as s:
                    if s.query(Vehicle).filter(Vehicle.plate == plate.strip().upper()).first():
                        st.error("❌ Vehicle already exists")
                    else:
                        s.add(Vehicle(plate=plate.strip().upper(),
                                      name=name.strip() or None,
                                      created_by=user["username"]))
                        s.commit()
                        st.success("Vehicle added")
                        st.experimental_rerun()
            else:
                st.error("Plate is required")
    st.dataframe(vdf, use_container_width=True)

# ============================================================
# ==== MULTI-ROW TRIP ENTRY ==================================
# ============================================================
if choice == "Trips":
    st.subheader("Encode Multiple Trips")

    with get_session() as s:
        vdf = pd.read_sql(select(Vehicle.id, Vehicle.plate, Vehicle.name), s.bind)

    if vdf.empty:
        st.warning("⚠️ No vehicles available. Add a vehicle first.")
    else:
        label_to_id = {f"{r.plate} — {r.name or ''}": int(r.id) for _, r in vdf.iterrows()}
        veh_label = st.selectbox("Select Vehicle", list(label_to_id.keys()))
        vid = label_to_id[veh_label]

        trips_input = st.data_editor(
            pd.DataFrame([{"Fuel Liters": 0.0, "Odo Start": 0, "Odo End": 0} for _ in range(5)]),
            num_rows="dynamic",
            use_container_width=True
        )

        if st.button("Save Trips"):
            saved = 0
            for _, row in trips_input.dropna().iterrows():
                if row["Odo End"] > row["Odo Start"] and row["Fuel Liters"] > 0:
                    with get_session() as s:
                        s.add(Trip(
                            vehicle_id=vid,
                            trip_date=date.today(),
                            odo_start=int(row["Odo Start"]),
                            odo_end=int(row["Odo End"]),
                            fuel_liters=float(row["Fuel Liters"]),
                            created_by=user["username"]
                        ))
                        s.commit()
                        saved += 1
            st.success(f"✅ {saved} trips saved for {veh_label}")

# ============================================================
# ==== NT REPORT =============================================
# ============================================================
if choice == "NT Report":
    st.subheader("Normal Travel Report")

    tol_pct = st.slider("Tolerance (±%)", 5, 25, int(DEFAULT_TOLERANCE*100)) / 100
    search_term = st.text_input("🔍 Search by Plate or Vehicle Name")

    q = """
      SELECT t.id, t.trip_date, v.plate, v.name,
             t.odo_start, t.odo_end, t.fuel_liters, v.id as vehicle_id
      FROM trips t JOIN vehicles v ON v.id = t.vehicle_id
      ORDER BY v.plate, t.trip_date
    """
    df = pd.read_sql(q, engine, parse_dates=["trip_date"])
    df = compute_trip_metrics(df)

    if df.empty:
        st.info("⚠️ No trips available")
    else:
        pdf_buffers = []

        for plate, g in df.groupby("plate"):
            if search_term and search_term.lower() not in plate.lower() and search_term.lower() not in str(g["name"].iloc[0]).lower():
                continue

            g = g.sort_values("trip_date").tail(20).reset_index(drop=True)
            ratios = g["ratio_km_per_l"].dropna()
            nt = trimmed_mean(ratios)
            if nt is None:
                continue

            low, high = nt*(1-tol_pct), nt*(1+tol_pct)
            max_row = g.loc[g["ratio_km_per_l"].idxmax()]
            min_row = g.loc[g["ratio_km_per_l"].idxmin()]

            g["Normal"] = g["ratio_km_per_l"]
            g["Highest & Lowest"] = ""
            g.loc[g["id"] == max_row["id"], ["Normal","Highest & Lowest"]] = [None, f"🔵 HIGH {max_row['ratio_km_per_l']:.2f}"]
            g.loc[g["id"] == min_row["id"], ["Normal","Highest & Lowest"]] = [None, f"🔴 LOW {min_row['ratio_km_per_l']:.2f}"]

            st.markdown(f"### {plate} — {g['name'].iloc[0]}")
            st.dataframe(
                g[["fuel_liters","odo_start","odo_end","distance_km","Normal","Highest & Lowest"]],
                use_container_width=True
            )

            st.markdown(f"**Total of Normal Ratios:** {g['ratio_km_per_l'].sum():.3f}")
            st.markdown(f"**Number of Samples:** {len(g)}")
            st.markdown(f"**Average Fuel Consumption Ratio (NT):** {nt:.3f} km/L")
            st.markdown(f"**10% of Average:** {(nt*0.1):.3f}")
            st.markdown(f"**Average +10% (High Range):** {high:.3f}")
            st.markdown(f"**Average -10% (Low Range):** {low:.3f}")
            st.markdown(f"### Fuel Consumption Range {low:.2f} to {high:.2f}")

            # Individual PDF
            pdf_buffer = generate_pdf_report(f"{plate} — {g['name'].iloc[0]}", g, nt, low, high, tol_pct)
            pdf_buffers.append(pdf_buffer.getvalue())

            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"NT_Report_{plate}_{date.today()}.pdf",
                mime="application/pdf"
            )

            # Delete vehicle option
            if user["role"] == "Admin":
                if st.button(f"Delete Vehicle {plate}"):
                    with get_session() as s:
                        v = s.query(Vehicle).filter(Vehicle.plate == plate).first()
                        if v:
                            s.delete(v)
                            s.commit()
                            st.warning(f"Vehicle {plate} deleted")
                            st.experimental_rerun()

        # Merge all PDFs into one
        if pdf_buffers:
            merger = PdfMerger()
            for buf in pdf_buffers:
                merger.append(BytesIO(buf))
            merged_output = BytesIO()
            merger.write(merged_output)
            merger.close()
            merged_output.seek(0)

            st.download_button(
                label="Download ALL Reports (Merged)",
                data=merged_output,
                file_name=f"NT_AllReports_{date.today()}.pdf",
                mime="application/pdf"
            )
