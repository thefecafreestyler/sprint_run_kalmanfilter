# Copyright (c) 2025 Zakaria El Ouardi
# All rights reserved.
#
# This software is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You are free to distribute, remix, adapt, and build upon this material in any medium or format for noncommercial purposes only,
# and only so long as proper attribution is given to the creator, Zakaria El Ouardi.
#
# For inquiries, contact Zakaria El Ouardi at zakaria.elouardi@gmx.de.
#
# License details: https://creativecommons.org/licenses/by-nc/4.0/

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from numpy.linalg import inv

# ----------------------------------------------------------
# Streamlit Config: single wide layout, hidden sidebar
# ----------------------------------------------------------
st.set_page_config(
    page_title="Run Overview",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------
# 1) Custom CSS
# ----------------------------------------------------------
st.markdown("""
<style>
#MainMenu, footer, [data-testid="collapsedControl"] {
    visibility: hidden;
    height: 0;
}
div[data-testid="stSidebar"] {
    display: none;
}
body {
    background-color: #F8F9FA;
    color: #333333;
    font-family: "Segoe UI", Tahoma, sans-serif;
    margin: 0; 
    padding: 0;
}
.upload-bar {
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #FFFFFF;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 16px;
    margin-bottom: 0.5rem;
}
.css-19u78sn, .css-h72f3o, .css-6h4ema {
    width: 300px !important;
    font-size: 14px !important;
    padding: 5px 8px !important;
}
.stFileUploader label {
    font-size: 14px !important;
    color: #444444;
}

/* "Nike style" headings */
.nike-heading-big {
    font-style: italic;
    font-weight: bold;
    font-size: 24px;
    margin-bottom: 0.75rem;
}
.nike-heading {
    font-style: italic;
    font-weight: bold;
    font-size: 20px;
    margin-bottom: 0.75rem;
}

/* Metric Cards */
.metric-card {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 16px 12px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
}
.metric-label {
    font-size: 14px;
    color: #6c757d;
    margin-bottom: 4px;
    font-weight: 500;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #343a40;
    margin-bottom: 0;
}
hr {
    border: none;
    border-top: 1px solid #dee2e6;
    margin: 0.75rem 0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 2) File Uploader (the only visible element if no file)
# ----------------------------------------------------------
st.markdown('<div class="upload-bar">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if not uploaded_file:
    st.stop()

# ----------------------------------------------------------
# 3) Helper Functions
# ----------------------------------------------------------
def read_csv_file(f):
    col_names = ["time","ax","ay","az","lat","lon","alt"]
    preview = pd.read_csv(f, header=None, nrows=5)
    has_header = False
    try:
        pd.to_numeric(preview.iloc[:,4], errors='raise')
        pd.to_numeric(preview.iloc[:,5], errors='raise')
    except:
        has_header = True
    f.seek(0)
    if has_header:
        data = pd.read_csv(f, header=0, usecols=range(7), names=col_names)
    else:
        data = pd.read_csv(f, header=None, usecols=range(7), names=col_names)
    return data

def create_filter(data, cutoff, fs, order, btype='high'):
    """Generic Butterworth filter: btype can be 'high' or 'low'."""
    nyq = 0.5*fs
    norm = cutoff/nyq
    b,a = butter(order, norm, btype=btype)
    return filtfilt(b,a,data)

def convert_to_local_coords(lat, lon):
    R=6371000
    lat_ref, lon_ref = lat[0], lon[0]
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat_ref_rad = np.deg2rad(lat_ref)
    lon_ref_rad = np.deg2rad(lon_ref)
    x = (lon_rad - lon_ref_rad)*np.cos(0.5*(lat_rad+lat_ref_rad))*R
    y = (lat_rad - lat_ref_rad)*R
    return x, y

def kalman_filter(xPos, yPos, ax, ay, fs=100, accel_noise=1.0, gps_noise=2.5, speed_thresh=0.05):
    N = len(xPos)
    X_k = np.array([xPos[0], yPos[0], 0.0, 0.0])
    P_k = np.eye(4)
    Q_base = accel_noise**2
    R_k = (gps_noise**2)*np.eye(2)
    
    X_est = np.zeros((4,N))
    X_est[:,0] = X_k
    
    H = np.array([[1,0,0,0],[0,1,0,0]])
    for i in range(1,N):
        dt=1/fs
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        B = np.array([[0.5*(dt**2),0],[0,0.5*(dt**2)],[dt,0],[0,dt]])
        
        u = np.array([ax[i], ay[i]])
        Q = B @ B.T * Q_base
        
        X_k_pred = F@X_k + B@u
        P_k_pred = F@P_k@F.T + Q
        
        z_meas = np.array([xPos[i], yPos[i]])
        S_k = H@P_k_pred@H.T + R_k
        K_k = P_k_pred@H.T@np.linalg.inv(S_k)
        X_k = X_k_pred + K_k@(z_meas - H@X_k_pred)
        P_k = (np.eye(4)-K_k@H)@P_k_pred
        
        # Zero out speeds if below threshold
        vx, vy = X_k[2], X_k[3]
        spd = np.sqrt(vx*vx + vy*vy)
        if spd < speed_thresh:
            X_k[2], X_k[3] = 0,0
        
        X_est[:,i] = X_k
    
    xFilt = X_est[0,:]
    yFilt = X_est[1,:]
    vxFilt= X_est[2,:]
    vyFilt= X_est[3,:]
    speed_ms = np.sqrt(vxFilt**2 + vyFilt**2)
    speed_kmh = speed_ms*3.6
    # smooth
    window=200
    smoothed = pd.Series(speed_kmh).rolling(window=window, center=True, min_periods=1).mean().values
    return xFilt,yFilt,speed_kmh,smoothed

def detect_peaks(az, distance=18, prominence=3):
    return find_peaks(az, distance=distance, prominence=prominence)

def calc_sprint_times(dist, time_s, speed, sprints=[10,20,50,75,100]):
    i_start = np.argmax(speed>=0.1)
    if i_start>=len(speed) or speed[i_start]<0.1:
        return {d:"N/A" for d in sprints}
    result={}
    for d in sprints:
        idx = np.searchsorted(dist,d,side='left')
        if idx<len(time_s) and idx>=i_start:
            t_val = time_s.iloc[idx] - time_s.iloc[i_start]
            result[d] = t_val
        else:
            result[d] = "N/A"
    return result

# ----------------------------------------------------------
# 4) Read CSV
# ----------------------------------------------------------
data = read_csv_file(uploaded_file)
if data.shape[1]<7:
    st.error("CSV must have at least 7 columns.")
    st.stop()

time_str = data["time"]
ax_raw  = data["ax"]
ay_raw  = data["ay"]
az_raw  = data["az"]
lat_ser = data["lat"]
lon_ser = data["lon"]
alt_ser = data["alt"]

# Clean lat/lon
lat_ser = pd.to_numeric(lat_ser, errors='coerce')
lon_ser = pd.to_numeric(lon_ser, errors='coerce')
data = data.dropna(subset=["lat","lon"]).reset_index(drop=True)
if len(data)==0:
    st.warning("No valid lat/lon data found. The map won't show properly.")

# Convert time
time_dt = pd.to_datetime(data["time"], utc=True)
time_s = (time_dt - time_dt.iloc[0]).dt.total_seconds()
date_str = time_dt.iloc[0].strftime("%d. %B %Y")  # e.g. 01. January 2020

# ----------------------------------------------------------
# 5) Tweak Options at the Bottom (but we need them NOW)
# ----------------------------------------------------------
tweak_container = st.container()
default_stationary_thresh = 0.05
default_fc_axay = 2.2
default_filter_mode = 'high'
default_accel_noise = 1.0
default_gps_noise = 2.5

with tweak_container.expander("Advanced Tweak Options", expanded=False):
    stationary_thresh = st.slider(
        "Stationary Threshold (m/s)",
        min_value=0.0, max_value=1.0, value=default_stationary_thresh, step=0.01
    )
    fc_axay = st.slider(
        "Cutoff Frequency for AX/AY",
        min_value=0.5, max_value=10.0, value=default_fc_axay, step=0.1
    )
    filter_mode = st.selectbox(
        "AX/AY Filter Type",
        ["high", "low"],
        index=0 if default_filter_mode=="high" else 1
    )
    accel_noise_param = st.slider(
        "Accel Noise",
        min_value=0.0, max_value=10.0, value=default_accel_noise, step=0.1
    )
    gps_noise_param = st.slider(
        "GPS Noise",
        min_value=0.0, max_value=10.0, value=default_gps_noise, step=0.1
    )

st.write("")  # Just some spacing

# ----------------------------------------------------------
# 6) Processing with Tweak Parameters
# ----------------------------------------------------------
if len(data)>0:
    xPos, yPos = convert_to_local_coords(data["lat"].values, data["lon"].values)
    dist_step = np.sqrt(np.diff(xPos, prepend=xPos[0])**2 + np.diff(yPos, prepend=yPos[0])**2)
    cum_dist = np.cumsum(dist_step)
else:
    xPos,yPos=[],[]
    dist_step=[]
    cum_dist=[]

Fs=100
ax_filtered = create_filter(ax_raw.values, fc_axay, Fs, order=2, btype=filter_mode)
ay_filtered = create_filter(ay_raw.values, fc_axay, Fs, order=2, btype=filter_mode)
az_filtered = create_filter(az_raw.values, 11, Fs, order=2, btype='low')

xF,yF,spd_kmh, spd_smooth = kalman_filter(
    xPos,yPos,
    ax_filtered,ay_filtered,
    fs=Fs,
    accel_noise=accel_noise_param,
    gps_noise=gps_noise_param,
    speed_thresh=stationary_thresh
)
dist_kf = np.sqrt(np.diff(xF, prepend=xF[0])**2 + np.diff(yF, prepend=yF[0])**2)
cum_dist_kf = np.cumsum(dist_kf)

run_time_s = time_s.iloc[-1] if len(time_s)>1 else 0
mm, ss = divmod(run_time_s, 60)
time_fmt = f"{int(mm)}:{int(ss):02d}"
dist_m = cum_dist_kf[-1] if len(cum_dist_kf)>0 else 0
avg_speed = (dist_m/run_time_s)*3.6 if run_time_s>0 else 0
max_speed= spd_smooth.max() if len(spd_smooth)>0 else 0

# Pace calculations
if dist_m>0:
    avg_pace_min = (run_time_s/60)/(dist_m/1000)
    ap_mins = int(avg_pace_min)
    ap_secs = int((avg_pace_min - ap_mins)*60)
    avg_pace_str = f"{ap_mins}:{ap_secs:02d}/km"
else:
    avg_pace_str = "0:00/km"

if max_speed>0:
    mxp_min = 60.0/max_speed
    mxp_mins = int(mxp_min)
    mxp_secs = int((mxp_min - mxp_mins)*60)
    max_pace_str = f"{mxp_mins}:{mxp_secs:02d}/km"
else:
    max_pace_str = "0:00/km"

peaks,_ = detect_peaks(az_filtered)
if len(peaks)>1:
    pk_times = time_s.iloc[peaks]
    stride_durations = np.diff(pk_times)
    cad_arr = 60.0/stride_durations
    avg_cad = cad_arr.mean()
else:
    avg_cad=0

alt_gain = 0
if len(alt_ser.dropna())>1:
    alt_gain = alt_ser.max() - alt_ser.min()

# Additional Metrics
num_steps = len(peaks)
mass_kg = 85  # Could be a user input
dist_km = dist_m/1000
calories = 1.036 * mass_kg * dist_km

# ----------------------------------------------------------
# 7) Display
# ----------------------------------------------------------
st.markdown(f"<div class='nike-heading-big'>{date_str} - RUN OVERVIEW</div>", unsafe_allow_html=True)

# First row of 4 metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">',unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Distance Covered</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{dist_m:.1f} m</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">',unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Running Time</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{time_fmt}</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">',unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Avg Speed</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{avg_speed:.2f} km/h</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">',unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Max Speed</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{max_speed:.2f} km/h</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# Second row of 4 metrics
col5, col6, col7, col8 = st.columns(4)
with col5:
    st.markdown('<div class="metric-card">',unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Steps</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{num_steps}</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

with col6:
    st.markdown('<div class="metric-card">',unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Calories</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{calories:.0f} kcal</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

with col7:
    st.markdown('<div class="metric-card">',unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Avg Pace</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{avg_pace_str}</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

with col8:
    st.markdown('<div class="metric-card">',unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Max Pace</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{max_pace_str}</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown("<hr/>",unsafe_allow_html=True)

# Speed Change
st.markdown('<div class="nike-heading">Speed Change</div>', unsafe_allow_html=True)
fig_spd = px.line(
    x=time_s, y=spd_smooth,
    labels={"x":"Time [s]", "y":"Speed [km/h]"},
    template="simple_white",
    width=800, height=400
)
fig_spd.update_traces(line=dict(width=3, color="#007BFF"))
fig_spd.update_layout(margin=dict(l=40,r=40,t=40,b=40))
st.plotly_chart(fig_spd, use_container_width=True)

st.markdown("<hr/>",unsafe_allow_html=True)

# Trajectory Map
st.markdown('<div class="nike-heading">Trajectory Map</div>', unsafe_allow_html=True)
if len(data) > 0:
    lat_vals = data["lat"]
    lon_vals = data["lon"]
    
    # Sync array lengths
    min_len = min(len(lat_vals), len(spd_smooth), len(cum_dist_kf))
    lat_vals = lat_vals.iloc[:min_len]
    lon_vals = lon_vals.iloc[:min_len]
    spd_smooth = spd_smooth[:min_len]
    cum_dist_kf = cum_dist_kf[:min_len]
    
    if len(lat_vals) > 1:
        fig_map = go.Figure()
        custom_data = np.vstack((spd_smooth, cum_dist_kf)).T
        
        fig_map.add_trace(go.Scattermapbox(
            lat=lat_vals,
            lon=lon_vals,
            mode="lines",
            line=dict(width=4, color="#3333cc"),
            name="Trajectory",
            customdata=custom_data,
            hovertemplate=(
                "Speed: %{customdata[0]:.2f} km/h<br>"
                "Distance: %{customdata[1]:.2f} m<extra></extra>"
            )
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=[lat_vals.iloc[0]],
            lon=[lon_vals.iloc[0]],
            mode="markers",
            marker=dict(size=8, color="green"),
            name="Start"
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=[lat_vals.iloc[-1]],
            lon=[lon_vals.iloc[-1]],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="End"
        ))
        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=np.mean(lat_vals), lon=np.mean(lon_vals)),
                zoom=14
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Not enough lat/lon points for a trajectory map.")
else:
    st.warning("No valid lat/lon data to plot a map.")

st.markdown("<hr/>",unsafe_allow_html=True)

# Show Sprint Times (plus Speed vs Distance plot)
if st.button("Show Sprint Times"):
    sprints = calc_sprint_times(cum_dist_kf, time_s, spd_smooth)
    sprint_df = pd.DataFrame({
        "Distance (m)": [k for k in sprints.keys()],
        "Time (s)": [f"{v:.2f}" if isinstance(v,float) else "N/A" for v in sprints.values()]
    })
    st.write("##### Sprint Times")
    st.table(sprint_df)
    
    # Speed vs Distance plot
    fig_spd_dist = px.line(
        x=cum_dist_kf, y=spd_smooth,
        labels={"x": "Distance [m]", "y": "Speed [km/h]"},
        template="simple_white",
        title="Speed vs Distance",
        width=800, height=400
    )
    fig_spd_dist.update_traces(line=dict(width=3, color="orange"))
    fig_spd_dist.update_layout(margin=dict(l=40,r=40,t=60,b=40))
    st.plotly_chart(fig_spd_dist, use_container_width=True)

st.markdown("<hr/>",unsafe_allow_html=True)

# Cadence
st.markdown('<div class="nike-heading">Cadence</div>', unsafe_allow_html=True)
if len(peaks) < 2:
    st.warning("Not enough peaks to plot cadence.")
else:
    pk_times = time_s.iloc[peaks]
    intervals = np.diff(pk_times)
    raw_cadence = 60.0 / intervals
    mid_t = 0.5*(pk_times[:-1].values + pk_times[1:].values)
    cad_smooth = pd.Series(raw_cadence).rolling(window=3, center=True, min_periods=1).mean().values
    
    fig_cad = go.Figure()
    fig_cad.add_trace(go.Scatter(
        x=mid_t,
        y=raw_cadence,
        mode="lines+markers",
        line=dict(width=2, color="#FF9900"),
        marker=dict(size=6),
        name="Raw Cadence"
    ))
    fig_cad.add_trace(go.Scatter(
        x=mid_t,
        y=cad_smooth,
        mode="lines",
        line=dict(width=3, color="#FF4C4C"),
        name="Smoothed Cadence"
    ))
    fig_cad.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Cadence [spm]",
        template="simple_white",
        margin=dict(l=40,r=40,t=40,b=40),
        width=800, height=400
    )
    st.plotly_chart(fig_cad, use_container_width=True)

st.markdown("<hr/>",unsafe_allow_html=True)

# Average Stride Length vs Speed
st.markdown('<div class="nike-heading">Average Stride Length vs Speed</div>', unsafe_allow_html=True)
peaks_idx = peaks
if len(peaks_idx)<2:
    st.warning("Not enough peaks to plot stride length vs speed.")
else:
    pk_dist = cum_dist_kf[peaks_idx]
    stride_len = np.diff(pk_dist)
    stride_spd = spd_smooth[peaks_idx[:-1]]
    
    fig_sl = px.scatter(
        x=stride_spd, y=stride_len,
        labels={"x":"Speed [km/h]","y":"Stride Length [m]"},
        template="simple_white",
        width=800, height=400
    )
    fig_sl.update_traces(mode="markers", marker=dict(size=6, color="#2CA02C"), name="Raw Data")
    
    bin_size=1
    max_spd = np.ceil(stride_spd.max()) if len(stride_spd)>0 else 1
    bin_edges = np.arange(0, max_spd+bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size/2
    vals=[]
    for i in range(len(bin_centers)):
        mask = (stride_spd>=bin_edges[i]) & (stride_spd<bin_edges[i+1])
        if np.any(mask):
            vals.append(stride_len[mask].mean())
        else:
            vals.append(np.nan)
    vals_smoothed = pd.Series(vals).rolling(window=2, center=True, min_periods=1).mean().values
    fig_sl.add_trace(go.Scatter(
        x=bin_centers,
        y=vals_smoothed,
        mode="lines",
        line=dict(width=3, color="#FF4C4C"),
        name="Smoothed"
    ))
    fig_sl.update_layout(margin=dict(l=40,r=40,t=40,b=40))
    st.plotly_chart(fig_sl, use_container_width=True)

st.markdown("<hr/>",unsafe_allow_html=True)

# GRF Plot
st.markdown('<div class="nike-heading">GRF Plot</div>', unsafe_allow_html=True)
mass=85
g=9.81
peaks_idx = peaks
F_grf = mass*(az_filtered[peaks_idx]+g) if len(peaks_idx)>0 else []

if len(peaks_idx)<2:
    st.warning("Not enough peaks to show GRF or stride distances.")
else:
    fig_grf = go.Figure()
    fig_grf.add_trace(go.Scatter(
        x=cum_dist_kf,
        y=az_filtered,
        mode="lines",
        line=dict(width=2, color="#343a40"),
        name="Z-Acceleration"
    ))
    fig_grf.add_trace(go.Scatter(
        x=cum_dist_kf[peaks_idx],
        y=az_filtered[peaks_idx],
        mode="markers",
        marker=dict(size=7, color="red"),
        name="Peaks"
    ))
    
    # Label each peak with force in N
    for i, pk in enumerate(peaks_idx):
        force_val = F_grf[i] if i<len(F_grf) else 0
        fig_grf.add_annotation(
            x=cum_dist_kf[pk],
            y=az_filtered[pk],
            text=f"{force_val:.1f} N",
            showarrow=True, arrowhead=2,
            arrowcolor="#FF4C4C",
            yshift=15,
            font=dict(color="#FF4C4C", size=10)
        )
    
    # Two-headed arrow for distances
    for i in range(len(peaks_idx)-1):
        d1, d2 = cum_dist_kf[peaks_idx[i]], cum_dist_kf[peaks_idx[i+1]]
        stride_val = d2 - d1
        y_mid = (az_filtered[peaks_idx[i]]+az_filtered[peaks_idx[i+1]])*0.5
        
        fig_grf.add_annotation(
            x=d2, y=y_mid,
            ax=d1, ay=y_mid,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, startarrowhead=3,
            arrowwidth=2, arrowcolor="#20c997",
            text=f"{stride_val:.2f} m",
            showarrow=True,
            font=dict(color="#20c997", size=10),
            yshift=-15
        )
    
    fig_grf.update_layout(
        xaxis_title="Distance [m]",
        yaxis_title="Z-Acceleration [m/s^2]",
        template="simple_white",
        width=900, height=400,
        margin=dict(l=40,r=40,t=40,b=40)
    )
    st.plotly_chart(fig_grf, use_container_width=True)

# ----------------------------------------------------------
# License and Project Information
# ----------------------------------------------------------
def render_license_header():
    st.markdown("""
    <div style="text-align: center; font-size: 14px; color: #6c757d; margin-bottom: 20px;">
        <p><b>Run Overview Web App</b></p>
        <p>Copyright &copy; 2025 Zakaria El Ouardi. All rights reserved.</p>
        <p>This software is licensed under the 
        <a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)</a>.</p>
        <p>Project created on: <b>07 January 2025</b></p>
        <p>Contact: <a href="mailto:zakaria.elouardi@gmx.de">zakaria.elouardi@gmx.de</a></p>
        <hr style="margin-top: 20px;">
    </div>
    """, unsafe_allow_html=True)

render_license_header()
