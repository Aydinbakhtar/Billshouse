import re
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pvlib
import requests
import streamlit as st


# ==========================================================
# TEXAS HOME SOLAR FORECAST WEB APP
# ZIP code + home PV settings -> forecast power, energy, map
# Data:
#   - ZIP geocoding: Zippopotam.us
#   - Solar/weather forecast: Open-Meteo Forecast API
# ==========================================================


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(
    page_title="Texas Home Solar Forecast",
    page_icon="☀️",
    layout="wide"
)

TIMEZONE = "America/Chicago"


# -----------------------------
# Utility functions
# -----------------------------
def localize_time_index(time_values, timezone=TIMEZONE):
    """Convert Open-Meteo time strings to timezone-aware pandas DatetimeIndex."""
    idx = pd.DatetimeIndex(pd.to_datetime(time_values))

    if idx.tz is None:
        idx = idx.tz_localize(timezone)
    else:
        idx = idx.tz_convert(timezone)

    return idx


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def zip_to_location(zip_code: str):
    """
    Convert a US ZIP code to lat/lon using Zippopotam.us.
    Restrict to Texas.
    """
    zip_code = zip_code.strip()

    if not re.fullmatch(r"\d{5}", zip_code):
        raise ValueError("Please enter a valid 5-digit ZIP code.")

    url = f"https://api.zippopotam.us/us/{zip_code}"
    r = requests.get(url, timeout=20)

    if r.status_code == 404:
        raise ValueError("ZIP code was not found.")

    r.raise_for_status()
    data = r.json()

    places = data.get("places", [])

    if not places:
        raise ValueError("No location information was returned for this ZIP code.")

    place = places[0]

    state_abbr = place.get("state abbreviation", "").upper()

    if state_abbr != "TX":
        raise ValueError(
            f"This app is currently limited to Texas ZIP codes. "
            f"The ZIP code you entered appears to be in {state_abbr or 'another state'}."
        )

    lat = float(place["latitude"])
    lon = float(place["longitude"])

    return {
        "zip_code": zip_code,
        "city": place.get("place name", "Unknown city"),
        "state": place.get("state", "Texas"),
        "state_abbr": state_abbr,
        "lat": lat,
        "lon": lon,
    }


def estimate_ac_power_kw(
    poa_wm2,
    air_temp_c,
    is_day=None,
    system_size_kw_dc=8.0,
    inverter_size_kw_ac=7.6,
    derate=0.86,
    inverter_eff=0.96,
    temp_coeff_per_c=-0.0035,
    noct_c=45.0,
    apply_temperature=True,
):
    """
    Simple home-scale PV power model.

    poa_wm2:
        Plane-of-array irradiance / global tilted irradiance, W/m2.
    air_temp_c:
        Air temperature in Celsius.
    is_day:
        1 for daylight, 0 for night.
    """

    poa = np.asarray(poa_wm2, dtype=float)
    temp = np.asarray(air_temp_c, dtype=float)

    poa = np.maximum(poa, 0.0)

    if apply_temperature:
        cell_temp_c = temp + ((noct_c - 20.0) / 800.0) * poa
        temp_factor = 1.0 + temp_coeff_per_c * (cell_temp_c - 25.0)
        temp_factor = np.maximum(temp_factor, 0.0)
    else:
        temp_factor = 1.0

    dc_kw = system_size_kw_dc * (poa / 1000.0) * temp_factor * derate
    ac_kw = dc_kw * inverter_eff

    ac_kw = np.minimum(ac_kw, inverter_size_kw_ac)
    ac_kw = np.maximum(ac_kw, 0.0)

    if is_day is not None:
        day = np.asarray(is_day, dtype=float)
        ac_kw = np.where(day > 0, ac_kw, 0.0)

    return ac_kw


def add_clear_sky_poa(df, lat, lon, panel_tilt_deg, panel_azimuth_openmeteo):
    """
    Add clear-sky plane-of-array irradiance using pvlib.

    Open-Meteo azimuth:
        0 = south, -90 = east, 90 = west

    pvlib azimuth:
        180 = south, 90 = east, 270 = west
    """
    panel_azimuth_pvlib = (180 + panel_azimuth_openmeteo) % 360

    site = pvlib.location.Location(
        latitude=lat,
        longitude=lon,
        tz=TIMEZONE
    )

    times = pd.DatetimeIndex(df["time"])

    clearsky = site.get_clearsky(times, model="ineichen")
    solpos = site.get_solarposition(times)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=panel_tilt_deg,
        surface_azimuth=panel_azimuth_pvlib,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )

    df["solar_elevation_deg"] = solpos["apparent_elevation"].values
    df["clear_sky_POA_W_m2"] = np.maximum(poa["poa_global"].values, 0.0)

    return df


def interpolate_series_to_times(source_times, source_values, target_times):
    """Linear interpolation from one datetime grid to another."""
    x = pd.DatetimeIndex(source_times).astype("int64") / 1e9
    xi = pd.DatetimeIndex(target_times).astype("int64") / 1e9
    y = np.asarray(source_values, dtype=float)

    if len(x) == 0:
        return np.full(len(target_times), np.nan)

    if len(x) == 1:
        return np.full(len(target_times), y[0])

    return np.interp(xi, x, y)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_open_meteo_forecast(
    lat,
    lon,
    panel_tilt_deg,
    panel_azimuth_openmeteo,
    forecast_hours,
):
    """
    Try 15-minute forecast first.
    If unavailable, fall back to hourly forecast.
    """
    base_url = "https://api.open-meteo.com/v1/forecast"

    minutely_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": TIMEZONE,
        "tilt": panel_tilt_deg,
        "azimuth": panel_azimuth_openmeteo,
        "forecast_minutely_15": int(forecast_hours * 4),
        "minutely_15": ",".join([
            "global_tilted_irradiance",
            "shortwave_radiation",
            "temperature_2m",
            "is_day",
        ]),
        "hourly": ",".join([
            "cloud_cover",
            "temperature_2m",
            "global_tilted_irradiance",
            "is_day",
        ]),
        "current": ",".join([
            "global_tilted_irradiance",
            "shortwave_radiation",
            "temperature_2m",
            "cloud_cover",
            "is_day",
        ]),
    }

    hourly_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": TIMEZONE,
        "tilt": panel_tilt_deg,
        "azimuth": panel_azimuth_openmeteo,
        "forecast_hours": int(forecast_hours),
        "hourly": ",".join([
            "global_tilted_irradiance",
            "shortwave_radiation",
            "temperature_2m",
            "cloud_cover",
            "is_day",
        ]),
        "current": ",".join([
            "global_tilted_irradiance",
            "shortwave_radiation",
            "temperature_2m",
            "cloud_cover",
            "is_day",
        ]),
    }

    try:
        r = requests.get(base_url, params=minutely_params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "minutely_15" not in data:
            raise ValueError("15-minute forecast data was not returned.")

        m15 = data["minutely_15"]
        times = localize_time_index(m15["time"])

        df = pd.DataFrame({
            "time": times,
            "GTI_W_m2": m15["global_tilted_irradiance"],
            "GHI_W_m2": m15["shortwave_radiation"],
            "air_temp_C": m15["temperature_2m"],
            "is_day": m15["is_day"],
        })

        # Cloud cover is usually hourly, so interpolate it to 15-minute timestamps.
        if "hourly" in data and "cloud_cover" in data["hourly"]:
            h = data["hourly"]
            h_times = localize_time_index(h["time"])
            df["cloud_cover_percent"] = interpolate_series_to_times(
                h_times,
                h["cloud_cover"],
                df["time"]
            )
        else:
            df["cloud_cover_percent"] = np.nan

        return data, df, "15-minute Open-Meteo forecast"

    except Exception:
        r = requests.get(base_url, params=hourly_params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "hourly" not in data:
            raise ValueError("Hourly forecast data was not returned.")

        h = data["hourly"]
        times = localize_time_index(h["time"])

        df = pd.DataFrame({
            "time": times,
            "GTI_W_m2": h["global_tilted_irradiance"],
            "GHI_W_m2": h["shortwave_radiation"],
            "air_temp_C": h["temperature_2m"],
            "cloud_cover_percent": h["cloud_cover"],
            "is_day": h["is_day"],
        })

        return data, df, "hourly Open-Meteo forecast"


def finalize_forecast_dataframe(
    df,
    lat,
    lon,
    panel_tilt_deg,
    panel_azimuth_openmeteo,
    system_size_kw_dc,
    inverter_size_kw_ac,
    derate,
    temp_coeff_per_c,
):
    """Add power, losses, and energy to forecast dataframe."""

    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    df = add_clear_sky_poa(
        df,
        lat=lat,
        lon=lon,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_openmeteo=panel_azimuth_openmeteo,
    )

    # Step length for energy calculation.
    step_hours = df["time"].diff().shift(-1).dt.total_seconds() / 3600.0
    median_step = step_hours.dropna().median()

    if pd.isna(median_step):
        median_step = 1.0

    df["step_hours"] = step_hours.fillna(median_step)

    # Prevent negative cloud-loss artifacts when forecast GTI slightly exceeds pvlib clear-sky POA.
    df["clear_sky_POA_for_loss_W_m2"] = np.maximum(
        df["clear_sky_POA_W_m2"],
        df["GTI_W_m2"]
    )

    df["power_clear_sky_kW"] = estimate_ac_power_kw(
        df["clear_sky_POA_for_loss_W_m2"],
        df["air_temp_C"],
        df["is_day"],
        system_size_kw_dc=system_size_kw_dc,
        inverter_size_kw_ac=inverter_size_kw_ac,
        derate=derate,
        temp_coeff_per_c=temp_coeff_per_c,
        apply_temperature=False,
    )

    df["power_cloud_only_no_temp_kW"] = estimate_ac_power_kw(
        df["GTI_W_m2"],
        df["air_temp_C"],
        df["is_day"],
        system_size_kw_dc=system_size_kw_dc,
        inverter_size_kw_ac=inverter_size_kw_ac,
        derate=derate,
        temp_coeff_per_c=temp_coeff_per_c,
        apply_temperature=False,
    )

    df["power_forecast_kW"] = estimate_ac_power_kw(
        df["GTI_W_m2"],
        df["air_temp_C"],
        df["is_day"],
        system_size_kw_dc=system_size_kw_dc,
        inverter_size_kw_ac=inverter_size_kw_ac,
        derate=derate,
        temp_coeff_per_c=temp_coeff_per_c,
        apply_temperature=True,
    )

    df["cloud_loss_kW"] = (
        df["power_clear_sky_kW"] -
        df["power_cloud_only_no_temp_kW"]
    ).clip(lower=0)

    df["temperature_loss_kW"] = (
        df["power_cloud_only_no_temp_kW"] -
        df["power_forecast_kW"]
    ).clip(lower=0)

    df["energy_kWh"] = df["power_forecast_kW"] * df["step_hours"]

    df["date"] = df["time"].dt.date

    df["clear_sky_index"] = (
        df["GTI_W_m2"] / df["clear_sky_POA_for_loss_W_m2"].replace(0, np.nan)
    ).clip(0, 1.25).fillna(0)

    df["inferred_cloudiness_percent"] = (
        1.0 - df["clear_sky_index"]
    ).clip(0, 1) * 100.0

    return df


def simulate_inverter_like_noise(
    df,
    system_size_kw_dc,
    inverter_size_kw_ac,
    derate,
    temp_coeff_per_c,
    resolution_label,
    noise_strength,
    seed=42,
):
    """
    Create synthetic inverter-like variability from forecast irradiance.

    This is not observation. It uses the forecast clear-sky index to create
    plausible short-term cloud dips and edge brightening.
    """

    if resolution_label.startswith("1 second"):
        rule = "1s"
        max_hours = 24
        step_seconds = 1
    elif resolution_label.startswith("10 second"):
        rule = "10s"
        max_hours = 24
        step_seconds = 10
    else:
        rule = "1min"
        max_hours = 72
        step_seconds = 60

    df = df.copy()
    start_time = df["time"].min()
    end_time = start_time + pd.Timedelta(hours=max_hours)

    base = df[df["time"] <= end_time].copy()

    cols = [
        "GTI_W_m2",
        "clear_sky_POA_W_m2",
        "air_temp_C",
        "is_day",
        "power_forecast_kW",
    ]

    df_sim = (
        base.set_index("time")[cols]
        .sort_index()
        .resample(rule)
        .interpolate("time")
    )

    clear = df_sim["clear_sky_POA_W_m2"].replace(0, np.nan)
    k_clear = (df_sim["GTI_W_m2"] / clear).clip(0, 1.25).fillna(0)

    df_sim["clear_sky_index"] = k_clear
    df_sim["inferred_cloudiness"] = (1 - k_clear).clip(0, 1)

    broken_cloud = (
        4 * df_sim["inferred_cloudiness"] * (1 - df_sim["inferred_cloudiness"])
    ).clip(0, 1).values

    irradiance_factor = (df_sim["clear_sky_POA_W_m2"] / 900.0).clip(0, 1).values
    cloudiness = df_sim["inferred_cloudiness"].values

    n = len(df_sim)
    rng = np.random.default_rng(seed)

    # Fast stochastic flicker.
    sigma = (
        0.004
        + noise_strength * 0.035 * broken_cloud
        + noise_strength * 0.010 * cloudiness
    ) * irradiance_factor

    eps = rng.normal(0, sigma)

    rho = 0.94 if step_seconds <= 10 else 0.70
    ar_noise = np.zeros(n)

    for i in range(1, n):
        ar_noise[i] = rho * ar_noise[i - 1] + eps[i]

    # Cloud-shadow dips and edge boosts.
    shade_loss = np.zeros(n)
    edge_boost = np.zeros(n)

    i = 0

    while i < n:
        event_rate_per_hour = 1.0 + 22.0 * broken_cloud[i]
        p_event = (
            event_rate_per_hour
            * (step_seconds / 3600.0)
            * noise_strength
            * irradiance_factor[i]
        )

        if rng.random() < p_event:
            duration_seconds = int(rng.integers(30, 240))
            duration_steps = max(1, int(duration_seconds / step_seconds))
            end = min(i + duration_steps, n)
            length = end - i

            if length > 2:
                local_broken = max(broken_cloud[i], 0.05)
                depth = min(rng.uniform(0.10, 0.75) * local_broken, 0.90)

                x = np.linspace(0, 1, length)
                shape = np.sin(np.pi * x) ** rng.uniform(0.5, 1.3)

                shade_loss[i:end] = np.maximum(shade_loss[i:end], depth * shape)

                boost_amp = rng.uniform(0.02, 0.16) * local_broken
                edge_steps = max(1, min(length // 4, int(30 / step_seconds)))

                edge_boost[i:i + edge_steps] += boost_amp

                edge_start = max(i, end - edge_steps)
                edge_boost[edge_start:end] += boost_amp

            i += max(1, duration_steps // 3)

        else:
            i += 1

    base_gti = df_sim["GTI_W_m2"].values
    clear_poa = df_sim["clear_sky_POA_W_m2"].values

    noisy_gti = base_gti * (1 + ar_noise)
    noisy_gti = noisy_gti * (1 - shade_loss)
    noisy_gti = noisy_gti + clear_poa * edge_boost

    noisy_gti = np.clip(
        noisy_gti,
        0,
        clear_poa * 1.20
    )

    df_sim["GTI_noisy_W_m2"] = noisy_gti

    df_sim["power_noisy_kW_raw"] = estimate_ac_power_kw(
        df_sim["GTI_noisy_W_m2"],
        df_sim["air_temp_C"],
        df_sim["is_day"],
        system_size_kw_dc=system_size_kw_dc,
        inverter_size_kw_ac=inverter_size_kw_ac,
        derate=derate,
        temp_coeff_per_c=temp_coeff_per_c,
        apply_temperature=True,
    )

    # Small inverter response smoothing.
    window = 5 if step_seconds <= 10 else 3

    df_sim["power_noisy_kW"] = (
        df_sim["power_noisy_kW_raw"]
        .rolling(window=window, min_periods=1, center=True)
        .mean()
    )

    # Preserve total forecast energy approximately.
    step_hours = step_seconds / 3600.0

    smooth_energy = (df_sim["power_forecast_kW"] * step_hours).sum()
    noisy_energy = (df_sim["power_noisy_kW"] * step_hours).sum()

    if noisy_energy > 0:
        scale = smooth_energy / noisy_energy
        df_sim["power_noisy_kW"] = (
            df_sim["power_noisy_kW"] * scale
        ).clip(lower=0, upper=inverter_size_kw_ac)

    df_sim["energy_noisy_kWh"] = df_sim["power_noisy_kW"] * step_hours

    return df_sim.reset_index()


def make_power_plot(df, sim_df=None):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["power_clear_sky_kW"],
            mode="lines",
            name="Clear-sky potential",
            line=dict(width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["power_forecast_kW"],
            mode="lines",
            name="Forecast power",
            line=dict(width=3)
        )
    )

    if sim_df is not None and not sim_df.empty:
        plot_sim = sim_df.copy()

        # Downsample only for display if too many points.
        if len(plot_sim) > 16000:
            plot_sim = (
                plot_sim.set_index("time")
                .resample("15s")
                .mean(numeric_only=True)
                .reset_index()
            )

        fig.add_trace(
            go.Scatter(
                x=plot_sim["time"],
                y=plot_sim["power_noisy_kW"],
                mode="lines",
                name="Inverter-like noisy simulation",
                line=dict(width=1)
            )
        )

    fig.update_layout(
        title="Home Solar Power Forecast",
        xaxis_title="Time",
        yaxis_title="AC Power (kW)",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h")
    )

    return fig


def make_loss_plot(df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["cloud_loss_kW"],
            mode="lines",
            name="Cloud / shading loss"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["temperature_loss_kW"],
            mode="lines",
            name="Temperature loss"
        )
    )

    fig.update_layout(
        title="Estimated Cloud/Shading Loss and Temperature Loss",
        xaxis_title="Time",
        yaxis_title="Power Loss (kW)",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h")
    )

    return fig


def homeowner_summary_text(
    current_kw,
    next24_kwh,
    electricity_rate,
    current_home_load_kw,
):
    coverage = 0.0

    if current_home_load_kw > 0:
        coverage = current_kw / current_home_load_kw * 100.0

    value = next24_kwh * electricity_rate

    if current_kw < 0.1:
        status = "very low production"
        explanation = "This is normal at night, near sunrise or sunset, or during heavy cloud cover."
    elif current_kw < 1.5:
        status = "low to moderate production"
        explanation = "This can help cover small household loads such as lights, Wi-Fi, refrigerator, and laptops."
    elif current_kw < 4.0:
        status = "good production"
        explanation = "This can cover many normal daytime home loads."
    else:
        status = "strong production"
        explanation = "This may cover most daytime demand and may support air conditioning, appliances, or battery charging."

    return status, explanation, coverage, value


# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.title("☀️ Home Solar Inputs")

zip_code = st.sidebar.text_input(
    "Texas ZIP code",
    value="76010",
    max_chars=5,
)

system_size_kw_dc = st.sidebar.number_input(
    "Solar system size (kW DC)",
    min_value=0.5,
    max_value=50.0,
    value=8.0,
    step=0.5,
)

default_inverter = min(system_size_kw_dc * 0.95, system_size_kw_dc)

inverter_size_kw_ac = st.sidebar.number_input(
    "Inverter size (kW AC)",
    min_value=0.5,
    max_value=50.0,
    value=float(round(default_inverter, 1)),
    step=0.5,
)

panel_tilt_deg = st.sidebar.slider(
    "Panel tilt (degrees)",
    min_value=0,
    max_value=60,
    value=25,
    step=1,
)

orientation_choice = st.sidebar.selectbox(
    "Panel direction",
    [
        "South",
        "Southeast",
        "Southwest",
        "East",
        "West",
        "Flat / horizontal",
        "Custom azimuth",
    ],
)

orientation_to_azimuth = {
    "South": 0,
    "Southeast": -45,
    "Southwest": 45,
    "East": -90,
    "West": 90,
    "Flat / horizontal": 0,
}

if orientation_choice == "Custom azimuth":
    panel_azimuth_openmeteo = st.sidebar.slider(
        "Custom azimuth: 0=south, -90=east, 90=west",
        min_value=-180,
        max_value=180,
        value=0,
        step=5,
    )
else:
    panel_azimuth_openmeteo = orientation_to_azimuth[orientation_choice]

if orientation_choice == "Flat / horizontal":
    panel_tilt_deg = 0

forecast_hours = st.sidebar.selectbox(
    "Forecast length",
    options=[24, 48, 72],
    index=1,
)

electricity_rate = st.sidebar.number_input(
    "Electricity price ($/kWh)",
    min_value=0.01,
    max_value=1.00,
    value=0.15,
    step=0.01,
)

current_home_load_kw = st.sidebar.number_input(
    "Assumed current home load (kW)",
    min_value=0.1,
    max_value=20.0,
    value=2.5,
    step=0.1,
)

derate = st.sidebar.slider(
    "System derate / performance ratio",
    min_value=0.60,
    max_value=1.00,
    value=0.86,
    step=0.01,
)

temp_coeff_per_c = st.sidebar.slider(
    "Temperature coefficient (% per °C)",
    min_value=-0.60,
    max_value=-0.10,
    value=-0.35,
    step=0.01,
) / 100.0

st.sidebar.divider()

show_noisy = st.sidebar.checkbox(
    "Show inverter-like noisy curve",
    value=True,
)

noise_resolution = st.sidebar.selectbox(
    "Noisy curve resolution",
    [
        "1 minute",
        "10 second",
        "1 second (first 24 hours, slower)",
    ],
    index=0,
)

noise_strength = st.sidebar.slider(
    "Noise strength",
    min_value=0.2,
    max_value=3.0,
    value=1.4,
    step=0.1,
)


# -----------------------------
# Main app
# -----------------------------
st.title("Texas Home Solar Forecast")
st.write(
    "Enter a Texas ZIP code and basic rooftop solar settings. "
    "The app estimates current and forecasted home-scale solar generation."
)

try:
    loc = zip_to_location(zip_code)

    raw_data, df_raw, data_mode = fetch_open_meteo_forecast(
        lat=loc["lat"],
        lon=loc["lon"],
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_openmeteo=panel_azimuth_openmeteo,
        forecast_hours=forecast_hours,
    )

    df = finalize_forecast_dataframe(
        df=df_raw,
        lat=loc["lat"],
        lon=loc["lon"],
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_openmeteo=panel_azimuth_openmeteo,
        system_size_kw_dc=system_size_kw_dc,
        inverter_size_kw_ac=inverter_size_kw_ac,
        derate=derate,
        temp_coeff_per_c=temp_coeff_per_c,
    )

    sim_df = None

    if show_noisy:
        sim_df = simulate_inverter_like_noise(
            df=df,
            system_size_kw_dc=system_size_kw_dc,
            inverter_size_kw_ac=inverter_size_kw_ac,
            derate=derate,
            temp_coeff_per_c=temp_coeff_per_c,
            resolution_label=noise_resolution,
            noise_strength=noise_strength,
            seed=42,
        )

    now_local = pd.Timestamp.now(tz=TIMEZONE)
    nearest_idx = (df["time"] - now_local).abs().idxmin()

    current_kw = float(df.loc[nearest_idx, "power_forecast_kW"])

    if sim_df is not None and not sim_df.empty:
        nearest_sim_idx = (sim_df["time"] - now_local).abs().idxmin()
        current_noisy_kw = float(sim_df.loc[nearest_sim_idx, "power_noisy_kW"])
    else:
        current_noisy_kw = current_kw

    start_time = df["time"].min()
    next24_end = start_time + pd.Timedelta(hours=24)

    next24_kwh = df[df["time"] <= next24_end]["energy_kWh"].sum()
    next24_value = next24_kwh * electricity_rate

    status, explanation, coverage, value = homeowner_summary_text(
        current_kw=current_noisy_kw,
        next24_kwh=next24_kwh,
        electricity_rate=electricity_rate,
        current_home_load_kw=current_home_load_kw,
    )

    daily = (
        df.groupby("date")
        .agg(
            forecast_energy_kWh=("energy_kWh", "sum"),
            peak_power_kW=("power_forecast_kW", "max"),
            avg_cloud_cover_percent=("cloud_cover_percent", "mean"),
            avg_air_temp_C=("air_temp_C", "mean"),
            cloud_loss_kWh=("cloud_loss_kW", lambda x: float(np.sum(x * df.loc[x.index, "step_hours"]))),
            temperature_loss_kWh=("temperature_loss_kW", lambda x: float(np.sum(x * df.loc[x.index, "step_hours"]))),
        )
        .reset_index()
    )

    daily["estimated_value_$"] = daily["forecast_energy_kWh"] * electricity_rate

    st.subheader(f"{loc['city']}, Texas {loc['zip_code']}")

    st.caption(
        f"Data mode: {data_mode}. "
        "Forecast-based estimate, not measured inverter telemetry."
    )

    metric_cols = st.columns(5)

    metric_cols[0].metric(
        "Current estimated power",
        f"{current_noisy_kw:.2f} kW"
    )

    metric_cols[1].metric(
        "Next 24h energy",
        f"{next24_kwh:.1f} kWh"
    )

    metric_cols[2].metric(
        "Next 24h value",
        f"${next24_value:.2f}"
    )

    metric_cols[3].metric(
        "Home load covered now",
        f"{coverage:.0f}%"
    )

    metric_cols[4].metric(
        "System size",
        f"{system_size_kw_dc:.1f} kW DC"
    )

    st.info(
        f"**Homeowner summary:** The system is showing **{status}** right now. "
        f"{explanation}"
    )

    left, right = st.columns([1.35, 0.85])

    with left:
        st.plotly_chart(
            make_power_plot(df, sim_df),
            use_container_width=True
        )

    with right:
        st.write("### Site map")
        map_df = pd.DataFrame({
            "lat": [loc["lat"]],
            "lon": [loc["lon"]],
        })
        st.map(
            map_df,
            latitude="lat",
            longitude="lon",
            zoom=10,
            size=80,
        )

        st.write("### System settings used")
        st.write(
            pd.DataFrame(
                {
                    "Setting": [
                        "ZIP code",
                        "Latitude",
                        "Longitude",
                        "DC system size",
                        "AC inverter size",
                        "Panel tilt",
                        "Open-Meteo azimuth",
                        "Derate",
                        "Electricity rate",
                    ],
                    "Value": [
                        loc["zip_code"],
                        f"{loc['lat']:.5f}",
                        f"{loc['lon']:.5f}",
                        f"{system_size_kw_dc:.1f} kW",
                        f"{inverter_size_kw_ac:.1f} kW",
                        f"{panel_tilt_deg}°",
                        f"{panel_azimuth_openmeteo}°",
                        f"{derate:.2f}",
                        f"${electricity_rate:.2f}/kWh",
                    ],
                }
            )
        )

    st.plotly_chart(
        make_loss_plot(df),
        use_container_width=True
    )

    st.write("### Daily forecast summary")
    st.dataframe(
        daily.round(2),
        use_container_width=True,
    )

    with st.expander("Show detailed forecast table"):
        detailed_cols = [
            "time",
            "GTI_W_m2",
            "GHI_W_m2",
            "air_temp_C",
            "cloud_cover_percent",
            "solar_elevation_deg",
            "power_clear_sky_kW",
            "power_forecast_kW",
            "cloud_loss_kW",
            "temperature_loss_kW",
            "energy_kWh",
        ]

        st.dataframe(
            df[detailed_cols].round(3),
            use_container_width=True,
        )

    csv_forecast = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download forecast CSV",
        data=csv_forecast,
        file_name=f"texas_home_solar_forecast_{loc['zip_code']}.csv",
        mime="text/csv",
    )

    if sim_df is not None and not sim_df.empty:
        with st.expander("Show noisy inverter-like simulation table"):
            st.dataframe(
                sim_df.head(5000).round(3),
                use_container_width=True,
            )

        csv_sim = sim_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download noisy simulation CSV",
            data=csv_sim,
            file_name=f"texas_home_solar_noisy_simulation_{loc['zip_code']}.csv",
            mime="text/csv",
        )

    st.warning(
        "This app estimates solar generation from weather-model irradiance. "
        "It is useful for homeowner-scale forecasting, but it is not a replacement "
        "for true inverter telemetry from Enphase, SolarEdge, Tesla, SMA, Fronius, or a smart meter."
    )

except Exception as e:
    st.error(str(e))
    st.stop()