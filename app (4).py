import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize_scalar
import requests
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Latin America Historical Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.markdown("## By Tyler Brito Brito")
st.title("📊 Latin America Historical Data Analysis")
st.markdown("### Polynomial Regression Analysis for the Wealthiest Latin American Countries")

# Constants
COUNTRIES = {
    "Brazil": "BR",
    "Mexico": "MX", 
    "Argentina": "AR"
}

DATA_CATEGORIES = {
    "Population": "SP.POP.TOTL",
    "Unemployment rate": "SL.UEM.TOTL.ZS",
    "Education levels": "SE.TER.ENRR",  # Tertiary education enrollment
    "Life expectancy": "SP.DYN.LE00.IN",
    "GDP per capita": "NY.GDP.PCAP.CD",  # Average wealth proxy
    "GNI per capita": "NY.GNP.PCAP.CD",  # Average income
    "Birth rate": "SP.DYN.CBRT.IN",
    "Net migration": "SM.POP.NETM",  # Immigration out proxy
    "Intentional homicides": "VC.IHR.PSRC.P5"  # Murder rate
}

@st.cache_data(ttl=3600)
def fetch_world_bank_data(country_code, indicator, start_year=1960, end_year=2023):
    """Fetch data from World Bank API"""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    params = {
        "date": f"{start_year}:{end_year}",
        "format": "json",
        "per_page": 1000
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if len(data) > 1 and data[1]:
            df_data = []
            for item in data[1]:
                if item['value'] is not None:
                    df_data.append({
                        'year': int(item['date']),
                        'value': float(item['value']),
                        'country': item['country']['value']
                    })
            return pd.DataFrame(df_data).sort_values('year')
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def create_polynomial_regression(x, y, degree=3):
    """Create polynomial regression model"""
    if len(x) < degree + 1:
        degree = max(1, len(x) - 1)
    
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(x.reshape(-1, 1), y)
    
    # Get coefficients
    coefficients = poly_model.named_steps['linearregression'].coef_
    intercept = poly_model.named_steps['linearregression'].intercept_
    
    return poly_model, coefficients, intercept, degree

def get_polynomial_equation(coefficients, intercept, degree):
    """Generate polynomial equation string"""
    equation = f"y = {intercept:.4f}"
    for i, coef in enumerate(coefficients[1:], 1):
        if coef >= 0:
            equation += f" + {coef:.4f}x^{i}"
        else:
            equation += f" - {abs(coef):.4f}x^{i}"
    return equation

def analyze_function(model, x_range, country, category, years):
    """Perform function analysis on the polynomial model"""
    x_extended = np.linspace(min(x_range), max(x_range), 1000)
    y_pred = model.predict(x_extended.reshape(-1, 1))
    
    # Calculate derivatives for rate analysis
    dx = x_extended[1] - x_extended[0]
    dy_dx = np.gradient(y_pred, dx)  # First derivative (rate of change)
    d2y_dx2 = np.gradient(dy_dx, dx)  # Second derivative (acceleration)
    
    analysis_results = {}
    
    # Find local maxima and minima
    local_maxima = []
    local_minima = []
    
    for i in range(1, len(d2y_dx2) - 1):
        if abs(dy_dx[i]) < 0.001:  # Near zero first derivative
            if d2y_dx2[i] < 0:  # Negative second derivative = maximum
                year = int(years[0] + x_extended[i])
                value = y_pred[i]
                local_maxima.append((year, value))
            elif d2y_dx2[i] > 0:  # Positive second derivative = minimum
                year = int(years[0] + x_extended[i])
                value = y_pred[i]
                local_minima.append((year, value))
    
    # Find where function increases/decreases fastest
    max_increase_idx = np.argmax(dy_dx)
    max_decrease_idx = np.argmin(dy_dx)
    
    max_increase_year = int(years[0] + x_extended[max_increase_idx])
    max_decrease_year = int(years[0] + x_extended[max_decrease_idx])
    max_increase_rate = dy_dx[max_increase_idx]
    max_decrease_rate = dy_dx[max_decrease_idx]
    
    # Domain and range
    domain = (int(min(years)), int(max(years)))
    range_vals = (float(min(y_pred)), float(max(y_pred)))
    
    analysis_results = {
        'local_maxima': local_maxima,
        'local_minima': local_minima,
        'max_increase_year': max_increase_year,
        'max_decrease_year': max_decrease_year,
        'max_increase_rate': max_increase_rate,
        'max_decrease_rate': max_decrease_rate,
        'domain': domain,
        'range': range_vals,
        'derivatives': (dy_dx, d2y_dx2),
        'x_extended': x_extended,
        'y_pred': y_pred
    }
    
    return analysis_results

def get_historical_context(country, category, year):
    """Provide historical context for significant changes"""
    contexts = {
        "Brazil": {
            1964: "Military coup and beginning of military dictatorship",
            1985: "End of military dictatorship, return to democracy",
            1994: "Introduction of Real Plan, economic stabilization",
            2003: "Beginning of Lula administration, social programs expansion",
            2016: "Political crisis and impeachment of President Dilma"
        },
        "Mexico": {
            1982: "Mexican debt crisis and economic recession",
            1994: "NAFTA implementation and Peso crisis",
            2000: "End of PRI's 71-year rule, democratic transition",
            2006: "Beginning of Drug War",
            2018: "AMLO presidency begins"
        },
        "Argentina": {
            1976: "Military coup and 'Dirty War' period",
            1983: "Return to democracy",
            1991: "Currency board and peso-dollar parity",
            2001: "Economic crisis and currency devaluation",
            2015: "End of Kirchner era, Macri presidency"
        }
    }
    
    country_events = contexts.get(country, {})
    closest_year = min(country_events.keys(), key=lambda x: abs(x - year), default=None)
    
    if closest_year and abs(closest_year - year) <= 3:
        return f"This period coincides with {country_events[closest_year]} ({closest_year})"
    
    return "No specific historical event identified for this period"

# Sidebar controls
st.sidebar.header("Analysis Configuration")

# Data category selection
selected_category = st.sidebar.selectbox(
    "Select Data Category:",
    list(DATA_CATEGORIES.keys())
)

# Country selection
selected_countries = st.sidebar.multiselect(
    "Select Countries:",
    list(COUNTRIES.keys()),
    default=list(COUNTRIES.keys())
)

# Time increment selection
time_increment = st.sidebar.slider(
    "Time Increment (years):",
    min_value=1,
    max_value=10,
    value=1
)

# Polynomial degree selection
poly_degree = st.sidebar.slider(
    "Polynomial Degree:",
    min_value=3,
    max_value=8,
    value=3
)

# Extrapolation years
extrapolation_years = st.sidebar.slider(
    "Extrapolation Years:",
    min_value=0,
    max_value=30,
    value=10
)

# Analysis options
show_comparison = st.sidebar.checkbox("Show Multi-Country Comparison", value=True)
show_function_analysis = st.sidebar.checkbox("Show Function Analysis", value=True)
printer_friendly = st.sidebar.checkbox("Printer Friendly Format")

# Main content
if selected_countries:
    # Fetch and display data
    all_data = {}
    
    with st.spinner("Fetching historical data..."):
        for country in selected_countries:
            country_code = COUNTRIES[country]
            indicator = DATA_CATEGORIES[selected_category]
            
            data = fetch_world_bank_data(country_code, indicator)
            if not data.empty:
                # Filter to last 70 years and apply time increment
                current_year = datetime.now().year
                start_year = current_year - 70
                data = data[data['year'] >= start_year]
                
                # Apply time increment
                if time_increment > 1:
                    data = data[data['year'] % time_increment == 0]
                
                all_data[country] = data
    
    if all_data:
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data & Regression", "📈 Function Analysis", "🔮 Predictions", "📋 Raw Data", "⚙️ Requirements"])
        
        with tab1:
            if show_comparison and len(selected_countries) > 1:
                st.subheader("Multi-Country Comparison")
                
                fig = go.Figure()
                
                for country, data in all_data.items():
                    if len(data) >= poly_degree + 1:
                        x = data['year'].values - data['year'].min()
                        y = data['value'].values
                        years = data['year'].values
                        
                        # Create model
                        model, coefficients, intercept, actual_degree = create_polynomial_regression(x, y, poly_degree)
                        
                        # Create prediction line
                        x_pred = np.linspace(0, max(x), 100)
                        y_pred = model.predict(x_pred.reshape(-1, 1))
                        
                        # Add scatter plot
                        fig.add_trace(go.Scatter(
                            x=years,
                            y=y,
                            mode='markers',
                            name=f'{country} (Data)',
                            marker=dict(size=8)
                        ))
                        
                        # Add regression line
                        fig.add_trace(go.Scatter(
                            x=years.min() + x_pred,
                            y=y_pred,
                            mode='lines',
                            name=f'{country} (Regression)',
                            line=dict(width=3)
                        ))
                        
                        # Add extrapolation if requested
                        if extrapolation_years > 0:
                            x_extrap = np.linspace(max(x), max(x) + extrapolation_years, 50)
                            y_extrap = model.predict(x_extrap.reshape(-1, 1))
                            
                            fig.add_trace(go.Scatter(
                                x=years.min() + x_extrap,
                                y=y_extrap,
                                mode='lines',
                                name=f'{country} (Projection)',
                                line=dict(width=2, dash='dash')
                            ))
                
                fig.update_layout(
                    title=f"{selected_category} Comparison Across Countries",
                    xaxis_title="Year",
                    yaxis_title=selected_category,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Single country analysis
                for country, data in all_data.items():
                    if len(data) >= poly_degree + 1:
                        st.subheader(f"{selected_category} Analysis for {country}")
                        
                        x = data['year'].values - data['year'].min()
                        y = data['value'].values
                        years = data['year'].values
                        
                        # Create model
                        model, coefficients, intercept, actual_degree = create_polynomial_regression(x, y, poly_degree)
                        
                        # Display equation
                        equation = get_polynomial_equation(coefficients, intercept, actual_degree)
                        st.markdown(f"**Regression Equation:** {equation}")
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Scatter plot
                        fig.add_trace(go.Scatter(
                            x=years,
                            y=y,
                            mode='markers',
                            name='Historical Data',
                            marker=dict(size=10, color='blue')
                        ))
                        
                        # Regression line
                        x_pred = np.linspace(0, max(x), 100)
                        y_pred = model.predict(x_pred.reshape(-1, 1))
                        
                        fig.add_trace(go.Scatter(
                            x=years.min() + x_pred,
                            y=y_pred,
                            mode='lines',
                            name='Regression Curve',
                            line=dict(width=3, color='red')
                        ))
                        
                        # Extrapolation
                        if extrapolation_years > 0:
                            x_extrap = np.linspace(max(x), max(x) + extrapolation_years, 50)
                            y_extrap = model.predict(x_extrap.reshape(-1, 1))
                            
                            fig.add_trace(go.Scatter(
                                x=years.min() + x_extrap,
                                y=y_extrap,
                                mode='lines',
                                name='Future Projection',
                                line=dict(width=3, color='orange', dash='dash')
                            ))
                        
                        fig.update_layout(
                            title=f"{selected_category} in {country}",
                            xaxis_title="Year",
                            yaxis_title=selected_category,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if show_function_analysis:
                st.subheader("Function Analysis")
                
                for country, data in all_data.items():
                    if len(data) >= poly_degree + 1:
                        st.write(f"### {country}")
                        
                        x = data['year'].values - data['year'].min()
                        y = data['value'].values
                        years = data['year'].values
                        
                        model, _, _, _ = create_polynomial_regression(x, y, poly_degree)
                        analysis = analyze_function(model, x, country, selected_category, years)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Local Maxima:**")
                            for year, value in analysis['local_maxima'][:3]:  # Show top 3
                                context = get_historical_context(country, selected_category, year)
                                st.write(f"- {selected_category} reached a local maximum in {year} with value {value:.2f}")
                                st.write(f"  *{context}*")
                            
                            st.write("**Local Minima:**")
                            for year, value in analysis['local_minima'][:3]:  # Show top 3
                                context = get_historical_context(country, selected_category, year)
                                st.write(f"- {selected_category} reached a local minimum in {year} with value {value:.2f}")
                                st.write(f"  *{context}*")
                        
                        with col2:
                            st.write("**Rate Analysis:**")
                            st.write(f"- Fastest increase occurred around {analysis['max_increase_year']}")
                            st.write(f"  Rate: {analysis['max_increase_rate']:.4f} units per year")
                            
                            st.write(f"- Fastest decrease occurred around {analysis['max_decrease_year']}")
                            st.write(f"  Rate: {analysis['max_decrease_rate']:.4f} units per year")
                            
                            st.write("**Domain and Range:**")
                            st.write(f"- Domain: {analysis['domain'][0]} to {analysis['domain'][1]}")
                            st.write(f"- Range: {analysis['range'][0]:.2f} to {analysis['range'][1]:.2f}")
        
        with tab3:
            st.subheader("Predictions and Calculations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Interpolation/Extrapolation")
                selected_country_pred = st.selectbox("Select Country for Prediction:", selected_countries)
                prediction_year = st.number_input("Enter Year for Prediction:", 
                                                min_value=1950, 
                                                max_value=2100, 
                                                value=2030)
                
                if selected_country_pred in all_data:
                    data = all_data[selected_country_pred]
                    if len(data) >= poly_degree + 1:
                        x = data['year'].values - data['year'].min()
                        y = data['value'].values
                        years = data['year'].values
                        
                        model, _, _, _ = create_polynomial_regression(x, y, poly_degree)
                        
                        pred_x = prediction_year - years.min()
                        predicted_value = model.predict([[pred_x]])[0]
                        
                        prediction_type = "Extrapolation" if prediction_year > years.max() or prediction_year < years.min() else "Interpolation"
                        
                        st.success(f"**{prediction_type} Result:**")
                        st.write(f"Predicted {selected_category} for {selected_country_pred} in {prediction_year}: **{predicted_value:.2f}**")
            
            with col2:
                st.write("#### Average Rate of Change")
                year1 = st.number_input("Start Year:", min_value=1950, max_value=2100, value=2000)
                year2 = st.number_input("End Year:", min_value=1950, max_value=2100, value=2020)
                
                if selected_country_pred in all_data and year2 > year1:
                    data = all_data[selected_country_pred]
                    if len(data) >= poly_degree + 1:
                        x = data['year'].values - data['year'].min()
                        y = data['value'].values
                        years = data['year'].values
                        
                        model, _, _, _ = create_polynomial_regression(x, y, poly_degree)
                        
                        pred_x1 = year1 - years.min()
                        pred_x2 = year2 - years.min()
                        
                        value1 = model.predict([[pred_x1]])[0]
                        value2 = model.predict([[pred_x2]])[0]
                        
                        avg_rate = (value2 - value1) / (year2 - year1)
                        
                        st.success(f"**Average Rate of Change:**")
                        st.write(f"From {year1} to {year2}: **{avg_rate:.4f}** units per year")
                        
                        # Provide context
                        if avg_rate > 0:
                            st.write(f"The {selected_category} increased on average by {abs(avg_rate):.4f} units per year during this period.")
                        else:
                            st.write(f"The {selected_category} decreased on average by {abs(avg_rate):.4f} units per year during this period.")
        
        with tab4:
            st.subheader("Raw Data")
            
            for country, data in all_data.items():
                st.write(f"### {country} - {selected_category}")
                
                if not data.empty:
                    # Make data editable
                    edited_data = st.data_editor(
                        data,
                        use_container_width=True,
                        num_rows="dynamic",
                        key=f"data_editor_{country}"
                    )
                    
                    # Option to download data
                    csv = edited_data.to_csv(index=False)
                    st.download_button(
                        label=f"Download {country} data as CSV",
                        data=csv,
                        file_name=f"{country}_{selected_category.replace(' ', '_')}_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning(f"No data available for {country}")
        
        with tab5:
            st.subheader("Requirements for Deployment")
            
            st.markdown("""
            ### 📋 Requirements.txt File
            Copy the following content to create a `requirements.txt` file for deploying this app on Streamlit Cloud:
            """)
            
            requirements_content = """streamlit
pandas
numpy
plotly
scikit-learn
scipy
requests"""
            
            st.code(requirements_content, language="text")
            
            # Download button for requirements
            st.download_button(
                label="📥 Download requirements.txt",
                data=requirements_content,
                file_name="requirements.txt",
                mime="text/plain"
            )
            
            st.markdown("---")
            st.markdown("""
            ### 🚀 Deployment Instructions
            1. **GitHub Setup:**
               - Create a new repository on GitHub
               - Upload both `app.py` and `requirements.txt` to the root directory
            
            2. **Streamlit Cloud Deployment:**
               - Visit [share.streamlit.io](https://share.streamlit.io)
               - Connect your GitHub account
               - Select your repository
               - Choose `app.py` as the main file
               - Click "Deploy"
            
            3. **Dependencies:**
               - The app will automatically install all packages from `requirements.txt`
               - No additional configuration needed
            """)
        
        # Printer-friendly option
        if printer_friendly:
            st.markdown("---")
            st.subheader("📄 Printer-Friendly Summary")
            
            for country, data in all_data.items():
                if len(data) >= poly_degree + 1:
                    st.write(f"**{country} - {selected_category} Analysis**")
                    
                    x = data['year'].values - data['year'].min()
                    y = data['value'].values
                    years = data['year'].values
                    
                    model, coefficients, intercept, actual_degree = create_polynomial_regression(x, y, poly_degree)
                    equation = get_polynomial_equation(coefficients, intercept, actual_degree)
                    
                    st.write(f"- Data Period: {years.min()} to {years.max()}")
                    st.write(f"- Regression Equation: {equation}")
                    st.write(f"- Number of Data Points: {len(data)}")
                    
                    # Function analysis summary
                    analysis = analyze_function(model, x, country, selected_category, years)
                    if analysis['local_maxima']:
                        max_year, max_val = analysis['local_maxima'][0]
                        st.write(f"- Primary Maximum: {max_val:.2f} in {max_year}")
                    if analysis['local_minima']:
                        min_year, min_val = analysis['local_minima'][0]
                        st.write(f"- Primary Minimum: {min_val:.2f} in {min_year}")
                    
                    st.write("")

    else:
        st.warning("No data available for the selected countries and category. Please try different selections.")

else:
    st.info("Please select at least one country to begin the analysis.")

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** World Bank Open Data  
**Note:** This application uses polynomial regression for trend analysis. 
Extrapolations should be interpreted with caution as they may not account for future policy changes or external factors.
""")
