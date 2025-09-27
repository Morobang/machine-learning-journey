import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import json
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="BMI Calculator & Health Tracker",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules (these will be created in src/)
try:
    from src.bmi_calculator import BMICalculator
    from src.health_insights import HealthInsights
    from src.data_manager import DataManager
    from src.visualizations import BMIVisualizations
except ImportError:
    st.error("Please ensure all source files are in the 'src/' directory")
    st.stop()

class BMIApp:
    """Main BMI Calculator Application"""
    
    def __init__(self):
        self.calculator = BMICalculator()
        self.insights = HealthInsights()
        self.data_manager = DataManager()
        self.viz = BMIVisualizations()
        
        # Initialize session state
        if 'user_data' not in st.session_state:
            st.session_state.user_data = self.data_manager.load_user_data()
    
    def main(self):
        """Main application function"""
        # Sidebar for navigation
        self.render_sidebar()
        
        # Main content area
        page = st.session_state.get('page', 'Calculator')
        
        if page == 'Calculator':
            self.calculator_page()
        elif page == 'Tracking':
            self.tracking_page()
        elif page == 'Health Info':
            self.health_info_page()
        elif page == 'About':
            self.about_page()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("🏃‍♂️ BMI Health Tracker")
        st.sidebar.markdown("---")
        
        # Navigation
        pages = ['Calculator', 'Tracking', 'Health Info', 'About']
        selected_page = st.sidebar.selectbox("Navigate to:", pages)
        st.session_state.page = selected_page
        
        st.sidebar.markdown("---")
        
        # Quick stats
        if st.session_state.user_data:
            st.sidebar.subheader("📊 Quick Stats")
            latest_record = st.session_state.user_data[-1]
            st.sidebar.metric("Latest BMI", f"{latest_record['bmi']:.1f}")
            st.sidebar.metric("Category", latest_record['category'])
            st.sidebar.metric("Records", len(st.session_state.user_data))
    
    def calculator_page(self):
        """BMI Calculator main page"""
        st.title("🏃‍♂️ BMI Calculator & Health Tracker")
        st.markdown("Calculate your Body Mass Index and get personalized health insights")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📏 Enter Your Measurements")
            
            # Unit system selection
            unit_system = st.radio("Unit System:", ["Metric (kg, cm)", "Imperial (lbs, ft/in)"])
            
            if unit_system == "Metric (kg, cm)":
                # Metric inputs
                weight = st.number_input("Weight (kg):", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
                height = st.number_input("Height (cm):", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
                height_m = height / 100  # Convert to meters
                weight_kg = weight
            else:
                # Imperial inputs
                weight_lbs = st.number_input("Weight (lbs):", min_value=44.0, max_value=660.0, value=154.0, step=0.1)
                
                col1a, col1b = st.columns(2)
                with col1a:
                    feet = st.number_input("Height (feet):", min_value=3, max_value=8, value=5, step=1)
                with col1b:
                    inches = st.number_input("Height (inches):", min_value=0, max_value=11, value=7, step=1)
                
                # Convert to metric
                height_m = ((feet * 12) + inches) * 0.0254
                weight_kg = weight_lbs * 0.453592
            
            # Additional info
            age = st.number_input("Age (optional):", min_value=1, max_value=120, value=30, step=1)
            gender = st.selectbox("Gender (optional):", ["Not specified", "Male", "Female", "Other"])
            
            # Calculate BMI button
            if st.button("🧮 Calculate BMI", type="primary"):
                bmi_result = self.calculator.calculate_bmi(weight_kg, height_m)
                st.session_state.current_bmi = bmi_result
        
        with col2:
            # Display results
            if hasattr(st.session_state, 'current_bmi'):
                result = st.session_state.current_bmi
                
                st.subheader("📊 Your Results")
                
                # BMI value with large display
                st.metric("Your BMI", f"{result['bmi']:.1f}")
                
                # Category with color coding
                category_colors = {
                    'Underweight': '#3498db',
                    'Normal weight': '#2ecc71',
                    'Overweight': '#f39c12',
                    'Obesity Class I': '#e74c3c',
                    'Obesity Class II': '#c0392b',
                    'Obesity Class III': '#8b0000'
                }
                
                category = result['category']
                color = category_colors.get(category, '#333333')
                
                st.markdown(f"<h3 style='color: {color};'>📋 {category}</h3>", unsafe_allow_html=True)
                
                # BMI gauge chart
                fig = self.viz.create_bmi_gauge(result['bmi'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Health insights
                insights = self.insights.get_insights(result['bmi'], age, gender)
                
                st.subheader("💡 Health Insights")
                for insight in insights:
                    st.info(insight)
                
                # Save data button
                if st.button("💾 Save to History"):
                    self.save_current_record(result, age, gender)
    
    def save_current_record(self, bmi_result, age, gender):
        """Save current BMI record to history"""
        record = {
            'date': datetime.now().isoformat(),
            'bmi': bmi_result['bmi'],
            'category': bmi_result['category'],
            'age': age,
            'gender': gender
        }
        
        st.session_state.user_data.append(record)
        self.data_manager.save_user_data(st.session_state.user_data)
        st.success("✅ Record saved to history!")
        st.rerun()
    
    def tracking_page(self):
        """BMI tracking and history page"""
        st.title("📈 BMI Tracking Dashboard")
        
        if not st.session_state.user_data:
            st.info("🔍 No tracking data available. Calculate your BMI first to start tracking!")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(st.session_state.user_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", len(df))
        
        with col2:
            latest_bmi = df.iloc[-1]['bmi']
            st.metric("🎯 Latest BMI", f"{latest_bmi:.1f}")
        
        with col3:
            if len(df) > 1:
                bmi_change = df.iloc[-1]['bmi'] - df.iloc[-2]['bmi']
                st.metric("📈 Change", f"{bmi_change:+.1f}", delta=f"{bmi_change:+.1f}")
            else:
                st.metric("📈 Change", "N/A")
        
        with col4:
            avg_bmi = df['bmi'].mean()
            st.metric("📊 Average BMI", f"{avg_bmi:.1f}")
        
        # BMI history chart
        st.subheader("📈 BMI History")
        fig = self.viz.create_bmi_timeline(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Category Distribution")
            fig = self.viz.create_category_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Recent Records")
            recent_df = df.tail(10).copy()
            recent_df['date'] = recent_df['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(recent_df[['date', 'bmi', 'category']], hide_index=True)
        
        # Data management
        st.subheader("🗂️ Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Data"):
                csv = df.to_csv(index=False)
                st.download_button("💾 Download CSV", csv, "bmi_history.csv", "text/csv")
        
        with col2:
            if st.button("🗑️ Clear All Data", type="secondary"):
                if st.confirm("Are you sure you want to delete all data?"):
                    st.session_state.user_data = []
                    self.data_manager.save_user_data([])
                    st.success("✅ All data cleared!")
                    st.rerun()
    
    def health_info_page(self):
        """Health information and education page"""
        st.title("📚 Health Information & Guidelines")
        
        # BMI Categories explanation
        st.subheader("📊 Understanding BMI Categories")
        
        categories_data = {
            'BMI Range': ['< 18.5', '18.5 - 24.9', '25.0 - 29.9', '30.0 - 34.9', '35.0 - 39.9', '≥ 40.0'],
            'Category': ['Underweight', 'Normal weight', 'Overweight', 'Obesity Class I', 'Obesity Class II', 'Obesity Class III'],
            'Health Risk': ['Increased', 'Minimal', 'Increased', 'High', 'Very High', 'Extremely High']
        }
        
        st.table(pd.DataFrame(categories_data))
        
        # Health tips
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🍎 Nutrition Tips")
            nutrition_tips = [
                "Eat a balanced diet with fruits and vegetables",
                "Control portion sizes",
                "Stay hydrated with water",
                "Limit processed foods and sugary drinks",
                "Include lean proteins in your meals"
            ]
            for tip in nutrition_tips:
                st.write(f"• {tip}")
        
        with col2:
            st.subheader("💪 Exercise Guidelines")
            exercise_tips = [
                "Aim for 150 minutes of moderate activity per week",
                "Include strength training 2-3 times per week",
                "Start slowly and gradually increase intensity",
                "Find activities you enjoy",
                "Consider walking as a great starting point"
            ]
            for tip in exercise_tips:
                st.write(f"• {tip}")
        
        # Important disclaimers
        st.subheader("⚠️ Important Information")
        st.warning("""
        **Please Note:**
        - BMI is a screening tool, not a diagnostic tool
        - It doesn't directly measure body fat or muscle mass
        - Results may not be accurate for athletes, pregnant women, or elderly individuals
        - Always consult with healthcare professionals for personalized medical advice
        - This tool is for educational purposes only
        """)
    
    def about_page(self):
        """About page with project information"""
        st.title("ℹ️ About BMI Health Tracker")
        
        st.markdown("""
        ## 🎯 Project Overview
        This BMI Calculator and Health Tracker is built with Streamlit to provide an easy-to-use tool for 
        calculating Body Mass Index and tracking health metrics over time.
        
        ## 🛠️ Technologies Used
        - **Streamlit**: Web application framework
        - **Plotly**: Interactive visualizations
        - **Pandas**: Data manipulation and analysis
        - **Python**: Core programming language
        
        ## 📊 Features
        - Real-time BMI calculation
        - Health category classification
        - Historical tracking and trends
        - Interactive visualizations
        - Personalized health insights
        - Data export capabilities
        
        ## 🔒 Privacy & Data
        - All data is stored locally on your device
        - No personal information is transmitted to external servers
        - You have full control over your data
        
        ## 🚀 Future Enhancements
        - Integration with fitness trackers
        - Advanced body composition metrics
        - Social features for sharing progress
        - AI-powered health recommendations
        
        ## 📞 Contact & Support
        For questions or suggestions, please refer to the project documentation.
        """)

def main():
    """Main application entry point"""
    app = BMIApp()
    app.main()

if __name__ == "__main__":
    main()