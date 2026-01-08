"""
Streamlit Comparator App

A tool for comparing evaluation runs across different experiments.
Allows side-by-side comparison of generated corrections, auxiliary metrics,
and evaluation results for student submissions.
"""

import streamlit as st
from utils import (
    load_config, 
    get_submissions, 
    get_submission_data, 
    get_available_runs, 
    get_run_data,
    format_timestamp
)


# Page configuration
st.set_page_config(
    page_title="Evaluation Comparator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .run-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .metric-box {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .score-display {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }
    .submission-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #0f3460;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .submission-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.2);
    }
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    /* Ensure text wraps in expanders */
    .stExpander div[data-testid="stMarkdownContainer"] {
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .stExpander p, .stExpander code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'selected_submission' not in st.session_state:
        st.session_state.selected_submission = None
    if 'selected_runs' not in st.session_state:
        st.session_state.selected_runs = []
    if 'config' not in st.session_state:
        st.session_state.config = load_config()


def navigate_to_submission(submission_name: str):
    """Navigate to submission page."""
    st.session_state.current_page = 'submission'
    st.session_state.selected_submission = submission_name
    st.session_state.selected_runs = []


def navigate_to_home():
    """Navigate back to home page."""
    st.session_state.current_page = 'home'
    st.session_state.selected_submission = None
    st.session_state.selected_runs = []


def add_run(experiment_type: str, timestamp: str):
    """Add a run to the comparison."""
    run_key = f"{experiment_type}/{timestamp}"
    if run_key not in st.session_state.selected_runs:
        st.session_state.selected_runs.append(run_key)


def remove_run(run_key: str):
    """Remove a run from the comparison."""
    if run_key in st.session_state.selected_runs:
        st.session_state.selected_runs.remove(run_key)


def render_home_page():
    """Render the home page with submission list."""
    st.markdown("# 📊 Evaluation Comparator")
    st.markdown("### Select a submission to compare evaluation runs")
    
    config = st.session_state.config
    submissions = get_submissions(config)
    
    if not submissions:
        st.warning("No submissions found. Check the assignment_path in streamlit.yaml")
        return
    
    st.markdown(f"**Assignment folder:** `{config['assignment_path']}`")
    st.markdown(f"**Found {len(submissions)} submissions**")
    st.divider()
    
    # Display submissions in a grid
    cols = st.columns(4)
    for i, submission in enumerate(submissions):
        with cols[i % 4]:
            if st.button(
                f"📝 {submission}",
                key=f"sub_{submission}",
                use_container_width=True
            ):
                navigate_to_submission(submission)
                st.rerun()


def render_reference_correction(reference):
    """Render reference correction in a readable format."""
    if reference is None:
        st.info("No reference correction available")
        return
    
    if isinstance(reference, list):
        for i, item in enumerate(reference, 1):
            st.markdown(f"**{i}.** {item}")
    elif isinstance(reference, dict) and 'corrections' in reference:
        for i, item in enumerate(reference['corrections'], 1):
            st.markdown(f"**{i}.** {item}")
    else:
        st.text(str(reference))


def render_run_data(run_data: dict, run_key: str):
    """Render the data for a single run."""
    if run_data is None:
        st.error("Could not load run data")
        return
    
    # Header with remove button
    col1, col2 = st.columns([5, 1])
    with col1:
        exp_type, timestamp = run_key.split('/')
        st.markdown(f"### 🔬 {exp_type}")
        st.caption(f"📅 {format_timestamp(timestamp)}")
    with col2:
        if st.button("❌", key=f"remove_{run_key}"):
            remove_run(run_key)
            st.rerun()
    
    # Tabs for different data types
    tab1, tab2, tab3 = st.tabs(["📝 Corrections", "📊 Metrics", "🎯 Evaluation"])
    
    with tab1:
        corrections = run_data.get('generated_corrections')
        if corrections:
            corr_list = corrections.get('corrections', [])
            if corr_list:
                for corr in corr_list:
                    req = corr.get('requirement', {})
                    with st.expander(f"**{req.get('function', 'Unknown')}** - {req.get('type', '')}", expanded=True):
                        st.markdown(f"**Requirement:** {req.get('requirement', 'N/A')}")
                        st.markdown("**Result:**")
                        st.markdown(corr.get('result', 'No result'))
            else:
                st.info("No corrections in this run")
        else:
            st.info("No corrections data available")
    
    with tab2:
        aux_metrics = run_data.get('auxiliary_metrics')
        if aux_metrics:
            metrics = aux_metrics.get('auxiliary_metrics', aux_metrics)
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    if metric_name != 'timings':
                        with st.expander(f"📈 {metric_name.upper()}", expanded=True):
                            if isinstance(metric_value, str):
                                st.markdown(metric_value)
                            elif isinstance(metric_value, list):
                                for item in metric_value:
                                    st.markdown(f"- {item}")
                            else:
                                st.markdown(f"```json\n{metric_value}\n```")
            else:
                st.json(metrics)
        else:
            st.info("No auxiliary metrics available")
    
    with tab3:
        eval_results = run_data.get('evaluation_results')
        if eval_results:
            # Overall score
            overall = eval_results.get('overall_score')
            if overall is not None:
                st.metric("Overall Score", f"{overall:.3f}")
            
            # Individual scores
            scores = eval_results.get('scores', {})
            if scores:
                st.markdown("**Individual Scores:**")
                score_cols = st.columns(len(scores))
                for i, (name, score) in enumerate(scores.items()):
                    with score_cols[i]:
                        st.metric(name, f"{score:.3f}")
            
            # Explanations
            explanations = eval_results.get('explanations', {})
            if explanations:
                st.markdown("**Explanations:**")
                for name, explanation in explanations.items():
                    with st.expander(f"💡 {name}"):
                        st.markdown(explanation)
            
            # Timings
            timings = eval_results.get('timings', {})
            if timings:
                with st.expander("⏱️ Timings"):
                    for name, time_val in timings.items():
                        st.markdown(f"**{name}:** {time_val:.2f}s")
        else:
            st.info("No evaluation results available")


def render_submission_page():
    """Render the submission detail page."""
    submission_name = st.session_state.selected_submission
    config = st.session_state.config
    
    # Header with back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back", use_container_width=True):
            navigate_to_home()
            st.rerun()
    with col2:
        st.markdown(f"# 📄 Submission: {submission_name}")
    
    # Load submission data
    sub_data = get_submission_data(config, submission_name)
    
    # Sidebar for run selection
    with st.sidebar:
        st.markdown("## 🔍 Add Runs to Compare")
        
        available_runs = get_available_runs(config, submission_name)
        
        if not available_runs:
            st.warning("No runs found for this submission")
        else:
            # Experiment type selector
            exp_type = st.selectbox(
                "Experiment Type",
                options=list(available_runs.keys()),
                key="exp_type_select"
            )
            
            if exp_type:
                timestamps = available_runs[exp_type]
                
                # Timestamp selector
                timestamp = st.selectbox(
                    "Timestamp",
                    options=timestamps,
                    format_func=format_timestamp,
                    key="timestamp_select"
                )
                
                if timestamp:
                    run_key = f"{exp_type}/{timestamp}"
                    if run_key in st.session_state.selected_runs:
                        st.success("✓ Already added")
                    else:
                        if st.button("➕ Add Run", use_container_width=True):
                            add_run(exp_type, timestamp)
                            st.rerun()
        
        st.divider()
        st.markdown("### Selected Runs")
        if st.session_state.selected_runs:
            for run_key in st.session_state.selected_runs:
                exp, ts = run_key.split('/')
                st.caption(f"• {exp[:20]}... / {format_timestamp(ts)}")
        else:
            st.caption("No runs selected")
    
    # Main content area
    tab_code, tab_ref, tab_compare = st.tabs([
        "📝 Submission Code", 
        "✅ Reference Correction",
        "🔄 Compare Runs"
    ])
    
    with tab_code:
        st.markdown("### Student Code")
        if sub_data['code']:
            st.code(sub_data['code'], language='python', line_numbers=True)
        else:
            st.warning("Code file not found")
    
    with tab_ref:
        st.markdown("### Reference Correction")
        render_reference_correction(sub_data['reference_correction'])
        
        if sub_data['assignment']:
            with st.expander("📋 Assignment (Consigna)"):
                st.markdown(sub_data['assignment'])
    
    with tab_compare:
        if not st.session_state.selected_runs:
            st.info("👈 Use the sidebar to add runs for comparison")
        else:
            # Create columns for side-by-side comparison
            num_runs = len(st.session_state.selected_runs)
            cols = st.columns(num_runs)
            
            for i, run_key in enumerate(st.session_state.selected_runs):
                with cols[i]:
                    exp_type, timestamp = run_key.split('/')
                    run_data = get_run_data(config, exp_type, timestamp, submission_name)
                    render_run_data(run_data, run_key)


def main():
    """Main application entry point."""
    init_session_state()
    
    if st.session_state.current_page == 'home':
        render_home_page()
    elif st.session_state.current_page == 'submission':
        render_submission_page()


if __name__ == "__main__":
    main()
